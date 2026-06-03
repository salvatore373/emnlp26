"""
Create a vector DB with the pages of the Knowledge Base that occur in a dataset for Agentic RAG.
This DB will later be searched by the agent.
"""
import io
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from imaplib import Debug
from typing import List

import faiss
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from tools.monaco_preprocessing.monaco_utils import MonacoUtils

# Configuration
MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
BATCH_SIZE = 128
MAX_CHUNK_SIZE = 250
TOKENIZER_MAX_LEN = 8192

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MonacoRAG:
    def __init__(self, model_name=MODEL_NAME, rerank_model=RERANK_MODEL, use_reranker=True, model_replicas: int = None):
        self.use_reranker = use_reranker
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = 10000000 # Increased to avoid sequence length warnings

        # Determine number of model replicas: explicit param > env var > 1
        if model_replicas is None:
            try:
                model_replicas = int(os.environ.get("MODEL_REPLICAS", "1"))
            except Exception:
                model_replicas = 1
        # only allow replicas on CUDA device
        self.model_replicas = model_replicas if self.device == "cuda" else 1

        # Load one or more model instances on the target device
        self.models = [AutoModel.from_pretrained(model_name).to(self.device).half().eval()
                       for _ in range(self.model_replicas)]
        # Keep `self.model` for backward compatibility (single-instance callers)
        self.model = self.models[0]

        # If using CUDA and multiple replicas, create streams to increase concurrency
        self.streams = []
        if self.device == "cuda" and self.model_replicas > 1:
            for _ in range(self.model_replicas):
                self.streams.append(torch.cuda.Stream(device=torch.device(self.device)))

        self.reranker = None
        if self.use_reranker:
            self.reranker = AutoModelForSequenceClassification.from_pretrained(rerank_model).to(
                self.device).half().eval()

        self.index = None
        self.chunks = []
        self.metadata = []

    def _get_detailed_chunks(self, text, max_tokens=MAX_CHUNK_SIZE):
        # Split by paragraph or section first
        blocks = re.split(r'\n+|(?<=[.!?])\s+', text)
        final_chunks = []
        current_chunk_ids = []

        for block in blocks:
            if not block.strip(): continue
            # This call won't warn anymore once model_max_length is set!
            sent_ids = self.tokenizer.encode(block, add_special_tokens=False)

            if len(sent_ids) > max_tokens:
                if current_chunk_ids:
                    final_chunks.append(self.tokenizer.decode(current_chunk_ids))
                    current_chunk_ids = []
                for i in range(0, len(sent_ids), max_tokens):
                    final_chunks.append(self.tokenizer.decode(sent_ids[i: i + max_tokens]))
                continue

            if len(current_chunk_ids) + len(sent_ids) > max_tokens:
                final_chunks.append(self.tokenizer.decode(current_chunk_ids))
                current_chunk_ids = sent_ids
            else:
                current_chunk_ids.extend(sent_ids)

        if current_chunk_ids:
            final_chunks.append(self.tokenizer.decode(current_chunk_ids))
        return [c.strip() for c in final_chunks if c.strip()]

    @torch.no_grad()
    def encode_dense(self, texts, batch_size=BATCH_SIZE, num_workers=None):
        """
        Tokenize batches in threads and run inference concurrently on multiple
        model replicas (if available). Preserves batch order.
        """
        if num_workers is None:
            try:
                cpu_count = os.cpu_count() or 4
            except Exception:
                cpu_count = 4
            num_workers = min(8, max(1, cpu_count - 1))

        batches = [texts[i: i + batch_size] for i in range(0, len(texts), batch_size)]
        num_batches = len(batches)

        tokenize = lambda b: self.tokenizer(
            b,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt"
        )

        # Fast path: single replica or CPU device — keep simple flow
        if self.device != "cuda" or self.model_replicas <= 1:
            all_embeddings = []
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                for inputs in tqdm(ex.map(tokenize, batches), total=len(batches), desc="Tokenizing+Encoding"):
                    inputs = {k: v.pin_memory() for k, v in inputs.items()}
                    inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                    out = self.model(**inputs).last_hidden_state[:, 0]
                    out = torch.nn.functional.normalize(out, p=2, dim=1)
                    all_embeddings.append(out.cpu().numpy())
            return np.vstack(all_embeddings)

        # Concurrent path: tokenize in parallel, submit inference jobs to replica-limited pool
        results = [None] * num_batches

        def _inference_job(batch_idx, inputs, replica_idx):
            # inputs: CPU tensors (possibly pinned) — move to device and run
            inputs = {k: v.pin_memory() for k, v in inputs.items()}
            if self.device == "cuda":
                stream = self.streams[replica_idx]
                with torch.cuda.stream(stream):
                    inputs_gpu = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                    with torch.no_grad():
                        out = self.models[replica_idx](**inputs_gpu).last_hidden_state[:, 0]
                        out = torch.nn.functional.normalize(out, p=2, dim=1)
                    return batch_idx, out.cpu().numpy()
            else:
                inputs_gpu = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = self.models[replica_idx](**inputs_gpu).last_hidden_state[:, 0]
                    out = torch.nn.functional.normalize(out, p=2, dim=1)
                return batch_idx, out.cpu().numpy()

        tok_ex = ThreadPoolExecutor(max_workers=num_workers)
        inf_ex = ThreadPoolExecutor(max_workers=self.model_replicas)

        # submit tokenization tasks
        tok_futs = {tok_ex.submit(tokenize, batches[i]): i for i in range(num_batches)}

        # as tokenization finishes, submit inference jobs
        inf_futs = {}
        for fut in tqdm(as_completed(list(tok_futs.keys())), total=num_batches, desc="Tokenizing"):
            idx = tok_futs[fut]
            inputs = fut.result()
            # schedule on a replica (round-robin by batch idx)
            replica_idx = idx % self.model_replicas
            inf_fut = inf_ex.submit(_inference_job, idx, inputs, replica_idx)
            inf_futs[inf_fut] = idx

        # collect inference results
        for fut in tqdm(as_completed(list(inf_futs.keys())), total=len(inf_futs), desc="Inferring"):
            idx, emb = fut.result()
            results[idx] = emb

        tok_ex.shutdown(wait=True)
        inf_ex.shutdown(wait=True)

        return np.vstack(results)

    def build_index(self, ds_path: str, num_entries_to_process: int = -1):
        """
        Builds the vector DB from the Knowledge Base pages stored in the dataset that `ds_path` points to.

        Arguments:
             ds_path (str): The path to the parquet file containing Knowledge Base pages.
             num_entries_to_process (int, optional): The number of entries to process. If -1, all the entries of the
              dataset are processed.
        """
        # Load MoNaCo and parallelize chunk extraction to use multiple CPU cores

        kb_utils = MonacoUtils(path_to_kb=ds_path)
        print('Selected MonacoUtils')
        print("🛠️ Chunking and Indexing...")
        pbar = tqdm(
            total=kb_utils.get_num_kb_entries() if num_entries_to_process == -1 else num_entries_to_process,
            desc="Chunking and Indexing Knowledge Base pages..."
        )
        wiki_cursor = kb_utils.get_kb_entries_cursor(["page_id", "page_url", "page_title"],
                                                         limit=num_entries_to_process)

        # Helper to process a single page (safe to run in threads)
        def _process_row(row):
            page_id, url, page_title = row
            try:
                wiki_retr = kb_utils.retrieve_wiki_page(url, integrate_infobox=True, integrate_lists_tables=True)
                row_chunks = self._get_detailed_chunks(wiki_retr.wiki_page)
                metas = [{
                    "url": url,
                    "page_id": page_id,
                    "title": page_title,
                } for _ in row_chunks]
                return row_chunks, metas, None
            except Exception as e:
                return [], [], e

        # Determine worker count for chunking (leave one CPU free)
        try:
            cpu_count = os.cpu_count() or 4
        except Exception:
            cpu_count = 4
        chunk_workers = min(24, max(1, cpu_count - 1))
        print("Chunking workers:", chunk_workers)

        with ThreadPoolExecutor(max_workers=chunk_workers) as ex:
            # Process pages in fetchmany batches; submit all rows in each batch
            while rows := wiki_cursor.fetchmany(10_000):
                futures = {ex.submit(_process_row, r): r for r in rows}
                for fut in as_completed(futures):
                    row_chunks, metas, err = fut.result()
                    if err:
                        # Keep processing other pages even if one fails
                        continue
                    if row_chunks:
                        self.chunks.extend(row_chunks)
                        self.metadata.extend(metas)
                    pbar.update(1)

        embeddings = self.encode_dense(self.chunks)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))

    def search(self, query, top_k=20, final_k=3):
        # 1. Dense Retrieval (Recall)
        query_emb = self.encode_dense([query]).astype('float32')
        distances, indices = self.index.search(query_emb, top_k)

        candidate_texts = [self.chunks[idx] for idx in indices[0]]
        candidate_meta = [self.metadata[idx] for idx in indices[0]]

        if self.use_reranker and self.reranker is not None:
            # 2. Cross-Encoder Re-Ranking (Precision)
            pairs = [[query, txt] for txt in candidate_texts]
            with torch.no_grad():
                inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt",
                                        max_length=512).to(self.device)
                scores = self.reranker(**inputs).logits.view(-1).float().cpu().numpy()

            # Sort by Re-ranker score
            ranked_indices = np.argsort(scores)[::-1][:final_k]

            return [{
                "text": candidate_texts[i],
                "title": candidate_meta[i]['title'],
                "metadata": candidate_meta[i],
                "score": scores[i]
            } for i in ranked_indices]
        else:
            # Return top results without re-ranking
            ranked_indices = list(range(min(final_k, len(candidate_texts))))
            return [{
                "text": candidate_texts[i],
                "title": candidate_meta[i]['title'],
                "metadata": candidate_meta[i],
                "score": float(distances[0][i])
            } for i in ranked_indices]

    def save(self, path="sota_rag"):
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.meta", "wb") as f:
            pickle.dump({"chunks": self.chunks, "metadata": self.metadata}, f)

    def load(self, path="sota_rag"):
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.meta", "rb") as f:
            data = pickle.load(f)
            self.chunks, self.metadata = data["chunks"], data["metadata"]

    def search_similar_page(self, target_url, top_k=50, final_k=3, urls_to_exclude: List[str] = None):
        """
        Searches for chunks from different pages that are semantically similar
        to the introduction of the target page at the given URL.

        Arguments:
            target_url (str): The URL of the page to take as reference. A page similar to the one at this link will be
             returned.
            top_k (int): The number of most similar passages to retrieve with dense search. This is the number of
             returned pages if reranker is not enabled.
            final_k (int): The number of most similar passages to keep after reranking. This is the number of
             returned pages if reranker is enabled; otherwise, this parameter is ignored.
            urls_to_exclude (List[str]): The returned page's URL must not belong to this list.
        """
        if urls_to_exclude is None:
            urls_to_exclude = []

        # Find the first chunk of the target page (the intro/summary)
        query_text = None
        for i, meta in enumerate(self.metadata):
            if meta["url"] == target_url:
                query_text = self.chunks[i]
                break

        if not query_text:
            raise ValueError(f"Page with URL '{target_url}' not found in the DB.")

        # Dense Retrieval using the intro chunk
        query_emb = self.encode_dense([query_text])
        # We fetch a larger top_k because we will filter out chunks from the target page itself
        distances, indices = self.index.search(query_emb.astype('float32'), top_k)

        # Filter out chunks that belong to the target page
        candidate_texts = []
        candidate_meta = []
        candidate_distances = []

        for i, idx in enumerate(indices[0]):
            if self.metadata[idx]["url"] not in urls_to_exclude:
                candidate_texts.append(self.chunks[idx])
                candidate_meta.append(self.metadata[idx])
                candidate_distances.append(distances[0][i])

        if not candidate_texts:
            return []

        # Cross-Encoder Re-Ranking (Precision)
        if self.use_reranker and self.reranker is not None:
            pairs = [[query_text, txt] for txt in candidate_texts]
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)
                scores = self.reranker(**inputs).logits.view(-1).float().cpu().numpy()

            # Sort by Re-ranker score
            ranked_indices = np.argsort(scores)[::-1][:final_k]

            return [{
                "text": candidate_texts[i],
                "title": candidate_meta[i]['title'],
                "metadata": candidate_meta[i],
                "score": scores[i]
            } for i in ranked_indices]

        else:
            # Return top results without re-ranking
            ranked_indices = list(range(min(final_k, len(candidate_texts))))
            return [{
                "text": candidate_texts[i],
                "title": candidate_meta[i]['title'],
                "url": candidate_meta[i]['url'],
                "metadata": candidate_meta[i],
                "score": float(candidate_distances[i])
            } for i in ranked_indices]


def _build_db(path_to_wiki: str, db_prefix: str, model_replicas: int = None):
    # Initialize and Run
    rag = MonacoRAG(model_replicas=model_replicas)
    rag.build_index(ds_path=path_to_wiki, num_entries_to_process=100 if DEBUG else -1)

    rag.save(db_prefix)
    print(f"DB saved with prefix: {db_prefix}")


def _search_db(db_prefix: str, use_reranker:bool = False):
    print('TESTING WITH STD BGE-m3 LOADING...')

    rag = MonacoRAG(use_reranker=use_reranker)
    rag.load(db_prefix)

    # test_query = "What is the height of the Warsaw Unit building?"
    test_query = "When was Kennedy President of the USA?"

    results = rag.search(test_query, final_k=5)
    for r in results:
        print(f"\n[Rerank Score: {r['score']:.2f}] {r['metadata']['page_id']} {r['metadata']['title']}")
        print(f"Content: {r['text']}...")


if __name__ == "__main__":
    test = 'search_db'  # 'search_db', 'build_db'

    DEBUG = False
    if DEBUG:
        print('DEBUG mode active')

    # MONACO CONSTANTS
    # The path where the database will be saved
    BASE_PATH = "/workspace/tools/monaco_preprocessing/data"
    monaco_wiki_rag_db_path = f"{BASE_PATH}/monaco_wiki_vec_db{'_debug' if DEBUG else ''}"
    db_prefix = f"{monaco_wiki_rag_db_path}/monaco_wiki_olderrev_full"
    os.makedirs(monaco_wiki_rag_db_path, exist_ok=True)
    # The path to the dataset
    path_to_wiki = f"{BASE_PATH}/scraped_wiki_singleurl_olderrev_full/pages_shard_0000_postproc.parquet"

    if test == 'build_db':
        _build_db(path_to_wiki, db_prefix, model_replicas=4)
        _search_db(db_prefix, use_reranker=False)
        print("Execution Complete")
    elif test == 'search_db':
        _search_db(db_prefix, use_reranker=False)
