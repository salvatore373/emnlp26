import gc
from typing import List, Dict, Tuple, Union

from tools.base_web_tool_monaco import BaseWebToolMonaco


class WebSearch(BaseWebToolMonaco):
    """
    This is a tool to simulate a web search. The search will be performed on the provided knowledge base.
    This is to overcome the dynamic nature of the web and ensure consistency between multiple evaluation runs.
    """

    name = "web_search"
    description = (
        "Searches the web for information related to an input query and returns a formatted list of relevant results. "
        "The input is a query string and the output is a string listing the search results."
    )
    output_type = "string"
    inputs = {
        "query": {
            "type": "string",
            "description": "A query string used to retrieve relevant information for the user's request."
        }
    }

    # The number of results to return in the response to the given query
    NUM_RESULTS_PER_QUERY = 10
    # The length of the search result summaries
    SUMMARY_LENGTH = 900
    # Proportions for obfuscated output: weak and hard distractors (must sum to 1.0)
    OBFUSCATED_OUTPUT_WEAK_PERC: float = 0.5
    OBFUSCATED_OUTPUT_HARD_PERC: float = 0.5
    # Positioning of golden passages in the returned results
    # 0 -> golden first, 0.5 -> golden in the middle, 1 -> golden last, <0 -> shuffle
    OBFUSCATED_OUTPUT_GOLDEN_POS: float = 0.5
    # Non-relevant passages whose DE score is greater than this threshold are considered hard distractors
    # Reference: Distracting Effect-DE (Amiraz et al., 2025)
    HARD_DISTRACTOR_DE_THRESHOLD: float = 0.8
    # Non-relevant passages whose DE score is lower than this threshold are considered weak distractors
    # Reference: Distracting Effect-DE (Amiraz et al., 2025)
    WEAK_DISTRACTOR_DE_THRESHOLD: float = 0.2

    def __init__(self, *args, **kwargs):
        """
        Initialize the structure to retrieve Wikipedia pages.
        """

        super().__init__(*args, **kwargs)

        # Initialize the dictionary storing how each used model tokenizes "NO-RESPONSE"
        self._no_resp_first_token: Dict = {}  # model_id -> first token of "NO-RESPONSE"
        # Initialize the tokenizer used to compute the DE score
        self._tokenizer = None

        # Initialize the list of requests and responses provided by this tool (for debugging)
        self.interactions: List[Dict[str, Union[
            str,
            Dict[str, List[Dict[str, Union[str, float]]]],
            List[Dict[str, Union[str, float]]]]]] = []

        # Load external prompts from YAML file (if present). If loading fails, fall back to empty dict.
        import os, yaml
        prompts_path = os.path.join(os.path.dirname(__file__), "web_search_monaco_prompts.yaml")
        if os.path.exists(prompts_path):
            with open(prompts_path, "r", encoding="utf-8") as f:
                self._external_prompts = yaml.safe_load(f) or {}

    @staticmethod
    def _format_response(search_results: List[Dict[str, str]]) -> str:
        """
        Given the list of search results, it returns the string to pass to the agent.
        """
        search_results = [(d['title'], d['url'], d['text'].split('\n')[0]) for d in search_results]
        return '## Search Results:\n' + '\n\n'.join([f"[{d[0]}][{d[1]}]\n{d[2]}..." for d in search_results])

    def _retrieve_relevant_wiki_urls_and_titles(self, query: str, final_k: int = 3, top_k: int = 20, ) -> \
            List[Tuple[str, str]]:
        """
        Returns the URLs and titles of the most relevant Wikipedia pages for the given query.

        Arguments:
            query (str): The query to retrieve relevant Wikipedia pages.
            final_k (int): The number of Wikipedia pages to return after re-ranking, i.e. the number of Wikipedia
             pages to return.
            top_k (int): The number of Wikipedia pages to pass to the re-ranking phase.

        Returns:
            List[Tuple[str, str]]: The URLs and titles of the most relevant Wikipedia pages, sorted by relevance.
        """
        unique_search_urls = set()
        return [
            (r['metadata']['url'], r['metadata']['title'])
            for r in self._wikipedia_rag.search(query, top_k=top_k, final_k=final_k)
            if not (r['metadata']['url'] in unique_search_urls or unique_search_urls.add(r['metadata']['url']))
        ]

    def _group_passages(self, raw_results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Retrieve chunks from the knowledge base and group them by page URL.

        This helper will request a number of chunks equal to `multiplier * self.NUM_RESULTS_PER_QUERY`.

        Returned passages preserve retrieval order and join multiple chunks from the same URL
        with '...' as a separator.

        Arguments:
            raw_results: The results from the retriever.

        Returns:
            A list of dicts with keys `url`, `title`, and `text`.
        """
        # Group chunks by page URL while preserving the retrieval order
        grouped: Dict[str, Dict] = {}
        ordered_urls: List[str] = []
        for r in raw_results:
            url = r.get('url', r.get('metadata', {}).get('url'))
            title = r.get('title')
            text = r.get('text')

            if url in grouped:
                grouped[url]['texts'].append(text)
            else:
                grouped[url] = {'title': title, 'url': url, 'texts': [text]}
                ordered_urls.append(url)

        return [
            {'url': grouped[u]['url'], 'title': grouped[u]['title'], 'text': '...'.join(grouped[u]['texts'])}
            for u in ordered_urls
        ]

    def _truncate_to_sentence(self, text: str, max_len: int) -> str:
        """
        Truncate `text` to be at most `max_len` characters, cutting only at sentence boundaries when possible.
        If no full sentence fits, fall back to the last punctuation within the limit or the last whitespace.
        """
        import re
        if not text:
            return text
        text = text.strip()
        if len(text) <= max_len:
            return text

        # Split into sentences using common sentence-ending punctuation.
        sentences = re.split(r'(?<=[.!?])\s+', text)
        accumulated = ''
        for s in sentences:
            if not s:
                continue
            candidate = f"{accumulated} {s}".strip() if accumulated else s
            if len(candidate) <= max_len:
                accumulated = candidate
            else:
                break

        if accumulated:
            return accumulated

        # No full sentence fits: try to cut at the last sentence-ending punctuation within the limit.
        prefix = text[:max_len]
        last_punc = max(prefix.rfind('.'), prefix.rfind('!'), prefix.rfind('?'))
        if last_punc != -1:
            return prefix[:last_punc + 1].strip()

        # As a last resort, cut at the last whitespace to avoid breaking a word.
        last_space = prefix.rfind(' ')
        if last_space != -1:
            return prefix[:last_space].strip()

        return prefix

    def _retrieve_random_results(self, desired_num_results: int) -> List[Dict[str, str]]:
        # Retrieve some random entries from the dataset of Wikipedia pages
        random_docs = self._monaco_utils.retrieve_random_kb_entries(
            desired_num_results + int(0.5 * desired_num_results), ['page_title', 'page_text', 'page_url'],
        )
        # Put the results in the right format, while checking that none of the pages annotated as golden for this
        # MoNaCo entry are among the `random_docs`.
        res = []
        for ind, d in enumerate(random_docs):
            if d['page_url'] not in self._monaco_entry_golden_w_urls:
                res.append({
                    'title': d['page_title'],
                    'text': self._truncate_to_sentence(d['page_text'], self.SUMMARY_LENGTH), 'url': d['page_url']
                })

            if ind >= desired_num_results:
                break
        return res

    def execute_tool(self, query: str) -> str:
        """
        Return the most relevant results to the query (sorted by relevance).
        """
        # Retrieve and group passages from the retriever
        passages = self._wikipedia_rag.search(query, top_k=self.NUM_RESULTS_PER_QUERY,
                                              final_k=self.NUM_RESULTS_PER_QUERY)
        # passages = self._group_passages(passages)
        for p in passages:
            p['url'] = p.get('url', p.get('metadata', {}).get('url'))

        return WebSearch._format_response([
            {'title': p['title'], 'text': p['text'], 'url': p['url']}
            for p in passages[:self.NUM_RESULTS_PER_QUERY]
        ])
