import threading
from abc import ABC
from typing import Dict, Optional, Set, Literal

from tools.base_noisy_tool import BaseNoisyTool
from tools.monaco_preprocessing.monaco_utils import BaseGoldenPassage

_GLOBAL_WIKIPEDIA_RAG = None
_GLOBAL_MONACO_UTILS = None
_GLOBAL_INIT_LOCK = threading.Lock()


class BaseWebToolMonaco(BaseNoisyTool, ABC):
    """
    This is the base class to build a tool that simulates web search and webpages access with the MoNaCo dataset.
    """

    # The path to the vector DB with the Wikipedia pages
    _PATH_TO_MONACO_WIKI_VEC_DB = \
        "/workspace/tools/monaco_preprocessing/data/monaco_wiki_vec_db/monaco_wiki_olderrev_full"
    # The path to the parquet file containing the Wikipedia dump for MoNaCo
    _PATH_TO_MONACO_WIKI = \
        "/workspace/tools/monaco_preprocessing/data/scraped_wiki_singleurl_olderrev_full/pages_shard_0000_postproc.duckdb"
    # The path to the parquet file containing the mapping between MoNaCo URLs and Wikipedia page URLs
    _PATH_TO_MONACO_TO_WIKI_URLS_MAPPING = "/workspace/tools/monaco_preprocessing/data/scraped_wiki_singleurl_olderrev_full/monaco_to_wiki_urls_mapping.parquet"
    # The path to the parquet file containing the mapping between Wikipedia page URLs and MoNaCo URLs.
    _PATH_TO_WIKI_TO_MONACO_URLS_MAPPING = "/workspace/tools/monaco_preprocessing/data/scraped_wiki_singleurl_olderrev_full/wiki_to_monaco_urls_mapping.parquet"

    SWITCH_HARDER_DISTRACTORS = False

    # The common prefix in the URLs of the entries of the MoNaCo dataset
    _MONACO_PAGES_BASE_URL = "https://en.wikipedia.org/wiki/"

    def __init__(
            self,
            db_nickname: Literal['monaco', 'bcp'] = 'monaco',
            path_to_vec_db: str = None,
            path_to_kb: str = None,
            path_to_ds_to_kb_urls_mapping: str = None,
            path_to_kb_to_ds_urls_mapping: str = None,
    ):
        """
        Initialize the structure to retrieve Wikipedia pages.
        """
        global _GLOBAL_WIKIPEDIA_RAG, _GLOBAL_MONACO_UTILS
        from tools.monaco_preprocessing.monaco_utils import MonacoUtils
        from tools.monaco_preprocessing.build_vector_db import MonacoRAG

        super().__init__()

        # Set the appropriate paths based on the chosen database
        self.db_nickname = db_nickname
        path_to_vec_db = path_to_vec_db or BaseWebToolMonaco._PATH_TO_MONACO_WIKI_VEC_DB
        path_to_kb = path_to_kb or BaseWebToolMonaco._PATH_TO_MONACO_WIKI
        path_to_ds_to_kb_urls_mapping = \
            path_to_ds_to_kb_urls_mapping or BaseWebToolMonaco._PATH_TO_MONACO_TO_WIKI_URLS_MAPPING
        path_to_kb_to_ds_urls_mapping = \
            path_to_kb_to_ds_urls_mapping or BaseWebToolMonaco._PATH_TO_WIKI_TO_MONACO_URLS_MAPPING

        # Use a lock to ensure thread-safe lazy initialization of the global singletons
        with _GLOBAL_INIT_LOCK:
            # Initialize the local knowledge base (Wikipedia)
            if _GLOBAL_WIKIPEDIA_RAG is None:
                _GLOBAL_WIKIPEDIA_RAG = MonacoRAG(use_reranker=False)
                _GLOBAL_WIKIPEDIA_RAG.load(path_to_vec_db)
            self._wikipedia_rag = _GLOBAL_WIKIPEDIA_RAG

            # Initialize the dataset helper
            if _GLOBAL_MONACO_UTILS is None:
                _GLOBAL_MONACO_UTILS = MonacoUtils(
                    path_to_kb=path_to_kb,
                    path_to_ds_to_kb_urls_mapping=path_to_ds_to_kb_urls_mapping,
                    path_to_kb_to_ds_urls_mapping=path_to_kb_to_ds_urls_mapping,
                )
            self._monaco_utils = _GLOBAL_MONACO_UTILS

        # Initialize the variable that will hold the MoNaCo entry whose question was passed to the agent
        self._monaco_entry: Optional[Dict] = None
        # Initialize the list of the URLs to the Wikipedia pages that are required to answer the MoNaCo main question.
        # This list contains Wikipedia URLs (i.e., not the URLs provided by MoNaCo)
        self._monaco_entry_golden_w_urls: Set[str] = set()

        # True if it was not possible to remove the golden passages from one or more pages returned to the agent
        self.failed_to_remove_golden_passages: bool = False

    def set_current_monaco_entry(self, monaco_entry: Dict):
        """
        Stores the given `monaco_entry` as the entry with the question that was initially provided to the agent.
        This will be taken into account when the WRONG_OUTPUT or DISTRACTOR_OUTPUT switches are enabled.
        """

        self._monaco_entry = monaco_entry

        golden_urls_set = set()
        for gp in self._monaco_entry['golden_passages']:
            if isinstance(gp, BaseGoldenPassage):
                for url in gp.url:
                    try:
                        wiki_url = self._monaco_utils.convert_ds_to_kb_url(url)
                        golden_urls_set.add(wiki_url)
                    except KeyError:
                        # Ignore the URL and move to the next one
                        pass
            else:
                wiki_url = self._monaco_utils.convert_ds_to_kb_url(gp)
                golden_urls_set.add(wiki_url)
        self._monaco_entry_golden_w_urls = list(golden_urls_set)
