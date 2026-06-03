"""
Define helper functions and classes to retrieve Wikipedia pages with or without golden information.
"""
import difflib
import json
import os.path
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, List, Dict, Union
from urllib.parse import urldefrag

import duckdb
from _duckdb import DuckDBPyConnection


class RAGKBUtils:
    """
    A helper class to retrieve pages from a Knowledge Base with or without golden information.
    """

    _DUCKDB_RANDOM_SEED = 123

    @dataclass
    class KBPageRetrievalResult:
        """ The object returned when a Knowledge Base page is retrieved. """
        # The entry of the dataset representing a Knowledge Base page.
        kb_entry: Dict = None

        @property
        def wiki_page(self):
            """ The text of the Knowledge Base page."""
            return self.kb_entry['page_text']

    def __init__(
            self,
            path_to_db_ds: str,
            path_to_ds_to_kb_urls_mapping: str = None,
            path_to_kb_to_ds_urls_mapping: str = None,
    ):
        """
        Create an instance of MonacoUtils class.

        Args:
            path_to_db_ds: The path to the parquet file containing the Knowledge Base,
             containing only the pages that occur in dataset.
            path_to_ds_to_kb_urls_mapping: The path to the parquet file containing the mapping between dataset URLs
             and a Knowledge Base page URLs.
            path_to_kb_to_ds_urls_mapping: The path to the parquet file containing the mapping between a Knowledge Base
             page URLs and dataset URLs.
        """
        # Load the dataset containing the Wikipedia pages that occur in MoNaCo
        self.kb = duckdb.connect(path_to_db_ds, read_only=True)
        # self.kb = duckdb.connect()
        # self.kb.execute(f"CREATE VIEW dataset AS SELECT * FROM read_parquet('{path_to_db_ds}')")
        # Save in a list the names of the columns of the Wikipedia dataset
        self._kb_columns: Optional[List[str]] = None

        if path_to_ds_to_kb_urls_mapping is None:
            path_to_ds_to_kb_urls_mapping = ''
        if path_to_kb_to_ds_urls_mapping is None:
            path_to_kb_to_ds_urls_mapping = ''

        # Load the mapping between MoNaCo and Wikipedia URLs
        self._ds_to_kb, self._kb_to_ds = None, None
        if os.path.exists(path_to_ds_to_kb_urls_mapping) and os.path.exists(path_to_kb_to_ds_urls_mapping):
            with duckdb.connect() as con:
                monaco_data = con.execute(
                    f"SELECT url, page_url FROM read_parquet('{path_to_ds_to_kb_urls_mapping}')").fetchall()
                self._ds_to_kb = {row[0]: row[1] for row in monaco_data}

                wiki_data = con.execute(
                    f"SELECT page_url, urls FROM read_parquet('{path_to_kb_to_ds_urls_mapping}')").fetchall()
                self._kb_to_ds = {row[0]: json.loads(row[1]) for row in wiki_data}

    _WIKI_HEADER_RE = re.compile(r"^(={1,6})[^=\n].*?\1[ \t]*$", re.MULTILINE)

    def convert_ds_to_kb_url(self, ds_url: str) -> str:
        """ Returns the Knowledge Base URL mapped to the given Dataset URL. """
        if self._ds_to_kb is None:
            return ds_url
        else:
            return self._ds_to_kb[ds_url]

    def convert_kb_to_monaco_ds(self, kb_url: str) -> List[str]:
        """ Returns all the Dataset URLs mapped to the given Knowledge Base URL.  """
        if self._kb_to_ds is None:
            return [kb_url]
        else:
            return self._kb_to_ds[kb_url]

    # ──────────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────────

    def retrieve_kb_entry(self, kb_url: Union[str, List[str]], columns: List[str] = None) -> Union[Dict, List[Dict], None]:
        """
        Retrieve the entry (or entries) of the Knowledge Base where the `page_url` column is equal to `kb_url` (or in `kb_url`).

        Arguments:
            kb_url: The URL or list of URLs of the Knowledge Base entry to retrieve.
            columns: The columns to retrieve from the Knowledge Base. Default is `None`: in this case all the columns
             will be retrieved.

        Returns:
            The entry of the Knowledge Base, or a list of entries if a list of URLs is given.
            If a single URL is provided and not found, returns `None`.
        """
        columns_sql = "*" if columns is None else ", ".join(columns)
        cursor = self.kb.cursor()

        is_single_url = isinstance(kb_url, str)
        kb_urls = [kb_url] if is_single_url else kb_url

        if not kb_urls:
            return [] if not is_single_url else None

        entries = cursor.execute(f"SELECT {columns_sql} FROM dataset WHERE page_url IN (SELECT unnest(?))", [kb_urls]).fetchall()

        if not entries:
            return None if is_single_url else []

        if columns is None:
            if self._kb_columns is None:
                # Retrieve the name of the dataset columns
                self._kb_columns = [d[0] for d in cursor.description]
            columns_to_use = self._kb_columns
        else:
            columns_to_use = columns

        result = [dict(zip(columns_to_use, entry)) for entry in entries]

        return result[0] if is_single_url else result

    def retrieve_random_kb_entries(self, num_entries: int, columns: List[str] = None) -> \
            List[Dict[str, Union[str, int]]]:
        """Retrieves `num_entries` random entries of the Knowledge Base as a Polars DataFrame."""

        return (self.kb.cursor().execute(
            f"""
            SELECT {', '.join(columns) if columns is not None else '*'}
            FROM dataset
            USING SAMPLE reservoir({num_entries} ROWS)
            REPEATABLE({random.getrandbits(16)})
            """
        ).pl().to_dicts())

    def get_num_kb_entries(self) -> int:
        """Returns the number of entries in the Knowledge Base."""
        return self.kb.cursor().execute("SELECT count(*) FROM dataset").fetchone()[0]

    def get_kb_entries_cursor(self, columns: List[str] = None, limit: int = -1) -> DuckDBPyConnection:
        """
        Returns a cursor to the Wikipedia dataset that can be used to iterate through its entries.
        If `columns` is provided, the entries will only have the specified columns.
        If `limit` is provided, the cursor will iterate over `limit` entries.
        """
        columns = ', '.join(columns) if columns is not None else '*'
        limit_sql = f"LIMIT {limit}" if limit > 0 else ""
        return self.kb.cursor().execute(f"SELECT {columns} FROM dataset {limit_sql}")
