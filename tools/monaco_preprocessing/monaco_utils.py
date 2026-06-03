"""
Define helper functions and classes to retrieve Wikipedia pages with or without golden information.
"""
import difflib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, List, Dict, Union
from urllib.parse import urldefrag

from tools.monaco_preprocessing.rag_kb_utils import RAGKBUtils

"""
Create the classes that represent the different types of golden passages.
The goal of each class is to store the directions to find the golden passage inside the Wikipedia page.
"""


class BaseGoldenPassage:
    def validate_fields(self):
        # Validate the URL format
        urls = [u.strip() for u in re.split(r'(?<!^)(?:,\s*|\s*)?(?=https?://)', self.url) if u.strip()]
        formatted_urls = []
        for u in urls:
            u = urldefrag(u).url
            u = re.sub(r"\s+", "", u)
            u = re.sub(r'#:~:text=.*$', '', u)

            if u:
                if "')" in u:
                    u = u.replace("')", '')
                if not u.startswith("http"):
                    u = None
                    self.url = None
                    return
                formatted_urls.append(u)
        self.url = formatted_urls

    @staticmethod
    def from_dict(data: dict):
        # Determine the subclass based on unique keys
        if "infobox_section" in data:
            target_cls = InfoboxGoldenPassage
        elif "list_or_table_section" in data:
            target_cls = ListOrTableGoldenPassage
        elif "sentence_section" in data:
            target_cls = SentenceGoldenPassage
        else:
            raise ValueError("Dictionary does not match any GoldenPassage subclass.")

        # Create instance without calling __init__ (bypasses step/index requirement)
        obj = target_cls.__new__(target_cls)
        obj.__dict__.update(data)
        return obj


class InfoboxGoldenPassage(BaseGoldenPassage):  # create one for each "infobox" in "context_type"
    def __init__(self, step, timestamp, infobox_ind):
        self.golden_passage: List[str] = ["Not Available"] if len(step['answers']) == 0 else step['answers']
        self.infobox_section = step['answers_section'][infobox_ind]
        self.infobox_column = step['answers_column'][infobox_ind]
        self.url = step['source_url'][
            infobox_ind]  # is a single string, but may contain multiple URL separated by comma
        self.subquestion = step['subquestion']
        self.timestamp = timestamp

        self.validate_fields()

    def to_json(self):
        return {
            "golden_passage": self.golden_passage,
            "infobox_section": self.infobox_section,
            "infobox_column": self.infobox_column,
            "url": self.url,
            "subquestion": self.subquestion,
            "timestamp": self.timestamp,
        }


class ListOrTableGoldenPassage(BaseGoldenPassage):  # create one for each "list"/"table" in "context_type"
    def __init__(self, step, timestamp, list_or_table_ind):
        self.golden_passage: List[str] = ["Not Available"] if len(step['answers']) == 0 else step['answers']
        # Multiple sections might be in the same string in list_or_table_section:
        # look for the sections of the page that are in this field.
        self.list_or_table_section = step['answers_section'][list_or_table_ind]
        self.url = step['source_url'][
            list_or_table_ind]  # is a single string, but may contain multiple URL separated by comma
        self.subquestion = step['subquestion']
        self.timestamp = timestamp

        self.validate_fields()

    def to_json(self):
        return {
            "golden_passage": self.golden_passage,
            "list_or_table_section": self.list_or_table_section,
            "url": self.url,
            "subquestion": self.subquestion,
            "timestamp": self.timestamp,
        }


class SentenceGoldenPassage(BaseGoldenPassage):  # create one for each "sentence" in "context_type"
    def __init__(self, step, timestamp, sentence_ind):
        self.golden_passage: List[str] = None if step['answers_sentence'][sentence_ind] == '{}' else \
            step['answers_sentence'][sentence_ind]
        # Multiple sections might be in the same string in sentence_section:
        # look for the sections of the page that are in this field.
        self.sentence_section = step['answers_section'][sentence_ind]
        self.url = step['source_url'][
            sentence_ind]  # is a single string, but may contain multiple URL separated by comma
        self.subquestion = step['subquestion']
        self.timestamp = timestamp

        self.validate_fields()

    def to_json(self):
        return {
            "golden_passage": self.golden_passage,
            "sentence_section": self.sentence_section,
            "url": self.url,
            "subquestion": self.subquestion,
            "timestamp": self.timestamp,
        }


class MonacoUtils(RAGKBUtils):
    """
    A helper class to retrieve Wikipedia pages with or without golden information from the MoNaCo knowledge base.
    """

    @dataclass
    class MonacoWikiPageRetrievalResult(RAGKBUtils.KBPageRetrievalResult):
        """ The object returned when a Wikipedia page is retrieved. """
        # A boolean that is True if the entry was modified (i.e., whether at least one golden passage has been removed)
        # and False otherwise.
        modified_entry: bool = False
        # A boolean that is True if one or more golden passages couldn't be removed from the entry
        unable_to_rem_some_passages: bool = False
        # A boolean that is True if one or more golden passages couldn't be removed from the entry's introduction
        unable_to_rem_passages_in_intro: bool = False

    def __init__(
            self,
            path_to_kb: str,
            path_to_ds_to_kb_urls_mapping: str = None,
            path_to_kb_to_ds_urls_mapping: str = None,
    ):
        """
        Create an instance of MonacoUtils class.

        Args:
            path_to_kb: The path to the parquet/duckdb file containing the filtered version of the Wikipedia,
             containing only the pages that occur in MoNaCo.
            path_to_ds_to_kb_urls_mapping: The path to the parquet file containing the mapping between MoNaCo URLs
             and Wikipedia page URLs.
            path_to_kb_to_ds_urls_mapping: The path to the parquet file containing the mapping between Wikipedia
             page URLs and MoNaCo URLs.
        """
        if path_to_ds_to_kb_urls_mapping is None:
            path_to_ds_to_kb_urls_mapping = Path(path_to_kb).parent / "monaco_to_wiki_urls_mapping.parquet"
        if path_to_kb_to_ds_urls_mapping is None:
            path_to_kb_to_ds_urls_mapping = Path(path_to_kb).parent / "wiki_to_monaco_urls_mapping.parquet"

        super().__init__(path_to_kb, path_to_ds_to_kb_urls_mapping, path_to_kb_to_ds_urls_mapping)

        # --- DEBUG variables ------

        # The path to a file where to write information during the process of removing golden passages from Wikipedia
        # pages. If None, no debug information is written.
        self.debug_file_path: Optional[str] = None
        self._debug_file = open(self.debug_file_path, 'w') if self.debug_file_path else None

        self._modified_entries = 0
        self._modified_entries_per_type = {
            'sentence': {'failed': 0, 'modified': 0},
            'infobox': {'failed': 0, 'modified': 0},
            'list_table': {'failed': 0, 'modified': 0},
        }
        self._modified_urls = {}

    _WIKI_HEADER_RE = re.compile(r"^(={1,6})[^=\n].*?\1[ \t]*$", re.MULTILINE)

    @staticmethod
    def retrieve_monaco_entries(path_to_monaco_jsonl: str) -> List[Dict]:
        """
        Returns the list of entries of the MoNaCo benchmark.
        """
        monaco: List[Dict] = []
        with open(path_to_monaco_jsonl, 'r') as f:
            for line in f:
                d = json.loads(line)
                d['golden_passages'] = \
                    [BaseGoldenPassage.from_dict(d) if isinstance(d, dict) else d for d in d['golden_passages']]
                monaco.append(d)
        return monaco

    def _dbg_write(self, *parts) -> None:
        """Write debug parts to the configured debug file if present.

        This helper centralizes debug writing so callers don't need to guard
        with ``if self._debug_file`` everywhere.
        """
        if not getattr(self, "_debug_file", None):
            return
        for p in parts:
            # Keep behavior identical to previous direct writes
            self._debug_file.write(str(p))

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase + collapse whitespace."""
        return re.sub(r"\s+", " ", text).strip().lower()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Word tokens in lowercase (punctuation stripped)."""
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def _parse_headers(text: str) -> List[Tuple[int, int, int]]:
        """Return [(char_offset, level, line_end), ...] for every section header."""
        results = []
        for m in MonacoUtils._WIKI_HEADER_RE.finditer(text):
            level = len(m.group(1))
            line_end = m.end() + (1 if m.end() < len(text) and text[m.end()] == "\n" else 0)
            results.append((m.start(), level, line_end))
        return results

    @staticmethod
    def _find_best_window(
            page: str,
            query: str,
            similarity_threshold: float,
            top_k: int = 10,
    ) -> Optional[Tuple[int, int, float]]:
        """
        Return ``(char_start, char_end, ratio)`` for the region of *page* that
        best matches *query*, or ``None`` if the best ratio < *similarity_threshold*.

        Complexity
        ----------
        Pre-filter : O(P)   — rolling multiset-Jaccard, O(1) per token step
        Verification: O(top_k · Q) — full SequenceMatcher on the k best candidates
        Total: O(P + top_k · Q)  vs  O(P · Q) for the naïve sliding-window approach
        """
        page_tokens: List[str] = MonacoUtils._tokenize(page)
        query_tokens: List[str] = MonacoUtils._tokenize(query)

        p_tok = len(page_tokens)
        q_tok = len(query_tokens)

        if not p_tok or not q_tok:
            return None

        # Build a char-offset map: token index → (char_start, char_end)
        # Aligned with the same regex used by _tokenize()
        tok_spans: List[Tuple[int, int]] = [
            (m.start(), m.end()) for m in re.finditer(r"\b\w+\b", page)
        ]
        n_spans = len(tok_spans)
        if n_spans == 0:
            return None

        # Guard: query has no word tokens (pure whitespace / symbols)
        if q_tok == 0:
            return None

        query_counter = Counter(query_tokens)
        norm_query = MonacoUtils._normalize(query)

        # ------------------------------------------------------------------
        # Phase 1 – rolling Jaccard for each window size
        # ------------------------------------------------------------------
        # We try three window sizes (75 %, 100 %, 125 % of the query token count)
        # to accommodate passages that are shorter/longer than the page region.
        candidates: List[Tuple[float, int, int]] = []  # (jaccard, tok_start, tok_end)

        for size_factor in (0.75, 1.00, 1.25):
            win = max(1, round(q_tok * size_factor))
            if win > n_spans:
                continue

            # Initialise the first window
            win_counter: Counter = Counter(page_tokens[:win])
            # Multiset intersection size: sum(min(a,b) for each token)
            intersect: int = sum((win_counter & query_counter).values())
            q_total: int = sum(query_counter.values())  # constant = q_tok

            def _jaccard() -> float:
                union = q_total + win - intersect
                return intersect / union if union else 0.0

            candidates.append((_jaccard(), 0, win))

            # Slide one token at a time — O(1) update per step
            for i in range(1, n_spans - win + 1):
                removed = page_tokens[i - 1]
                added = page_tokens[i + win - 1]

                # Remove outgoing token
                if win_counter[removed] <= query_counter[removed]:
                    intersect -= 1
                win_counter[removed] -= 1
                if win_counter[removed] == 0:
                    del win_counter[removed]

                # Add incoming token (increment first, then compare)
                win_counter[added] += 1
                if win_counter[added] <= query_counter[added]:
                    intersect += 1

                candidates.append((_jaccard(), i, i + win))

        # ------------------------------------------------------------------
        # Phase 2 – SequenceMatcher on the top-k Jaccard candidates only
        # ------------------------------------------------------------------
        # partial sort: O(P log top_k) but typically much faster than full sort
        candidates.sort(key=lambda x: -x[0])

        best_ratio = 0.0
        best_char_start = tok_spans[0][0]
        best_char_end = tok_spans[min(q_tok, p_tok, n_spans) - 1][1]

        seen_starts: set = set()  # deduplicate candidates at the same offset
        seen_bounds: set = set()  # deduplicate candidates with the same snapped boundaries

        for _, tok_start, tok_end in candidates[:top_k]:
            if tok_start in seen_starts:
                continue
            seen_starts.add(tok_start)

            tok_start_clamped = min(tok_start, n_spans - 1)
            tok_end_clamped = min(tok_end, n_spans)
            if tok_end_clamped <= tok_start_clamped:
                continue

            char_start = tok_spans[tok_start_clamped][0]
            char_end = tok_spans[tok_end_clamped - 1][1]

            # Snap to sentence boundaries to avoid splitting sentences
            while char_start > 0 and page[char_start - 1] not in '.?!\n':
                char_start -= 1
            while char_start < char_end and page[char_start] in ' \t':
                char_start += 1
            while char_end < len(page) and page[char_end] not in '.?!\n':
                char_end += 1
            if char_end < len(page) and page[char_end] in '.?!':
                char_end += 1

            if (char_start, char_end) in seen_bounds:
                continue
            seen_bounds.add((char_start, char_end))

            ratio = difflib.SequenceMatcher(
                None, norm_query, MonacoUtils._normalize(page[char_start:char_end])
            ).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_char_start = char_start
                best_char_end = char_end

            # Early exit: perfect match
            if best_ratio == 1.0:
                break

        if best_ratio >= similarity_threshold:
            return best_char_start, best_char_end, best_ratio
        return None

    # ──────────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _delete_section(
            page_text: str,
            passage: str,
            what_to_remove: Literal["sentences", "section"] = "sentences",
            similarity_threshold: float = 0.6,
            top_k: int = 10,
    ) -> Optional[str]:
        """
        Find a (possibly fuzzy-matched) passage in *page_text* and remove either
        the matched text or its entire containing section.

        Parameters
        ----------
        page_text : str
            Full text of a parsed Wikipedia page.
        passage : str
            Passage to locate (exact or approximate).
        what_to_remove : {"sentences", "section"}
            ``"sentences"`` – delete only the matched region.
            ``"section"``   – delete the whole section (and sub-sections) that
            contains the match, up to the next sibling/ancestor header.
        similarity_threshold : float
            Minimum SequenceMatcher ratio (0–1) to accept a match.
        top_k : int
            Number of top Jaccard candidates forwarded to the exact similarity
            check.  Increase if you suspect very noisy passages.

        Returns
        -------
        str   – modified page text (passage or section removed).
        None  – no sufficiently similar region was found.
        """
        # Check whether `passage` is inside `page_text` and get the index
        idx = page_text.find(passage)
        if idx != -1:
            result = page_text[:idx] + page_text[idx + len(passage):]
            return re.sub(r"\n{3,}", "\n\n", result).strip()

        # Search for a match of a string similar to `passage` in `page_text`
        match = MonacoUtils._find_best_window(page_text, passage, similarity_threshold, top_k)
        if match is None:
            return None

        match_start, match_end, score = match
        # ── sentences mode ──────────────────────────────────────────────────────
        if what_to_remove == "sentences":
            # Expand to the nearest line boundary to avoid orphan fragments
            seg_start = page_text.rfind("\n", 0, match_start) + 1  # start of line
            seg_end = match_end
            while seg_end < len(page_text) and page_text[seg_end] == "\n":
                seg_end += 1

            result = page_text[:seg_start] + page_text[seg_end:]
            return re.sub(r"\n{3,}", "\n\n", result).strip()

        # ── section mode ────────────────────────────────────────────────────────
        if what_to_remove == "section":
            headers = MonacoUtils._parse_headers(page_text)

            # Last header whose offset ≤ match_start
            containing_idx = next(
                (i for i in range(len(headers) - 1, -1, -1)
                 if headers[i][0] <= match_start),
                -1,
            )

            if containing_idx == -1:
                # Match is in the lead prose; fall back to sentence removal
                result = page_text[:match_start] + page_text[match_end:]
                return re.sub(r"\n{3,}", "\n\n", result).strip()

            sec_offset, sec_level, _ = headers[containing_idx]

            # Section ends at the next header of equal or higher level
            sec_end = next(
                (h[0] for h in headers[containing_idx + 1:] if h[1] <= sec_level),
                len(page_text),
            )

            result = page_text[:sec_offset] + page_text[sec_end:]
            return re.sub(r"\n{3,}", "\n\n", result).strip()

        raise ValueError(f"Unknown what_to_remove value: {what_to_remove!r}")

    @staticmethod
    def _remove_infobox_key(
            infobox: dict,
            key: str,
            similarity_threshold: float = 0.8,
    ) -> Optional[dict]:
        """
        Remove the most similar key to *key* from dictionary *d*, but only if its
        similarity score meets or exceeds *similarity_threshold*.

        Parameters
        ----------
        infobox : dict
            The dictionary to operate on.  Modified **in-place** and also returned.
        key : str
            The key to look for (exact or approximate).
        similarity_threshold : float
            Minimum ``difflib.SequenceMatcher`` ratio (0–1) required to treat a
            candidate key as a match.  Default is 0.8.

        Returns
        -------
        dict
            The modified dictionary (same object as *d*) if a key was removed.
        None
            If no key with sufficient similarity was found.

        Notes
        -----
        Complexity
            ``difflib.get_close_matches`` uses the same SequenceMatcher internally
            but applies a fast *quick_ratio* pre-filter that skips hopeless
            candidates in O(1), making the overall scan effectively O(n) for
            typical inputs rather than O(n·m) where m is key length.
        """
        # Try to remove an infobox key exactly matching the given key
        if key in infobox:
            del infobox[key]
            return infobox

        # Collect all the keys of the infobox that are strings
        str_keys = [k for k in infobox if isinstance(k, str)]
        if not str_keys:
            return None

        # With get_close_matches(), get n=1 matches between the given key and the infobox keys sorted by score
        matches = difflib.get_close_matches(
            key, str_keys, n=1, cutoff=similarity_threshold
        )

        if not matches:
            return None

        best_key = matches[0]
        del infobox[best_key]
        return infobox

    @staticmethod
    def _infobox_to_wiki_page_format(infobox: Dict, page_title: str) -> str:
        """
        Formats the given dictionary containing the infobox data as a string in the same format of the sections of the
        parsed Wikipedia pages.
        If the infobox is empty or contains just placeholders, it returns an empty string.
        """
        if not infobox or infobox.get("dummy_label"):
            return ""

        # Prepare and clean the data rows first
        rows = []
        for key, value in infobox.items():
            if isinstance(value, list):
                value = " • ".join(value)

            # Clean formatting and add bolding to keys
            clean_key, clean_value = \
                [t.replace('\u00b7', '•').strip().replace("|", "\\|") for t in [key, value]]
            # Escape pipes to prevent table breakage
            rows.append((f"**{clean_key}**", clean_value))

        # Calculate the maximum width for each column
        col1_label, col2_label = "Property", "Details"

        # Max width is the longer of the header or the longest cell content
        width1 = max([len(row[0]) for row in rows] + [len(col1_label)])
        width2 = max([len(row[1]) for row in rows] + [len(col2_label)])

        # Build the table with padding using .ljust(width)
        header = f"| {col1_label.ljust(width1)} | {col2_label.ljust(width2)} |"
        separator = f"| {'-' * width1} | {'-' * width2} |"

        table_lines = [f"=== {page_title} ===", header, separator]

        for k, v in rows:
            table_lines.append(f"| {k.ljust(width1)} | {v.ljust(width2)} |")

        return "\n".join(table_lines)

    @staticmethod
    def _integrate_infobox_to_page(wiki_page, infobox_md):
        """
        Integrates an infobox after the intro text but before
         the first sub-section (H2).
        """
        lines = wiki_page.splitlines()
        title_index = -1

        # Find the first H1 title (# Title)
        for i, line in enumerate(lines):
            if line.strip().startswith("= "):
                title_index = i
                break

        # Handle case where no title is found
        if title_index == -1:
            return f"{infobox_md}\n\n{wiki_page}"

        # Search for the next section (H2) starting after the title
        next_section_index = len(lines)  # Default to end of page if no H2 exists
        for i in range(title_index + 1, len(lines)):
            # We look for '##' because H2 is the standard next section
            if lines[i].strip().startswith("== "):
                next_section_index = i
                break

        # Slice the page
        # Everything from the start of the page through the intro text
        intro_and_title = lines[:next_section_index]
        # Everything from the H2 heading to the end
        remaining_sections = lines[next_section_index:]

        # 4. Reconstruct with the infobox in the middle
        # .strip() handles extra trailing newlines in the intro
        new_page = (
                "\n".join(intro_and_title).strip() +
                "\n\n" +
                infobox_md +
                "\n\n" +
                "\n".join(remaining_sections).strip()
        )

        return new_page.strip()

    def _remove_golden_passages(
            self,
            monaco_entry: Dict, wiki_entry: Dict,
            sentence_sim_threshold: float = 0.8,
            lists_or_tables_sim_threshold: float = 0.8,
            infobox_sim_threshold: float = 0.6,
            introduction_check_threshold: float = 0.9,
    ) -> MonacoWikiPageRetrievalResult:
        """
        Remove from the Wikipedia page represented by `wiki_entry` all the passages in the `golden_passages` field of
        `monaco_entry`.

        Arguments:
            monaco_entry (dict): The entry whose golden_passages` will be removed from the Wikipedia page.
            wiki_entry (dict): The entry of the dataset of Wikipedia pages from which to remove the golden passages.
            sentence_sim_threshold (float): The similarity threshold for the sentence golden passages. A sentence is
             removed from the Wikipedia page if its similarity with the golden passage is higher than this threshold.
            lists_or_tables_sim_threshold (float): The similarity threshold for the list or table golden passages.
             A list or table is removed from the Wikipedia page if its similarity with the golden passage is higher
             than this threshold.
            infobox_sim_threshold (float): The similarity threshold for the infobox golden passages. An infobox key is
             removed from the Wikipedia page if its similarity with the golden passage is higher than this threshold.
            introduction_check_threshold (float): The similarity threshold to check whether a golden passage's title
             is the Wikipedia page's title.

        Returns:
            A WikiPageRetrievalResult object.
        """
        # Get the URLs associated with this Wikipedia entry
        wiki_entry_url = wiki_entry["page_url"]

        # Keep the original text and infobox (to check whether the golden passage was not in the golder or was deleted)
        original_text = wiki_entry["page_text"]
        original_infobox = json.loads(wiki_entry["infobox"])

        # Make a variable to remember whether the entry will be modified or not (at least one passage was removed)
        modified_entry = False
        # Make a variable to remember whether we failed to delete one or more golden passage from the wikipedia page
        unable_to_rem_some_passages = False
        # A boolean that is True if one or more golden passages couldn't be removed from the entry's introduction
        unable_to_rem_passages_in_intro: bool = False

        # Remember whether the page and title was printed to the debug file
        wrote_title_to_debug = False

        # For each golden passage in the MoNaCo entry, if it is referred to the given Wikipedia entry, remove the
        # corresponding information form the Wikipedia page.
        for gp in monaco_entry['golden_passages']:
            # Skip this passage if not related to the given Wikipedia page
            this_passage_urls = set()
            for m_url in gp.url:
                this_passage_urls.add(m_url)
                try:
                    this_passage_urls.add(self.convert_ds_to_kb_url(m_url))
                except KeyError:
                    # The Monaco URL of this passage was not scraped
                    pass
            if wiki_entry_url not in this_passage_urls:
                continue

            # Write to the debug file only the random X% of the total instances
            should_write_debug = random.randint(1, 100) <= 2
            if should_write_debug and not wrote_title_to_debug:
                self._dbg_write('## ORIGINAL TEXT\n', original_text, '\n', '## ORIGINAL INFOBOX\n',
                                json.dumps(original_infobox, indent=4), '\n')
                wrote_title_to_debug = True

            if isinstance(gp, SentenceGoldenPassage) or isinstance(gp, ListOrTableGoldenPassage):
                # The golden passage is inside a sentence or a list of the text: then, remove it from the text
                if gp.golden_passage is None:
                    # The golden passage is not available for this step
                    continue

                # Convert the golden passage to a string
                txt_passage = '\n'.join(gp.golden_passage) if isinstance(gp.golden_passage, list) else gp.golden_passage

                # Determine the similarity threshold
                similarity_threshold = sentence_sim_threshold if isinstance(gp, SentenceGoldenPassage) \
                    else lists_or_tables_sim_threshold

                # Remove the section containing the golden passage
                text_wo_section = self._delete_section(
                    page_text=wiki_entry["page_text"],
                    passage=txt_passage,
                    what_to_remove='section',
                    similarity_threshold=similarity_threshold,
                )

                if text_wo_section is None:
                    # Check whether the golden passage was not in the original text or was already deleted by a
                    # previous golden passage.
                    original_text_wo_section = self._delete_section(
                        page_text=original_text,
                        passage=txt_passage,
                        what_to_remove='section',
                        similarity_threshold=similarity_threshold
                    )
                    if original_text_wo_section is None:
                        # The golden passage was not in the original text: this means that the golden passage is not
                        # actually present in the Wikipedia page.

                        # Remember whether this golden passage was in the introduction or not
                        unable_to_rem_passages_in_intro = \
                            (isinstance(gp, SentenceGoldenPassage) and
                             difflib.SequenceMatcher(
                                 None,
                                 repr(gp.sentence_section),
                                 repr(wiki_entry["page_title"].replace("_", " "))
                             ).ratio() >= introduction_check_threshold)
                        unable_to_rem_some_passages = True

                        # Update debug variables
                        if should_write_debug:
                            self._dbg_write("### Failed to remove\n", json.dumps(gp.to_json(), indent=4), '\n')
                        self._modified_urls.setdefault(wiki_entry_url, {"successfully_modif": 0, "unable_to_modif": 0})[
                            "unable_to_modif"] += 1
                        self._modified_entries_per_type[
                            'sentence' if isinstance(gp, SentenceGoldenPassage) else 'list_table']['failed'] += 1

                        continue
                else:
                    # The golden passage has been removed. Then, update the page_text
                    wiki_entry['page_text'] = text_wo_section
                    modified_entry = True

                    # Update debug variables
                    self._modified_urls.setdefault(wiki_entry_url, {"successfully_modif": 0, "unable_to_modif": 0})[
                        "successfully_modif"] += 1
                    self._modified_entries_per_type[
                        'sentence' if isinstance(gp, SentenceGoldenPassage) else 'list_table']['modified'] += 1
                    if should_write_debug:
                        self._dbg_write("### Successfully removed sentece/list\n", json.dumps(gp.to_json(), indent=4),
                                        '\n')

            elif isinstance(gp, InfoboxGoldenPassage):
                # The golden passage is inside an infobox: then, remove that portion of the infobox
                infoboxes = json.loads(wiki_entry["infobox"])

                infoboxes = self._remove_infobox_key(infoboxes, gp.infobox_column, infobox_sim_threshold)
                if not infoboxes:
                    # Check whether the golden passage was not in the original infobox or was already deleted by a
                    # previous golden passage.
                    infoboxes = self._remove_infobox_key(original_infobox, gp.infobox_column, 0.6)
                    if not infoboxes:
                        # The golden passage was not in the original infobox: this means that the golden passage is not
                        # actually present in the Wikipedia page.
                        unable_to_rem_some_passages = True

                        # Update debug variables
                        if should_write_debug:
                            self._dbg_write("### Failed to remove\n", json.dumps(gp.to_json(), indent=4), '\n')
                        self._modified_urls.setdefault(wiki_entry_url, {"successfully_modif": 0, "unable_to_modif": 0})[
                            "unable_to_modif"] += 1
                        self._modified_entries_per_type['infobox']['failed'] += 1

                        continue
                else:
                    # The golden passage has been successfully removed. Then, update the infobox of the entry.
                    wiki_entry["infobox"] = json.dumps(infoboxes)
                    modified_entry = True

                    # Update debug variables
                    self._modified_urls.setdefault(wiki_entry_url, {"successfully_modif": 0, "unable_to_modif": 0})[
                        "successfully_modif"] += 1
                    self._modified_entries_per_type['infobox']['modified'] += 1
                    if should_write_debug:
                        self._dbg_write("### Successfully removed infobox\n", json.dumps(gp.to_json(), indent=4), '\n')
            else:
                raise ValueError(f"Golden passage type {type(gp)} not supported.")

        # Update debug variables
        if wrote_title_to_debug:
            self._dbg_write('## SUCCESSFULLY MODIFIED\n' if modified_entry else '## DID NOT MODIFY TEXT\n')
        self._modified_entries += int(modified_entry)

        # Return the resulting entry after removing all the golden passages
        return self.MonacoWikiPageRetrievalResult(
            kb_entry=wiki_entry,
            modified_entry=modified_entry,
            unable_to_rem_some_passages=unable_to_rem_some_passages,
            unable_to_rem_passages_in_intro=unable_to_rem_passages_in_intro,
        )

    def retrieve_wiki_page(self, wiki_url: Union[str, List[str]], monaco_entry_to_remove: Union[Dict, List[Dict]] = None,
                           integrate_lists_tables: bool = True, integrate_infobox: bool = True) -> \
            Union[MonacoWikiPageRetrievalResult, List[MonacoWikiPageRetrievalResult]]:
        """
        Returns the Wikipedia page (taken from Wikipedia) associated with the given URL (or a list of pages if a list of URLs is given).
        If `monaco_entry_to_remove` is given, the golden passages highlighted by this entry are removed from the
        returned page.

        Arguments:
            wiki_url: The URL or list of URLs of the Wikipedia pages to retrieve.
            monaco_entry_to_remove: The MoNaCo entry or list of entries containing the golden passages to remove from the returned page.
            integrate_lists_tables: Whether to add the extracted lists and tables to the text of the returned
             Wikipedia page.
            integrate_infobox: Whether to add a textual version of the infobox to the text of the returned
             Wikipedia page.

        Returns:
            A WikiPageRetrievalResult object, or a list of such objects if a list of URLs is given.

        Raises:
            ValueError: If any `url` is not associated with an entry in the Wikipedia pages dataset.
        """
        is_single_url = isinstance(wiki_url, str)
        wiki_urls = [wiki_url] if is_single_url else wiki_url

        if monaco_entry_to_remove is None:
            monaco_entries_to_remove = [None] * len(wiki_urls)
        elif isinstance(monaco_entry_to_remove, list):
            monaco_entries_to_remove = monaco_entry_to_remove
            if len(monaco_entries_to_remove) != len(wiki_urls):
                raise ValueError("The length of 'wiki_url' and 'monaco_entry_to_remove' lists must be the same.")
        else:
            monaco_entries_to_remove = [monaco_entry_to_remove] * len(wiki_urls)

        if not wiki_urls:
            return [] if not is_single_url else None

        wiki_entries = self.retrieve_kb_entry(wiki_urls)
        wiki_entries_map = {entry['page_url']: entry for entry in wiki_entries}

        results = []
        for url, m_entry in zip(wiki_urls, monaco_entries_to_remove):
            original_wiki_entry = wiki_entries_map.get(url)
            if original_wiki_entry is None:
                raise ValueError(f"Wikipedia entry with URL '{url}' not found.")

            # Make a shallow copy of the dictionary so we don't mutate the cached dictionary
            # if the same URL appears multiple times.
            wiki_entry = dict(original_wiki_entry)

            if integrate_lists_tables:
                # Add to the page text the tables/lists that were extracted from the HTML and are not yet in the page text
                wiki_entry['page_text'], added_tables = self._integrate_extr_lists_tables_in_page(
                    page_text=wiki_entry['page_text'],
                    extracted_lists_tables=json.loads(wiki_entry['extracted_lists_tables']),
                    similarity_threshold=0.8,
                )

            # Get the title of the Wikipedia page
            title = wiki_entry['page_title'].replace('_', ' ')

            wiki_entry_retrieval_res = None
            if m_entry is not None:
                # Remove the golden passages from the Wikipedia page
                wiki_entry_retrieval_res = \
                    self._remove_golden_passages(m_entry, wiki_entry,
                                                 sentence_sim_threshold=0.8,
                                                 lists_or_tables_sim_threshold=0.6,
                                                 infobox_sim_threshold=0.6, )
                wiki_entry = wiki_entry_retrieval_res.kb_entry

            if integrate_infobox:
                # Add the infobox to the page text
                wiki_entry['page_text'] = self._integrate_infobox_to_page(
                    wiki_entry['page_text'], self._infobox_to_wiki_page_format(json.loads(wiki_entry["infobox"]), title)
                )

            res = wiki_entry_retrieval_res or self.MonacoWikiPageRetrievalResult()
            res.kb_entry = wiki_entry
            results.append(res)

        return results[0] if is_single_url else results

    @staticmethod
    def _integrate_extr_lists_tables_in_page(
            page_text: str,
            extracted_lists_tables: List[Tuple[str, str]],
            similarity_threshold: float = 0.8
    ) -> Tuple[str, bool]:
        """
        Integrates lists or tables extracted from HTML into the Wikipedia page text.

        For each list/table, checks if it's already present in the text using a similarity match.
        If it is not, it finds the section referenced by the extraction (using similarity-based
        matching on the section titles) and inserts the list/table at the bottom of that section.

        Args:
            page_text: Full text of the Wikipedia page.
            extracted_lists_tables: A list of tuples, where the first element is the section of the
                                    page where they were found, and the second element is the list or table.
            similarity_threshold: Minimum similarity ratio required to match a list/table to the page
                                  text, and to match a section title to the headers in the page.

        Returns:
            A tuple containing the modified page text and a boolean indicating whether it was modified.
        """
        modified = False

        for section_name, content in extracted_lists_tables:
            # Handle potential lists in content
            if isinstance(content, list):
                content_str = '\n'.join(content)
            else:
                content_str = str(content)

            # Skip entries that have no word tokens — _find_best_window
            # returns None for these anyway, but the q_tok == 0 path
            # used to mis-initialise best_char_end before the early-return guard.
            if not re.search(r"\w", content_str):
                continue

            # 1. Check if the list/table is already in the page with the _find_best_window method
            match = MonacoUtils._find_best_window(page_text, content_str, similarity_threshold)

            if match is not None:
                # The list/table is already present in the page, do nothing
                continue

            # 2. Add it at the bottom of the section where it was found
            headers = MonacoUtils._parse_headers(page_text)

            best_header_idx = -1
            best_ratio = 0.0

            # Normalize the target section name to improve matching accuracy
            norm_target_section = MonacoUtils._normalize(str(section_name))

            for i, (offset, _, line_end) in enumerate(headers):
                # Extract the header text and remove the '=' characters
                header_line = page_text[offset:line_end].strip()
                clean_header = header_line.strip("= \t")
                norm_header = MonacoUtils._normalize(clean_header)

                # Check similarity against the target section
                ratio = difflib.SequenceMatcher(None, norm_target_section, norm_header).ratio()

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_header_idx = i

            if best_header_idx != -1 and best_ratio >= similarity_threshold:
                # If a matching section was found, insert the list/table at the bottom of the section
                # (i.e. right before the next header, or at the end of the text if it's the last section)
                if best_header_idx + 1 < len(headers):
                    insertion_point = headers[best_header_idx + 1][0]
                else:
                    insertion_point = len(page_text)

                # Add newlines for separation
                insertion = f"\n\n{content_str}\n\n"

                page_text = page_text[:insertion_point] + insertion + page_text[insertion_point:]
                modified = True

        return page_text, modified
