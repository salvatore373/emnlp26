import json
from typing import Iterator, Union, List, Any, Dict

from benchmark_data.benchmark_data import BenchmarkData, BenchmarkEntry
from tools.monaco_preprocessing.monaco_utils import BaseGoldenPassage


class MonacoBenchmarkEntry(BenchmarkEntry):
    """
    An entry in the MoNaCo benchmark.
    """

    def __init__(self, entry: dict):
        """
        Create an instance of the MonacoBenchmarkEntry class.

        Args:
            entry: A row of the table of the MoNaCo benchmark.
        """
        self.restructured_answer: Union[List[List[str], str], Dict[str, Any]] = \
            entry.get("restructured_answer", entry.get("valid_answers", []))
        self.response_formatting_instructions = self.restructured_answer.get("instruction", None) if isinstance(self.restructured_answer, dict) else None

        super().__init__(
            entry.get("question") +
            ("\n" + self.response_formatting_instructions if self.response_formatting_instructions else ""),
            json.dumps(self.restructured_answer),
        )

        self.id = entry["id"]
        self.original_entry = entry.copy()

    def to_json(self):
        return {
            "id": self.original_entry["id"],
            "question": self.question,
            "valid_answers": self.original_entry.get("valid_answers", []),
            "golden_passages": [gp.to_json() for gp in self.original_entry["golden_passages"]],
            "restructured_answer": self.original_entry.get("restructured_answer", {}),
        }


class MonacoBenchmark(BenchmarkData):
    PATH_TO_MONACO_JSONL = "/workspace/tools/monaco_preprocessing/data/restructured_monaco.jsonl"
    nickname = "monaco"

    def __init__(self, path_to_monaco_jsonl: str = None):
        """
        Arguments:
            path_to_monaco_jsonl (str): The path to the JSONL file containing the restructured version of the
             MoNaCo benchmark.
        """
        self.path_to_monaco_jsonl = path_to_monaco_jsonl or MonacoBenchmark.PATH_TO_MONACO_JSONL

    def retrieve_benchmark_data(self) -> Iterator[MonacoBenchmarkEntry]:
        """
        Yields the entries of the MoNaCo benchmark.
        """
        with open(self.path_to_monaco_jsonl, 'r') as f:
            for line in f:
                d = json.loads(line)
                d['golden_passages'] = \
                    [BaseGoldenPassage.from_dict(d) if isinstance(d, dict) else d for d in d['golden_passages']]

                yield MonacoBenchmarkEntry(d)


if __name__ == "__main__":
    bnc = MonacoBenchmark().retrieve_benchmark_data()
    for e in bnc:
        print(e)
        print(e.to_json())
        break
