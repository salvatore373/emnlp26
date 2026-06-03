import os
import json
from typing import Iterator

from benchmark_data.benchmark_data import BenchmarkData, BenchmarkEntry


class ToolQABenchmarkEntry(BenchmarkEntry):
    """
    An entry in the ToolQA benchmark.
    """

    def __init__(self, entry: dict):
        """
        Create an instance of the ToolQABenchmarkEntry class.

        Args:
            entry: A dict representing a ToolQA benchmark entry.
        """
        super().__init__(entry.get("question"), entry.get("answer"))

        self.task_id = entry.get("qid")
        self.id = self.task_id
        
        # Example qid: "easy-dblp-0030"
        parts = self.task_id.split("-", 2)
        if len(parts) >= 2:
            self.difficulty = parts[0]
            self.dataset = parts[1]
        else:
            self.difficulty = "unknown"
            self.dataset = "unknown"

    def to_json(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "dataset": self.dataset
        }


class ToolQABenchmark(BenchmarkData):
    DATASET_ID = "toolqa"
    nickname = "toolqa"

    @staticmethod
    def retrieve_benchmark_data(*args) -> Iterator[ToolQABenchmarkEntry]:
        jsonl_path = "/workspace/tools/toolqa_preprocessing/data/toolqa_full.jsonl"

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"ToolQA dataset not found at {jsonl_path}")
            
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    yield ToolQABenchmarkEntry(data)


if __name__ == "__main__":
    benchmark = ToolQABenchmark()
    for entry in benchmark.retrieve_benchmark_data():
        print(entry)
        print(entry.to_json())
        break