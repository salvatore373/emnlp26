from abc import ABC, abstractmethod
from typing import List, Iterator


class BenchmarkEntry(ABC):
    """
    This class represents a single entry in the benchmark data, consisting of a user question and the expected answer.
    """

    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer


class BenchmarkData(ABC):
    DATASET_ID: str = ""
    PARTITION_NAMES: List[str] = []
    SPLITS: List[str] = []

    @staticmethod
    @abstractmethod
    def retrieve_benchmark_data(*args) -> Iterator[BenchmarkEntry]:
        """
        For each entry of the benchmark, yield a tuple containing the user question, the answer and some additional
        data.

        Yields: A BenchmarkEntry object.
        """
        pass
