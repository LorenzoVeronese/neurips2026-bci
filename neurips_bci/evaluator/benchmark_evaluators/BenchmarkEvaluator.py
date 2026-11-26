"""Abstract sub-family base class for cross-validation and archive survival BenchmarkEvaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from neurips_bci.evaluator.Evaluator import Evaluator


class BenchmarkEvaluator(Evaluator, ABC):
    """Base contract for streaming benchmark evaluators that persist structured audit reports."""

    def __init__(self, evaluation_protocol: str, output_directory: Path | None = None) -> None:
        super().__init__(evaluation_protocol=evaluation_protocol)
        self._output_directory = output_directory

    @property
    def output_directory(self) -> Path | None:
        return self._output_directory

    @abstractmethod
    def evaluate(self) -> dict[str, object]:
        """Run the benchmark evaluation and return a JSON-serializable summary report."""
