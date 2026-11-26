"""Abstract root class for metric, CV, and archive contract Evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Evaluator(ABC):
    """Base contract for statistical scoring, cross-validation, and survival verification."""

    def __init__(self, evaluation_protocol: str) -> None:
        if not isinstance(evaluation_protocol, str) or not evaluation_protocol.strip():
            raise TypeError("evaluation_protocol must be a non-empty str")
        self._evaluation_protocol = evaluation_protocol

    @property
    def evaluation_protocol(self) -> str:
        return self._evaluation_protocol

    @abstractmethod
    def evaluate(self) -> dict[str, object]:
        """Run the evaluation protocol and return a structured summary dictionary."""
