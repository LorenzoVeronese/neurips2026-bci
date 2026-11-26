"""Abstract sub-family base class for statistical and competition metric Evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from neurips_bci.evaluator.Evaluator import Evaluator


class MetricEvaluator(Evaluator, ABC):
    """Base contract for balanced accuracy, cell aggregation, and clustered hypothesis tests."""

    def __init__(self, evaluation_protocol: str) -> None:
        super().__init__(evaluation_protocol=evaluation_protocol)

    @abstractmethod
    def score_predictions(
        self,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        session_ids: np.ndarray,
        interface_contexts: np.ndarray,
    ) -> dict[str, object]:
        """Compute competition cell-mean and pooled metrics on window-level predictions."""
