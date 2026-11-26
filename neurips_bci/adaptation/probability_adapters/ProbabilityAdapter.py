"""Abstract base class for causal B=1 marginal prior probability balancers."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from neurips_bci.adaptation.Adapter import Adapter
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class ProbabilityAdapter(Adapter, ABC):
    """Base contract for causal simplex probability balancers (BalStd, BalKal, and PC-Sinkhorn)."""

    def __init__(self, adapter_role: str, class_count: int = 3) -> None:
        super().__init__(adapter_role=adapter_role)
        self._class_count = ParameterTypeValidator.require_integer(class_count, "class_count", minimum_bound=2)

    @property
    def class_count(self) -> int:
        return self._class_count

    def _validate_and_normalize_simplex(self, candidate_probabilities: np.ndarray) -> np.ndarray:
        ParameterTypeValidator.require_numpy_array(
            candidate_probabilities,
            "candidate_probabilities",
            expected_shape=(self._class_count,),
        )
        clipped = np.clip(np.asarray(candidate_probabilities, dtype=np.float64), 1e-12, 1.0)
        return clipped / float(np.sum(clipped))

    @abstractmethod
    def adapt_probabilities(self, raw_probabilities: np.ndarray) -> np.ndarray:
        """Apply causal marginal class-prior balancing to a single window's probability vector."""
