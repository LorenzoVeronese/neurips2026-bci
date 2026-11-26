"""Abstract parent class for multi-organ causal ensemble BCI decoders."""

from __future__ import annotations

from abc import ABC
import numpy as np

from neurips_bci.decoder.Decoder import Decoder
from neurips_bci.operator.matrix_operators.CovarianceMatrixOperator import CovarianceMatrixOperator


class EnsembleDecoder(Decoder, ABC):
    """Shared centered log-probability fusion utilities for causal multi-organ ensemble decoders."""

    def __init__(self, decoder_variant: str) -> None:
        super().__init__(decoder_variant=decoder_variant)

    @staticmethod
    def centered_log_probabilities_1d(probabilities: np.ndarray) -> np.ndarray:
        return CovarianceMatrixOperator.centered_log_probabilities(np.asarray(probabilities, dtype=np.float64))

    @staticmethod
    def softmax_1d(logits: np.ndarray) -> np.ndarray:
        return CovarianceMatrixOperator.stable_softmax(np.asarray(logits, dtype=np.float64)[None])[0]

    @classmethod
    def blend_two_probabilities_log_linear(
        cls,
        first_prob: np.ndarray,
        second_prob: np.ndarray,
        second_weight: float,
    ) -> np.ndarray:
        w2 = float(second_weight)
        if w2 <= 0.0:
            return np.asarray(first_prob, dtype=np.float64)
        if w2 >= 1.0:
            return np.asarray(second_prob, dtype=np.float64)
        fused_lp = (1.0 - w2) * cls.centered_log_probabilities_1d(first_prob) + w2 * cls.centered_log_probabilities_1d(
            second_prob
        )
        return cls.softmax_1d(fused_lp)
