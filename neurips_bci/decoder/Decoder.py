"""Abstract root class for top-level causal streaming BCI Decoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class Decoder(ABC):
    """Base contract for full causal BCI decoders compatible with the Benchopt streaming API."""

    def __init__(self, decoder_variant: str) -> None:
        if not isinstance(decoder_variant, str) or not decoder_variant.strip():
            raise TypeError("decoder_variant must be a non-empty str")
        self._decoder_variant = decoder_variant

    @property
    def decoder_variant(self) -> str:
        return self._decoder_variant

    @abstractmethod
    def reset_state(self) -> None:
        """Reset all session-local causal state at session boundaries."""

    @abstractmethod
    def predict_proba(self, single_window_batch: np.ndarray) -> np.ndarray:
        """Return (B, 3) class probabilities for causal streaming input (B=1, C, T)."""

    def predict(self, single_window_batch: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(single_window_batch)
        return np.argmax(probabilities, axis=-1).astype(np.int64)
