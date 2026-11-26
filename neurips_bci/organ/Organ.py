"""Abstract root class for trainable and causally streaming H10 decoding Organs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class Organ(ABC):
    """Base contract for every modular decoding organ (CNN40, Riemannian, Foundation, Multimodal)."""

    def __init__(self, organ_identifier: str) -> None:
        if not isinstance(organ_identifier, str) or not organ_identifier.strip():
            raise TypeError("organ_identifier must be a non-empty str")
        self._organ_identifier = organ_identifier

    @property
    def organ_identifier(self) -> str:
        return self._organ_identifier

    @abstractmethod
    def reset_organ_state(self) -> None:
        """Reset all per-session online adaptation buffers."""

    @abstractmethod
    def save_organ_artifacts(self, target_directory: Path) -> list[Path]:
        """Persist pure-NumPy (.npz, allow_pickle=False) or safe state_dict (.pt) weights."""

    @abstractmethod
    def step_causal_window(self, cleaned_window: np.ndarray, is_artifact_flagged: bool) -> np.ndarray:
        """Process a single (C, T) window causally and return (3,) class probabilities."""
