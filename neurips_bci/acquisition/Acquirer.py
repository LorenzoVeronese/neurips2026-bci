"""Abstract root class for the Acquirer family (datasets, checkpoints, and CV splits)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Acquirer(ABC):
    """Base contract for reproducible dataset/checkpoint downloaders, extractors, and fold builders."""

    def __init__(self, acquirer_source: str) -> None:
        if not isinstance(acquirer_source, str) or not acquirer_source.strip():
            raise TypeError("acquirer_source must be a non-empty str")
        self._acquirer_source = acquirer_source

    @property
    def acquirer_source(self) -> str:
        return self._acquirer_source

    @abstractmethod
    def acquire(self) -> Path | list:
        """Execute the acquisition or extraction step and return produced artifact path(s) or splits."""
