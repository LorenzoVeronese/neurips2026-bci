"""Abstract root class for typed EEG Containers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Container(ABC):
    """Base contract for validated EEG window, session stream, and fold split containers."""

    def __init__(self, container_kind: str) -> None:
        if not isinstance(container_kind, str) or not container_kind.strip():
            raise TypeError("container_kind must be a non-empty str")
        self._container_kind = container_kind

    @property
    def container_kind(self) -> str:
        return self._container_kind

    @abstractmethod
    def window_count(self) -> int:
        """Return the number of EEG windows held in this container."""
