"""Abstract root class for the Configuration family."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Configuration(ABC):
    """Base contract for typed, validated experiment configuration objects."""

    def __init__(self, configuration_scope: str) -> None:
        if not isinstance(configuration_scope, str) or not configuration_scope.strip():
            raise TypeError("configuration_scope must be a non-empty str")
        self._configuration_scope = configuration_scope

    @property
    def configuration_scope(self) -> str:
        return self._configuration_scope

    @abstractmethod
    def to_dictionary(self) -> dict[str, object]:
        """Serialize all configuration fields into a plain JSON-compatible dictionary."""
