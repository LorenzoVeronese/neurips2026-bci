"""Abstract root class for the Validator family."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Validator(ABC):
    """Base contract for construction-time and artifact wiring validators."""

    def __init__(self, validator_scope: str) -> None:
        if not isinstance(validator_scope, str) or not validator_scope.strip():
            raise TypeError("validator_scope must be a non-empty str")
        self._validator_scope = validator_scope

    @property
    def validator_scope(self) -> str:
        return self._validator_scope

    @abstractmethod
    def verify_contract(self) -> bool:
        """Validate the configured invariant and raise on failure."""
