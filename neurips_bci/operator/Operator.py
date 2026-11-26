"""Abstract root class for stateless mathematical and signal Operators."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Operator(ABC):
    """Base contract for stateless mathematical, Riemannian, subspace, and spectral operators."""

    def __init__(self, operator_domain: str) -> None:
        if not isinstance(operator_domain, str) or not operator_domain.strip():
            raise TypeError("operator_domain must be a non-empty str")
        self._operator_domain = operator_domain

    @property
    def operator_domain(self) -> str:
        return self._operator_domain

    @abstractmethod
    def describe_operator(self) -> str:
        """Return a concise description of the operator domain."""
