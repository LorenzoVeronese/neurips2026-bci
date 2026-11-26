"""Abstract root class for causal session-stateful Adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Adapter(ABC):
    """Base contract for causal B=1 signal whiteners, artifact guards, balancers, and reliability gates."""

    def __init__(self, adapter_role: str) -> None:
        if not isinstance(adapter_role, str) or not adapter_role.strip():
            raise TypeError("adapter_role must be a non-empty str")
        self._adapter_role = adapter_role

    @property
    def adapter_role(self) -> str:
        return self._adapter_role

    @abstractmethod
    def reset_session_state(self) -> None:
        """Clear all per-session causal buffers so zero state leaks across sessions."""
