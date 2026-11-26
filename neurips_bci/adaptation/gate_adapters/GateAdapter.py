"""Abstract base class for causal B=1 reliability, confidence-ceiling, and deadline pace gates."""

from __future__ import annotations

from abc import ABC

from neurips_bci.adaptation.Adapter import Adapter


class GateAdapter(Adapter, ABC):
    """Base contract for session-stateful fusion gates and wall-clock deadline governors."""

    def __init__(self, adapter_role: str) -> None:
        super().__init__(adapter_role=adapter_role)
