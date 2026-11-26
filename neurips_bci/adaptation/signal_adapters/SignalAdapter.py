"""Abstract base class for causal B=1 EEG signal adapters (artifact guards and manifold whiteners)."""

from __future__ import annotations

from abc import ABC

from neurips_bci.adaptation.Adapter import Adapter


class SignalAdapter(Adapter, ABC):
    """Base contract for per-session causal time-series sanitizers and Riemannian/Euclidean aligners."""

    def __init__(self, adapter_role: str) -> None:
        super().__init__(adapter_role=adapter_role)
