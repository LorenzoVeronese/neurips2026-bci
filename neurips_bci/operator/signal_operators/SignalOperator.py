"""Abstract parent for the SignalOperator sub-family."""

from __future__ import annotations

from abc import ABC

from neurips_bci.operator.Operator import Operator


class SignalOperator(Operator, ABC):
    """Base class for spectral filtering, resampling, referencing, and band-power extraction."""

    def __init__(self, operator_domain: str) -> None:
        super().__init__(operator_domain=operator_domain)
