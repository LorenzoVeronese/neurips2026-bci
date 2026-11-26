"""Abstract parent for the MatrixOperator sub-family."""

from __future__ import annotations

from abc import ABC

from neurips_bci.operator.Operator import Operator


class MatrixOperator(Operator, ABC):
    """Base class for symmetric positive-definite and linear subspace matrix operators."""

    def __init__(self, operator_domain: str) -> None:
        super().__init__(operator_domain=operator_domain)
