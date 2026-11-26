"""Exports for the MatrixOperator sub-family."""

from neurips_bci.operator.matrix_operators.CovarianceMatrixOperator import CovarianceMatrixOperator
from neurips_bci.operator.matrix_operators.MatrixOperator import MatrixOperator
from neurips_bci.operator.matrix_operators.SubspaceMatrixOperator import SubspaceMatrixOperator

__all__ = ["MatrixOperator", "CovarianceMatrixOperator", "SubspaceMatrixOperator"]
