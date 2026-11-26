"""Symmetric positive-definite covariance, Euclidean alignment, and Riemannian tangent operator."""

from __future__ import annotations

import numpy as np

from neurips_bci.operator.matrix_operators.MatrixOperator import MatrixOperator
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class CovarianceMatrixOperator(MatrixOperator):
    """Computes Ledoit-Wolf covariances, matrix square-roots/logarithms, and tangent-space projections."""

    def __init__(self, eigenvalue_floor: float = 1e-12) -> None:
        super().__init__(operator_domain="symmetric_positive_definite_manifold")
        self._eigenvalue_floor = ParameterTypeValidator.require_float(
            eigenvalue_floor, "eigenvalue_floor", minimum_bound=0.0, strict_lower=True
        )

    @property
    def eigenvalue_floor(self) -> float:
        return self._eigenvalue_floor

    def describe_operator(self) -> str:
        return f"SPD manifold operator (eigenvalue_floor={self._eigenvalue_floor:.1e})"

    @staticmethod
    def apply_eigenvalue_function(symmetric_matrices: np.ndarray, scalar_transform) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrices)
        transformed_eigenvalues = scalar_transform(eigenvalues)
        return (eigenvectors * transformed_eigenvalues[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2)

    @classmethod
    def matrix_exponential(cls, symmetric_matrices: np.ndarray) -> np.ndarray:
        return cls.apply_eigenvalue_function(symmetric_matrices, np.exp)

    @classmethod
    def inverse_matrix_square_root(cls, spd_matrices: np.ndarray, floor_epsilon: float = 1e-12) -> np.ndarray:
        return cls.apply_eigenvalue_function(
            spd_matrices, lambda eig_vals: 1.0 / np.sqrt(np.maximum(eig_vals, floor_epsilon))
        )

    @classmethod
    def matrix_square_root(cls, spd_matrices: np.ndarray, floor_epsilon: float = 1e-12) -> np.ndarray:
        return cls.apply_eigenvalue_function(
            spd_matrices, lambda eig_vals: np.sqrt(np.maximum(eig_vals, floor_epsilon))
        )

    @classmethod
    def matrix_logarithm(cls, spd_matrices: np.ndarray, floor_epsilon: float = 1e-12) -> np.ndarray:
        return cls.apply_eigenvalue_function(
            spd_matrices, lambda eig_vals: np.log(np.maximum(eig_vals, floor_epsilon))
        )

    @staticmethod
    def ledoit_wolf_covariance(window_batch: np.ndarray) -> np.ndarray:
        centered_batch = np.asarray(window_batch, dtype=np.float64)
        centered_batch = centered_batch - centered_batch.mean(axis=-1, keepdims=True)
        _, channel_count, sample_count = centered_batch.shape
        sample_covariance = centered_batch @ np.swapaxes(centered_batch, 1, 2) / float(sample_count)
        identity_matrix = np.eye(channel_count, dtype=np.float64)
        trace_mean = np.trace(sample_covariance, axis1=1, axis2=2) / float(channel_count)
        delta_term = ((sample_covariance - trace_mean[:, None, None] * identity_matrix) ** 2).sum((1, 2)) / float(
            channel_count
        )
        squared_batch = centered_batch ** 2
        beta_term = (
            (squared_batch @ np.swapaxes(squared_batch, 1, 2)) / float(sample_count) - sample_covariance ** 2
        ).sum((1, 2)) / float(channel_count * sample_count)
        beta_term = np.minimum(beta_term, delta_term)
        shrinkage_alpha = np.where(delta_term > 0.0, beta_term / np.maximum(delta_term, 1e-300), 0.0)
        return (1.0 - shrinkage_alpha)[:, None, None] * sample_covariance + (
            shrinkage_alpha * trace_mean
        )[:, None, None] * identity_matrix

    @staticmethod
    def sample_covariance(window_batch: np.ndarray) -> np.ndarray:
        centered_batch = np.asarray(window_batch, dtype=np.float64)
        centered_batch = centered_batch - centered_batch.mean(axis=-1, keepdims=True)
        return centered_batch @ np.swapaxes(centered_batch, 1, 2) / float(centered_batch.shape[-1])

    @staticmethod
    def trace_shrunk_covariance(window_matrix: np.ndarray, shrinkage_lambda: float = 0.10) -> np.ndarray:
        centered_window = np.asarray(window_matrix, dtype=np.float64)
        centered_window = centered_window - centered_window.mean(axis=-1, keepdims=True)
        channel_count, sample_count = centered_window.shape
        raw_covariance = (centered_window @ centered_window.T) / float(sample_count)
        mean_variance = float(np.trace(raw_covariance)) / float(channel_count)
        return (1.0 - shrinkage_lambda) * raw_covariance + shrinkage_lambda * max(mean_variance, 1e-12) * np.eye(
            channel_count, dtype=np.float64
        )

    @classmethod
    def project_to_tangent_vector(cls, aligned_spd_matrices: np.ndarray) -> np.ndarray:
        channel_count = aligned_spd_matrices.shape[-1]
        upper_rows, upper_cols = np.triu_indices(channel_count)
        off_diagonal_scale = np.where(upper_rows == upper_cols, 1.0, np.sqrt(2.0))
        log_matrices = cls.matrix_logarithm(aligned_spd_matrices)
        return log_matrices[..., upper_rows, upper_cols] * off_diagonal_scale

    @classmethod
    def compute_cnn_whitener(cls, reference_covariances: np.ndarray, scale_mode: str = "none") -> np.ndarray:
        ParameterTypeValidator.require_choice(scale_mode, ("none", "zstd"), "scale_mode")
        whitener_matrices = cls.inverse_matrix_square_root(reference_covariances)
        if scale_mode == "zstd":
            broadband_ref = reference_covariances[..., -1, :, :]
            diag_std = np.sqrt(np.maximum(np.diagonal(broadband_ref, axis1=-2, axis2=-1), 1e-12))
            corr_matrix = broadband_ref / (diag_std[..., :, None] * diag_std[..., None, :])
            whitener_matrices[..., -1, :, :] = cls.inverse_matrix_square_root(corr_matrix) / diag_std[..., None, :]
        return whitener_matrices

    @staticmethod
    def centered_log_probabilities(probabilities: np.ndarray) -> np.ndarray:
        clipped_probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
        log_probs = np.log(clipped_probs)
        return log_probs - np.mean(log_probs, axis=-1, keepdims=True)

    @staticmethod
    def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
        shifted_logits = np.asarray(logits, dtype=np.float64) - np.max(logits, axis=axis, keepdims=True)
        exponentials = np.exp(shifted_logits)
        return exponentials / np.sum(exponentials, axis=axis, keepdims=True)

    @staticmethod
    def sinkhorn_knopp_balance(probability_matrix: np.ndarray, iteration_count: int = 10) -> np.ndarray:
        balanced_matrix = np.maximum(np.asarray(probability_matrix, dtype=np.float64), 1e-12)
        row_count, class_count = balanced_matrix.shape
        for _ in range(iteration_count):
            balanced_matrix = balanced_matrix / balanced_matrix.sum(axis=1, keepdims=True)
            balanced_matrix = balanced_matrix / (
                balanced_matrix.sum(axis=0, keepdims=True) * float(class_count) / float(row_count)
            )
        return balanced_matrix
