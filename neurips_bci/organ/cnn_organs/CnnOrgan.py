"""Abstract parent class for multi-seed convolutional ensemble decoding organs."""

from __future__ import annotations

from abc import ABC
import numpy as np
from scipy.optimize import minimize

from neurips_bci.operator.matrix_operators.CovarianceMatrixOperator import CovarianceMatrixOperator
from neurips_bci.organ.Organ import Organ


class CnnOrgan(Organ, ABC):
    """Shared whitening, probability calibration, and regularized logistic-refit utilities for CNN organs."""

    @staticmethod
    def centered_log_probabilities(prob_vector: np.ndarray, floor_epsilon: float = 1e-12) -> np.ndarray:
        clipped_probs = np.clip(np.asarray(prob_vector, dtype=np.float64), floor_epsilon, 1.0)
        log_probs = np.log(clipped_probs)
        return log_probs - float(np.mean(log_probs))

    @staticmethod
    def balance_probability_step(
        raw_probs: np.ndarray,
        running_sum: np.ndarray,
        step_count: int,
        prior_pseudo_count: float = 10.0,
    ) -> np.ndarray:
        prob_vec = np.asarray(raw_probs, dtype=np.float64)
        class_count = len(prob_vec)
        running_mean = (prior_pseudo_count / float(class_count) + running_sum) / (
            prior_pseudo_count + float(step_count)
        )
        debiased = prob_vec / np.maximum(running_mean, 1e-12)
        return debiased / float(np.sum(debiased))

    @staticmethod
    def zstd_whitener(reference_cov: np.ndarray) -> np.ndarray:
        diag_std = np.sqrt(np.maximum(np.diagonal(reference_cov, axis1=-2, axis2=-1), 1e-12))
        corr_mat = reference_cov / (diag_std[..., :, None] * diag_std[..., None, :])
        return CovarianceMatrixOperator.inverse_matrix_square_root(corr_mat) / diag_std[..., None, :]

    @classmethod
    def cnn_whitener(
        cls,
        reference_covs: np.ndarray,
        ea_scale: str = "none",
        whiten_last_broadband: bool = True,
    ) -> np.ndarray:
        whiten_mats = CovarianceMatrixOperator.inverse_matrix_square_root(reference_covs)
        if ea_scale == "zstd" and whiten_last_broadband:
            whiten_mats = whiten_mats.copy()
            whiten_mats[..., -1, :, :] = cls.zstd_whitener(reference_covs[..., -1, :, :])
        return whiten_mats

    @staticmethod
    def refit_logistic_regression_toward(
        coef0: np.ndarray,
        intercept0: np.ndarray,
        feat_matrix: np.ndarray,
        pseudo_labels: np.ndarray,
        sample_weights: np.ndarray,
        lr_c: float = 0.1,
        pull_strength: float = 3.0,
        max_iter: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        class_count, feat_dim = coef0.shape
        one_hot_y = np.eye(class_count, dtype=np.float64)[pseudo_labels]
        reg_lambda = float(pull_strength) / float(lr_c)

        def _objective_and_grad(flat_theta: np.ndarray) -> tuple[float, np.ndarray]:
            weight_mat = flat_theta[: class_count * feat_dim].reshape(class_count, feat_dim)
            bias_vec = flat_theta[class_count * feat_dim :]
            logits = feat_matrix @ weight_mat.T + bias_vec
            logits = logits - logits.max(axis=1, keepdims=True)
            log_sum_exp = np.log(np.exp(logits).sum(axis=1, keepdims=True))
            grad_logits = (np.exp(logits - log_sum_exp) - one_hot_y) * sample_weights[:, None]
            diff_w = weight_mat - coef0
            obj_loss = float(
                -(sample_weights[:, None] * one_hot_y * (logits - log_sum_exp)).sum()
                + 0.5 * reg_lambda * (diff_w ** 2).sum()
            )
            grad_flat = np.concatenate(
                [(grad_logits.T @ feat_matrix + reg_lambda * diff_w).ravel(), grad_logits.sum(axis=0)]
            )
            return obj_loss, grad_flat

        init_theta = np.concatenate([coef0.ravel(), intercept0]).astype(np.float64)
        opt_out = minimize(_objective_and_grad, init_theta, jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
        fitted_coef = opt_out.x[: class_count * feat_dim].reshape(class_count, feat_dim)
        fitted_inter = opt_out.x[class_count * feat_dim :]
        return fitted_coef, fitted_inter
