"""Abstract parent class for foundation-model decoding organs in NeurIPS BCI."""

from __future__ import annotations

from abc import ABC
import numpy as np
from scipy.optimize import minimize
from scipy.signal import resample_poly
import torch

from neurips_bci.backbone.foundation_backbones.FoundationBackbone import FoundationBackbone
from neurips_bci.operator.matrix_operators.CovarianceMatrixOperator import CovarianceMatrixOperator
from neurips_bci.organ.Organ import Organ


class FoundationOrgan(Organ, ABC):
    """Shared frontend preprocessing and collapsed readout fitting for foundation organs."""

    _CACHED_RESAMPLE_200_MATRIX: np.ndarray | None = None

    def __init__(
        self,
        organ_identifier: str,
        backbone: FoundationBackbone | None,
        execution_device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(organ_identifier=organ_identifier)
        if backbone is not None and not isinstance(backbone, FoundationBackbone):
            raise TypeError(f"backbone must be FoundationBackbone or None, got {type(backbone).__name__}")
        self._backbone = backbone
        self._execution_device = torch.device(execution_device)

    @property
    def backbone(self) -> FoundationBackbone | None:
        return self._backbone

    @property
    def execution_device(self) -> torch.device:
        return self._execution_device

    @classmethod
    def prepare_200hz_car_window(cls, raw_window: np.ndarray) -> tuple[np.ndarray, bool]:
        """Resample a `(41, 480)` 120 Hz window to `(41, 800)` at 200 Hz with CAR/100 scaling."""
        is_extreme = bool(np.max(np.abs(raw_window)) > 5000.0)
        source_f64 = np.asarray(
            np.clip(raw_window, -5000.0, 5000.0) if is_extreme else raw_window,
            dtype=np.float64,
        )
        if source_f64.shape[-1] == 480:
            if cls._CACHED_RESAMPLE_200_MATRIX is None:
                cls._CACHED_RESAMPLE_200_MATRIX = resample_poly(
                    np.eye(480, dtype=np.float64), 5, 3, axis=-1
                )
            resampled_200 = source_f64 @ cls._CACHED_RESAMPLE_200_MATRIX
        else:
            resampled_200 = resample_poly(source_f64, 5, 3, axis=-1)
        car_scaled = ((resampled_200 - resampled_200.mean(axis=0, keepdims=True)) / 100.0).astype(np.float32)
        if not np.isfinite(car_scaled).all():
            car_scaled = np.nan_to_num(car_scaled, nan=0.0, posinf=50.0, neginf=-50.0).astype(np.float32)
            is_extreme = True
        return car_scaled, is_extreme

    @staticmethod
    def fit_multinomial_logistic_lbfgs(
        design_matrix: np.ndarray,
        class_labels: np.ndarray,
        inverse_reg_c: float = 0.02,
        sample_weights: np.ndarray | None = None,
        anchor_weights: np.ndarray | None = None,
        anchor_intercept: np.ndarray | None = None,
        pull_strength: float = 1.0,
        max_iterations: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit L2-penalized (or anchor-pulled) multinomial logistic regression in float64 via L-BFGS-B."""
        feat_f64 = np.asarray(design_matrix, dtype=np.float64)
        labels_i64 = np.asarray(class_labels, dtype=np.int64)
        sample_cnt, feat_dim = feat_f64.shape
        class_cnt = int(anchor_weights.shape[0]) if anchor_weights is not None else max(3, int(labels_i64.max()) + 1)
        one_hot_targets = np.eye(class_cnt, dtype=np.float64)[labels_i64]
        weights_vec = (
            np.ones(sample_cnt, dtype=np.float64)
            if sample_weights is None
            else np.asarray(sample_weights, dtype=np.float64)
        )
        w0_mat = (
            np.zeros((class_cnt, feat_dim), dtype=np.float64)
            if anchor_weights is None
            else np.asarray(anchor_weights, dtype=np.float64)
        )
        b0_vec = (
            np.zeros(class_cnt, dtype=np.float64)
            if anchor_intercept is None
            else np.asarray(anchor_intercept, dtype=np.float64)
        )
        reg_lambda = float(pull_strength) / max(float(inverse_reg_c), 1e-12)

        def _objective_and_grad(flat_params: np.ndarray) -> tuple[float, np.ndarray]:
            cur_w = flat_params[: class_cnt * feat_dim].reshape(class_cnt, feat_dim)
            cur_b = flat_params[class_cnt * feat_dim :]
            logits = feat_f64 @ cur_w.T + cur_b
            logits = logits - logits.max(axis=1, keepdims=True)
            log_sum_exp = np.log(np.exp(logits).sum(axis=1, keepdims=True))
            probs = np.exp(logits - log_sum_exp)
            grad_logits = (probs - one_hot_targets) * weights_vec[:, None]
            diff_w = cur_w - w0_mat
            loss_scalar = float(
                -(weights_vec[:, None] * one_hot_targets * (logits - log_sum_exp)).sum()
                + 0.5 * reg_lambda * (diff_w ** 2).sum()
            )
            grad_w = grad_logits.T @ feat_f64 + reg_lambda * diff_w
            grad_b = grad_logits.sum(axis=0)
            return loss_scalar, np.concatenate([grad_w.ravel(), grad_b])

        init_vec = np.concatenate([w0_mat.ravel(), b0_vec])
        try:
            import threadpoolctl

            with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
                opt_out = minimize(
                    _objective_and_grad,
                    init_vec,
                    jac=True,
                    method="L-BFGS-B",
                    options={"maxiter": int(max_iterations)},
                )
        except Exception:
            opt_out = minimize(
                _objective_and_grad,
                init_vec,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": int(max_iterations)},
            )
        fitted_w = opt_out.x[: class_cnt * feat_dim].reshape(class_cnt, feat_dim)
        fitted_b = opt_out.x[class_cnt * feat_dim :]
        return fitted_w, fitted_b

    @classmethod
    def fit_collapsed_svd_logistic_readout(
        cls,
        feature_matrix: np.ndarray,
        class_labels: np.ndarray,
        svd_components: int = 128,
        inverse_reg_c: float = 0.02,
        w_eff_transposed: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit `StandardScaler -> SVD(k) -> LogisticRegression(C)` and collapse into `(mu0, W_eff, b_eff)`."""
        feats_f64 = np.asarray(feature_matrix, dtype=np.float64)
        mu0 = feats_f64.mean(axis=0)
        sd0 = np.maximum(feats_f64.std(axis=0), 1e-6)
        standardized = (feats_f64 - mu0[None, :]) / sd0[None, :]
        eff_k = min(int(svd_components), standardized.shape[0], standardized.shape[1])
        _, _, vh_mat = np.linalg.svd(standardized, full_matrices=False)
        proj_basis = vh_mat[:eff_k].T
        reduced_feats = standardized @ proj_basis
        lr_w, lr_b = cls.fit_multinomial_logistic_lbfgs(
            reduced_feats, class_labels, inverse_reg_c=inverse_reg_c, pull_strength=1.0
        )
        # Collapsed linear map acting on centered raw features `(f - mu0)`:
        # logits = ((f - mu0) / sd0) @ proj_basis @ lr_w.T + lr_b
        w_d_by_k = (proj_basis @ lr_w.T) / sd0[:, None]
        b_centered = lr_b - np.mean(lr_b)
        if w_eff_transposed:
            return mu0, w_d_by_k, b_centered
        return mu0, w_d_by_k.T, b_centered

    @staticmethod
    def causal_ratio_balance(
        raw_probs: np.ndarray,
        cumulative_prob_sum: np.ndarray,
        effective_count: int,
        prior_strength: float = 10.0,
    ) -> np.ndarray:
        """Apply strictly causal marginal ratio balancing against running probability sum."""
        marginal_bar = np.clip(
            (float(prior_strength) / 3.0 + cumulative_prob_sum) / (float(prior_strength) + float(effective_count)),
            1e-6,
            1.0,
        )
        adjusted = CovarianceMatrixOperator.stable_softmax(np.log(np.clip(raw_probs, 1e-12, 1.0))) / marginal_bar
        return adjusted / float(np.sum(adjusted))
