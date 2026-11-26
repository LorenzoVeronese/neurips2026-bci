"""Causal per-session filter-bank Common Spatial Patterns + Ledoit-Wolf shrinkage LDA organ."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.linalg import eigh, lstsq

from neurips_bci.container.stream_containers.WindowedCorpusStreamContainer import WindowedCorpusStreamContainer
from neurips_bci.operator.matrix_operators.CovarianceMatrixOperator import CovarianceMatrixOperator
from neurips_bci.operator.signal_operators.SpectralSignalOperator import SpectralSignalOperator
from neurips_bci.organ.riemannian_organs.RiemannianOrgan import RiemannianOrgan
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class SessionCspLdaRiemannianOrgan(RiemannianOrgan):
    """Refits 6-band soft-posterior adaptive-shrinkage one-vs-rest CSP + Ledoit-Wolf sLDA causally within a session."""

    ARTIFACT_FILENAME = "csp.npz"
    SIX_CSP_BANDS_HZ: tuple[tuple[int, int], ...] = (
        (4, 8),
        (8, 13),
        (13, 20),
        (20, 30),
        (30, 45),
        (8, 30),
    )

    def __init__(
        self,
        train_reference_covs: np.ndarray,
        train_class_covs: np.ndarray,
        csp_weight: float = 0.10,
        adaptive_shrinkage: float = 10.0,
        ea_prior_count: float = 10.0,
        csp_pairs_per_end: int = 2,
        active_from_window: int = 66,
        refit_checkpoints: tuple[int, ...] = (66, 96, 126, 156),
        sampling_rate_hz: float = 120.0,
    ) -> None:
        super().__init__(organ_identifier="session_csp_lda_riemannian")
        self._train_ref = np.asarray(
            ParameterTypeValidator.require_numpy_array(train_reference_covs, "train_reference_covs", expected_ndim=3),
            dtype=np.float64,
        )
        self._train_cls = np.asarray(
            ParameterTypeValidator.require_numpy_array(train_class_covs, "train_class_covs", expected_ndim=4),
            dtype=np.float64,
        )
        self._csp_weight = ParameterTypeValidator.require_float(csp_weight, "csp_weight", 0.0)
        self._adaptive_shrinkage = ParameterTypeValidator.require_float(adaptive_shrinkage, "adaptive_shrinkage", 0.0)
        self._ea_prior = ParameterTypeValidator.require_float(ea_prior_count, "ea_prior_count", 0.0)
        self._csp_pairs = ParameterTypeValidator.require_integer(csp_pairs_per_end, "csp_pairs_per_end", 1)
        self._active_from = ParameterTypeValidator.require_integer(active_from_window, "active_from_window", 0)
        self._refit_checkpoints = tuple(int(ckpt) for ckpt in refit_checkpoints)
        self._sampling_rate_hz = ParameterTypeValidator.require_float(sampling_rate_hz, "sampling_rate_hz", 1.0)
        self._sos_bank = SpectralSignalOperator.build_butterworth_sos_bank(
            [list(band_pair) for band_pair in self.SIX_CSP_BANDS_HZ],
            sampling_rate_hz=self._sampling_rate_hz,
        )

        self._cov_sum = np.zeros_like(self._train_ref)
        self._aligned_covs: list[np.ndarray] = []
        self._teacher_probs: list[np.ndarray] = []
        self._skipped_flags: list[bool] = []
        self._seen_count = 0
        self._clean_count = 0
        self._fitted_decoder: tuple | None = None
        self._last_log_probs: np.ndarray | None = None
        self._last_band_covs: np.ndarray | None = None
        self.reset_organ_state()

    @property
    def csp_weight(self) -> float:
        return self._csp_weight

    @property
    def refit_checkpoints(self) -> tuple[int, ...]:
        return self._refit_checkpoints

    def set_refit_checkpoints(self, checkpoints: tuple[int, ...] | list[int]) -> None:
        self._refit_checkpoints = tuple(int(ckpt) for ckpt in checkpoints)

    @property
    def last_band_covariances(self) -> np.ndarray | None:
        return self._last_band_covs

    @property
    def last_aligned_covariance(self) -> np.ndarray | None:
        return self._aligned_covs[-1] if self._aligned_covs else None

    def reset_organ_state(self) -> None:
        self._cov_sum = np.zeros_like(self._train_ref)
        self._aligned_covs = []
        self._teacher_probs = []
        self._skipped_flags = []
        self._seen_count = 0
        self._clean_count = 0
        self._fitted_decoder = None
        self._last_log_probs = None
        self._last_band_covs = None

    @classmethod
    def fit_from_corpus(
        cls,
        corpus_container: WindowedCorpusStreamContainer,
        csp_weight: float = 0.10,
        adaptive_shrinkage: float = 10.0,
    ) -> SessionCspLdaRiemannianOrgan:
        raw_windows = np.asarray(corpus_container.eeg_windows, dtype=np.float64)
        demeaned_windows = raw_windows - raw_windows.mean(axis=-1, keepdims=True)
        band_covs = SpectralSignalOperator.compute_band_covariances(
            demeaned_windows,
            [list(band_pair) for band_pair in cls.SIX_CSP_BANDS_HZ],
            sampling_rate_hz=corpus_container.sampling_rate_hz,
        )[:, :-1]
        sessions = np.asarray(corpus_container.session_ids)
        labels = np.asarray(corpus_container.class_labels, dtype=np.int64)
        sess_refs = np.empty_like(band_covs)
        for sess_id in np.unique(sessions):
            sess_mask = sessions == sess_id
            sess_refs[sess_mask] = band_covs[sess_mask].mean(axis=0)
        whiteners = CovarianceMatrixOperator.inverse_matrix_square_root(sess_refs)
        aligned_all = whiteners @ band_covs @ whiteners
        class_means = []
        for cls_idx in range(3):
            cls_mask = labels == cls_idx
            class_means.append(
                aligned_all[cls_mask].mean(axis=0) if np.any(cls_mask) else aligned_all.mean(axis=0)
            )
        return cls(
            train_reference_covs=band_covs.mean(axis=0),
            train_class_covs=np.stack(class_means, axis=0),
            csp_weight=csp_weight,
            adaptive_shrinkage=adaptive_shrinkage,
            sampling_rate_hz=corpus_container.sampling_rate_hz,
        )

    @classmethod
    def load_from_npz(
        cls,
        csp_npz_path: Path,
        csp_weight: float = 0.10,
        adaptive_shrinkage: float = 10.0,
        refit_checkpoints: tuple[int, ...] = (66, 96, 126, 156),
    ) -> SessionCspLdaRiemannianOrgan:
        resolved_path = ParameterTypeValidator.require_path(csp_npz_path, "csp_npz_path", must_exist=True)
        with np.load(resolved_path, allow_pickle=False) as archive:
            train_ref = archive["csp_train_ref"].copy()
            train_cls = archive["csp_train_class"].copy()
        return cls(
            train_reference_covs=train_ref,
            train_class_covs=train_cls,
            csp_weight=csp_weight,
            adaptive_shrinkage=adaptive_shrinkage,
            refit_checkpoints=refit_checkpoints,
        )

    def save_organ_artifacts(self, target_directory: Path) -> list[Path]:
        resolved_dir = ParameterTypeValidator.require_path(target_directory, "target_directory")
        resolved_dir.mkdir(parents=True, exist_ok=True)
        out_path = resolved_dir / self.ARTIFACT_FILENAME
        np.savez(
            out_path,
            csp_train_ref=self._train_ref.astype(np.float64),
            csp_train_class=self._train_cls.astype(np.float64),
        )
        return [out_path]

    def observe_teacher_probabilities(self, teacher_probs: np.ndarray) -> None:
        self._teacher_probs.append(np.asarray(teacher_probs, dtype=np.float64).copy())

    def additive_log_term(self) -> np.ndarray:
        class_count = self._train_cls.shape[0]
        if self._last_log_probs is None:
            return np.zeros(class_count, dtype=np.float64)
        return self._csp_weight * self._last_log_probs

    def log_odds_correction(self, base_log_prob: np.ndarray | None = None) -> np.ndarray:
        class_count = self._train_cls.shape[0]
        if self._last_log_probs is None:
            return np.zeros(class_count, dtype=np.float64)
        if base_log_prob is None:
            return self._csp_weight * self._last_log_probs
        return self._csp_weight * (self._last_log_probs - np.asarray(base_log_prob, dtype=np.float64))

    def step_causal_window(
        self,
        cleaned_window: np.ndarray,
        is_artifact_flagged: bool,
        known_covariances: dict | None = None,
    ) -> np.ndarray:
        win_idx = self._seen_count
        win_f64 = np.asarray(cleaned_window, dtype=np.float64)
        demeaned_win = win_f64 - win_f64.mean(axis=-1, keepdims=True)
        band_covs = self.window_band_covariances(
            demeaned_win,
            self.SIX_CSP_BANDS_HZ,
            self._sos_bank,
            known_covariances=known_covariances,
            include_broadband=False,
            sampling_rate_hz=self._sampling_rate_hz,
        )
        self._last_band_covs = band_covs
        if not is_artifact_flagged:
            self._cov_sum = self._cov_sum + band_covs
            self._clean_count += 1
        self._skipped_flags.append(bool(is_artifact_flagged))

        ref_cov = (self._ea_prior * self._train_ref + self._cov_sum) / (self._ea_prior + float(self._clean_count))
        whitener = CovarianceMatrixOperator.inverse_matrix_square_root(ref_cov)
        self._aligned_covs.append(whitener @ band_covs @ whitener)
        self._seen_count += 1

        if win_idx in self._refit_checkpoints and len(self._teacher_probs) == win_idx:
            self._refit_at_checkpoint(win_idx)

        self._last_log_probs = None
        if self._fitted_decoder is not None and win_idx >= self._active_from:
            self._last_log_probs = self._decode_log_proba(self._fitted_decoder, self._aligned_covs[-1][None])[0]
        class_count = self._train_cls.shape[0]
        return (
            np.full(class_count, 1.0 / float(class_count), dtype=np.float64)
            if self._last_log_probs is None
            else np.exp(self._last_log_probs)
        )

    def step_reusing_peer_covariances(
        self,
        peer_organ: SessionCspLdaRiemannianOrgan,
        is_artifact_flagged: bool,
    ) -> np.ndarray:
        win_idx = self._seen_count
        band_covs = peer_organ.last_band_covariances
        aligned_cov = peer_organ.last_aligned_covariance
        if band_covs is None or aligned_cov is None:
            raise RuntimeError("peer_organ must run step_causal_window before step_reusing_peer_covariances")
        self._last_band_covs = band_covs
        if not is_artifact_flagged:
            self._cov_sum = self._cov_sum + band_covs
            self._clean_count += 1
        self._skipped_flags.append(bool(is_artifact_flagged))
        self._aligned_covs.append(aligned_cov)
        self._seen_count += 1

        if win_idx in self._refit_checkpoints and len(self._teacher_probs) == win_idx:
            self._refit_at_checkpoint(win_idx)

        self._last_log_probs = None
        if self._fitted_decoder is not None and win_idx >= self._active_from:
            self._last_log_probs = self._decode_log_proba(self._fitted_decoder, self._aligned_covs[-1][None])[0]
        class_count = self._train_cls.shape[0]
        return (
            np.full(class_count, 1.0 / float(class_count), dtype=np.float64)
            if self._last_log_probs is None
            else np.exp(self._last_log_probs)
        )

    def _refit_at_checkpoint(self, win_idx: int) -> None:
        ok_mask = ~np.asarray(self._skipped_flags[:win_idx], dtype=bool)
        ok_slice = slice(None) if bool(ok_mask.all()) else ok_mask
        aligned_past = np.stack(self._aligned_covs[:win_idx], axis=0)[ok_slice]
        probs_past = np.stack(self._teacher_probs[:win_idx], axis=0)[ok_slice]
        if len(aligned_past) < 2:
            return
        assigned_y = probs_past.argmax(axis=1)
        if len(np.unique(assigned_y)) < 2:
            return
        conf_weights = probs_past[np.arange(len(assigned_y)), assigned_y]
        self._fitted_decoder = self._fit_csp_lda_decoder(aligned_past, assigned_y, conf_weights)

    def _fit_csp_lda_decoder(
        self,
        aligned_covs: np.ndarray,
        assigned_labels: np.ndarray,
        conf_weights: np.ndarray,
    ) -> tuple:
        class_count, band_count = self._train_cls.shape[:2]
        shrunk_class_covs = []
        for cls_idx in range(class_count):
            cls_mask = assigned_labels == cls_idx
            cls_w = conf_weights[cls_mask]
            eff_mass = float(cls_w.sum())
            if eff_mass <= 1e-9:
                shrunk_class_covs.append(self._train_cls[cls_idx])
                continue
            shrink_beta = (
                self._adaptive_shrinkage / (self._adaptive_shrinkage + eff_mass)
                if self._adaptive_shrinkage > 0.0
                else 0.5
            )
            weighted_mean = np.tensordot(cls_w / eff_mass, aligned_covs[cls_mask], axes=(0, 0))
            shrunk_class_covs.append((1.0 - shrink_beta) * weighted_mean + shrink_beta * self._train_cls[cls_idx])
        stacked_ck = np.stack(shrunk_class_covs, axis=0)
        total_cov = stacked_ck.sum(axis=0)

        spatial_filters = []
        for cls_idx in range(class_count):
            for band_idx in range(band_count):
                _, eig_vecs = eigh(stacked_ck[cls_idx, band_idx], total_cov[band_idx])
                spatial_filters.append(
                    np.concatenate([eig_vecs[:, : self._csp_pairs], eig_vecs[:, -self._csp_pairs :]], axis=1)
                )
        csp_features = self._extract_csp_logvar_features(spatial_filters, aligned_covs)
        lda_params = self._fit_weighted_shrinkage_lda(csp_features, assigned_labels, class_count, conf_weights)
        return (spatial_filters, lda_params)

    @staticmethod
    def _extract_csp_logvar_features(spatial_filters: list[np.ndarray], aligned_covs: np.ndarray) -> np.ndarray:
        band_count = aligned_covs.shape[1]
        return np.concatenate(
            [
                np.log(
                    np.maximum(
                        np.einsum("cd,nce,ed->nd", filt_mat, aligned_covs[:, filt_idx % band_count], filt_mat),
                        1e-12,
                    )
                )
                for filt_idx, filt_mat in enumerate(spatial_filters)
            ],
            axis=1,
        )

    @staticmethod
    def _ledoit_wolf_feature_covariance(feat_matrix: np.ndarray, norm_weights: np.ndarray | None = None) -> np.ndarray:
        row_count_f = float(feat_matrix.shape[0])
        feat_dim = feat_matrix.shape[1]
        if norm_weights is None:
            centered = feat_matrix - feat_matrix.mean(axis=0)
            feat_scale = centered.std(axis=0)
        else:
            centered = feat_matrix - norm_weights @ feat_matrix
            feat_scale = np.sqrt(norm_weights @ (centered ** 2))
            row_count_f = 1.0 / float((norm_weights ** 2).sum())
        feat_scale[feat_scale == 0.0] = 1.0
        std_feat = centered / feat_scale
        if norm_weights is None:
            sample_cov = (std_feat.T @ std_feat) / row_count_f
        else:
            sample_cov = (std_feat * norm_weights[:, None]).T @ std_feat
        mu_trace = float(np.trace(sample_cov)) / float(feat_dim)
        sq_feat = std_feat ** 2
        delta_sq = float((sample_cov ** 2).sum())
        if norm_weights is None:
            beta_est = ((sq_feat.T @ sq_feat).sum() / row_count_f - delta_sq) / (row_count_f * float(feat_dim))
        else:
            beta_est = (((sq_feat * norm_weights[:, None]).T @ sq_feat).sum() - delta_sq) / (
                row_count_f * float(feat_dim)
            )
        delta_est = (delta_sq - 2.0 * mu_trace * float(np.trace(sample_cov)) + float(feat_dim) * (mu_trace ** 2)) / float(
            feat_dim
        )
        beta_est = min(float(beta_est), float(delta_est))
        shrink_alpha = (
            0.0 if (row_count_f < 2.0 or beta_est == 0.0 or delta_est == 0.0) else beta_est / delta_est
        )
        shrunk_cov = (1.0 - shrink_alpha) * sample_cov + shrink_alpha * mu_trace * np.eye(feat_dim)
        return feat_scale[:, None] * shrunk_cov * feat_scale[None, :]

    @classmethod
    def _fit_weighted_shrinkage_lda(
        cls,
        feat_matrix: np.ndarray,
        assigned_labels: np.ndarray,
        total_classes: int,
        conf_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        present_classes = np.unique(assigned_labels)
        priors = np.array([(assigned_labels == cls_idx).mean() for cls_idx in present_classes], dtype=np.float64)
        norm_w_per_cls = [
            conf_weights[assigned_labels == cls_idx] / conf_weights[assigned_labels == cls_idx].sum()
            for cls_idx in present_classes
        ]
        cls_means = np.stack(
            [w_vec @ feat_matrix[assigned_labels == cls_idx] for w_vec, cls_idx in zip(norm_w_per_cls, present_classes)],
            axis=0,
        )
        pooled_cov = sum(
            prior_p * cls._ledoit_wolf_feature_covariance(feat_matrix[assigned_labels == cls_idx], w_vec)
            for prior_p, cls_idx, w_vec in zip(priors, present_classes, norm_w_per_cls)
        )
        coef_mat = lstsq(pooled_cov, cls_means.T)[0].T
        intercept_vec = -0.5 * (coef_mat * cls_means).sum(axis=1) + np.log(priors)
        return present_classes, coef_mat, intercept_vec, total_classes

    @classmethod
    def _decode_log_proba(cls, decoder_tuple: tuple, aligned_covs: np.ndarray) -> np.ndarray:
        spatial_filters, (present_classes, coef_mat, intercept_vec, total_classes) = decoder_tuple
        logits = cls._extract_csp_logvar_features(spatial_filters, aligned_covs) @ coef_mat.T + intercept_vec
        logits = logits - logits.max(axis=1, keepdims=True)
        log_probs = np.full((len(aligned_covs), total_classes), np.log(1e-6), dtype=np.float64)
        log_probs[:, present_classes] = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return log_probs
