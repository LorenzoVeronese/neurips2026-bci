"""Causal known-participant tangent-space logistic regression specialist router organ."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

from neurips_bci.container.stream_containers.WindowedCorpusStreamContainer import WindowedCorpusStreamContainer
from neurips_bci.operator.matrix_operators.CovarianceMatrixOperator import CovarianceMatrixOperator
from neurips_bci.operator.signal_operators.SpectralSignalOperator import SpectralSignalOperator
from neurips_bci.organ.riemannian_organs.RiemannianOrgan import RiemannianOrgan
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class ParticipantSpecialistRiemannianOrgan(RiemannianOrgan):
    """Scores per-participant TS-LR specialists and gates additive log-odds by strictly-past base agreement."""

    ARTIFACT_FILENAME = "specialists.npz"
    DEFAULT_SIX_BANDS_HZ: tuple[tuple[int, int], ...] = (
        (4, 8),
        (8, 13),
        (13, 20),
        (20, 30),
        (30, 45),
        (1, 45),
    )

    def __init__(
        self,
        participant_labels: list[str] | tuple[str, ...],
        frequency_bands_hz: list[list[float]] | tuple[tuple[float, float], ...],
        prior_covariances_r0: np.ndarray,
        scaler_means: np.ndarray,
        scaler_scales: np.ndarray,
        lr_coefficients: np.ndarray,
        lr_intercepts: np.ndarray,
        specialist_weight: float = 0.20,
        kappa: float = 30.0,
        gap_delta: float = 0.05,
        prior_n0: float = 20.0,
        ea_prior_count: float = 10.0,
        sampling_rate_hz: float = 120.0,
    ) -> None:
        super().__init__(organ_identifier="participant_specialist_riemannian")
        if len(participant_labels) < 2:
            raise ValueError("ParticipantSpecialistRiemannianOrgan requires at least 2 participant specialists")
        self._participant_labels = tuple(str(pid) for pid in participant_labels)
        self._bands = tuple((float(b[0]), float(b[1])) for b in frequency_bands_hz)
        self._r0 = np.asarray(
            ParameterTypeValidator.require_numpy_array(prior_covariances_r0, "prior_covariances_r0", expected_ndim=4),
            dtype=np.float64,
        )
        self._scaler_mean = np.asarray(
            ParameterTypeValidator.require_numpy_array(scaler_means, "scaler_means", expected_ndim=2),
            dtype=np.float64,
        )
        self._scaler_scale = np.asarray(
            ParameterTypeValidator.require_numpy_array(scaler_scales, "scaler_scales", expected_ndim=2),
            dtype=np.float64,
        )
        self._coef = np.asarray(
            ParameterTypeValidator.require_numpy_array(lr_coefficients, "lr_coefficients", expected_ndim=3),
            dtype=np.float64,
        )
        self._intercept = np.asarray(
            ParameterTypeValidator.require_numpy_array(lr_intercepts, "lr_intercepts", expected_ndim=2),
            dtype=np.float64,
        )
        self._specialist_weight = ParameterTypeValidator.require_float(specialist_weight, "specialist_weight", 0.0)
        self._kappa = ParameterTypeValidator.require_float(kappa, "kappa", 0.0)
        self._gap_delta = ParameterTypeValidator.require_float(gap_delta, "gap_delta", 0.0, strict_lower=True)
        self._prior_n0 = ParameterTypeValidator.require_float(prior_n0, "prior_n0", 0.0)
        self._ea_prior = ParameterTypeValidator.require_float(ea_prior_count, "ea_prior_count", 0.0)
        self._sampling_rate_hz = ParameterTypeValidator.require_float(sampling_rate_hz, "sampling_rate_hz", 1.0)
        self._sos_bank = SpectralSignalOperator.build_butterworth_sos_bank(
            [list(band_pair) for band_pair in self._bands],
            sampling_rate_hz=self._sampling_rate_hz,
        )

        spec_count, _, chan_count, _ = self._r0.shape
        self._spec_count = spec_count
        self._chan_count = chan_count
        self._r0_6b = self._r0[:, :-1].copy()
        self._hits = np.zeros(spec_count, dtype=np.float64)
        self._seen_windows = 0
        self._clean_ref_count = 0
        self._cov_sum_6b = np.zeros_like(self._r0[0, :-1])
        self._p0_running_sum: np.ndarray | None = None
        self._last_specialist_log_probs: np.ndarray | None = None
        self._last_block_log_probs: np.ndarray | None = None
        self._last_gate_weight = 0.0
        self.reset_organ_state()

    @property
    def participant_labels(self) -> tuple[str, ...]:
        return self._participant_labels

    @property
    def prior_covariances_r0(self) -> np.ndarray:
        return self._r0

    @property
    def last_specialist_log_probs(self) -> np.ndarray | None:
        return self._last_specialist_log_probs

    def reset_organ_state(self) -> None:
        self._hits = np.zeros(self._spec_count, dtype=np.float64)
        self._seen_windows = 0
        self._clean_ref_count = 0
        self._cov_sum_6b = np.zeros_like(self._r0[0, :-1])
        self._p0_running_sum = None
        self._last_specialist_log_probs = None
        self._last_block_log_probs = None
        self._last_gate_weight = 0.0

    @classmethod
    def fit_from_corpus(
        cls,
        corpus_container: WindowedCorpusStreamContainer,
        specialist_weight: float = 0.20,
        kappa: float = 30.0,
        gap_delta: float = 0.05,
        prior_n0: float = 20.0,
    ) -> ParticipantSpecialistRiemannianOrgan:
        raw_windows = np.asarray(corpus_container.eeg_windows, dtype=np.float64)
        demeaned_windows = raw_windows - raw_windows.mean(axis=-1, keepdims=True)
        labels = np.asarray(corpus_container.class_labels, dtype=np.int64)
        sessions = np.asarray(corpus_container.session_ids)
        participants = np.asarray(corpus_container.participant_ids)
        unique_participants = sorted(set(str(pid) for pid in participants.tolist()))
        if len(unique_participants) < 2:
            unique_sessions = sorted(set(str(sid) for sid in sessions.tolist()))
            if len(unique_sessions) < 2:
                raise ValueError("Corpus must contain at least 2 participants or sessions to fit specialists")
            participants = sessions
            unique_participants = unique_sessions

        all_covs = SpectralSignalOperator.compute_band_covariances(
            demeaned_windows,
            [list(band_pair) for band_pair in cls.DEFAULT_SIX_BANDS_HZ],
            sampling_rate_hz=corpus_container.sampling_rate_hz,
        )
        r0_list, mean_list, scale_list, coef_list, inter_list = [], [], [], [], []
        for part_key in unique_participants:
            part_mask = participants == part_key
            cov_p = all_covs[part_mask]
            sess_p = sessions[part_mask]
            y_p = labels[part_mask]
            r0_p = cov_p.mean(axis=0)
            sess_refs = np.empty_like(cov_p)
            for sess_id in np.unique(sess_p):
                sm = sess_p == sess_id
                sess_refs[sm] = cov_p[sm].mean(axis=0)
            whiteners = CovarianceMatrixOperator.inverse_matrix_square_root(sess_refs)
            tang_feats = CovarianceMatrixOperator.project_to_tangent_vector(
                whiteners[:, :-1] @ cov_p[:, :-1] @ whiteners[:, :-1]
            ).reshape(len(cov_p), -1)
            feat_mean = tang_feats.mean(axis=0)
            feat_scale = tang_feats.std(axis=0) + 1e-8
            std_feats = (tang_feats - feat_mean) / feat_scale
            lr_clf = LogisticRegression(C=0.1, max_iter=3000)
            lr_clf.fit(std_feats, y_p)
            coef_full = np.zeros((3, tang_feats.shape[1]), dtype=np.float64)
            inter_full = np.full(3, -10.0, dtype=np.float64)
            for loc_i, cls_val in enumerate(lr_clf.classes_):
                coef_full[int(cls_val)] = lr_clf.coef_[loc_i]
                inter_full[int(cls_val)] = lr_clf.intercept_[loc_i]
            r0_list.append(r0_p)
            mean_list.append(feat_mean)
            scale_list.append(feat_scale)
            coef_list.append(coef_full)
            inter_list.append(inter_full)

        return cls(
            participant_labels=unique_participants,
            frequency_bands_hz=cls.DEFAULT_SIX_BANDS_HZ,
            prior_covariances_r0=np.stack(r0_list, axis=0),
            scaler_means=np.stack(mean_list, axis=0),
            scaler_scales=np.stack(scale_list, axis=0),
            lr_coefficients=np.stack(coef_list, axis=0),
            lr_intercepts=np.stack(inter_list, axis=0),
            specialist_weight=specialist_weight,
            kappa=kappa,
            gap_delta=gap_delta,
            prior_n0=prior_n0,
            sampling_rate_hz=corpus_container.sampling_rate_hz,
        )

    @classmethod
    def load_from_npz(
        cls,
        specialists_npz_path: Path,
        specialist_weight: float = 0.20,
        kappa: float = 30.0,
        gap_delta: float = 0.05,
        prior_n0: float = 20.0,
    ) -> ParticipantSpecialistRiemannianOrgan:
        resolved_path = ParameterTypeValidator.require_path(
            specialists_npz_path, "specialists_npz_path", must_exist=True
        )
        with np.load(resolved_path, allow_pickle=False) as archive:
            part_names = [str(item) for item in archive["names"].tolist()]
            ea_prior_val = 10.0
            if "bands" in archive.files:
                bands_arr = [list(map(float, row)) for row in archive["bands"].tolist()]
                if "cfgs" in archive.files:
                    parsed_cfgs = json.loads(str(archive["cfgs"]))
                    ea_prior_val = float(parsed_cfgs[0].get("ea_prior", 10.0))
            else:
                parsed_cfgs = json.loads(str(archive["cfgs"]))
                bands_arr = parsed_cfgs[0]["bands"]
                ea_prior_val = float(parsed_cfgs[0].get("ea_prior", 10.0))
            r0_arr = archive["R0"].copy()
            sm_arr = archive["scaler_mean"].copy()
            ss_arr = archive["scaler_scale"].copy()
            coef_arr = archive["coef"].copy()
            inter_arr = archive["intercept"].copy()

        return cls(
            participant_labels=part_names,
            frequency_bands_hz=bands_arr,
            prior_covariances_r0=r0_arr,
            scaler_means=sm_arr,
            scaler_scales=ss_arr,
            lr_coefficients=coef_arr,
            lr_intercepts=inter_arr,
            specialist_weight=specialist_weight,
            kappa=kappa,
            gap_delta=gap_delta,
            prior_n0=prior_n0,
            ea_prior_count=ea_prior_val,
        )

    def save_organ_artifacts(self, target_directory: Path) -> list[Path]:
        resolved_dir = ParameterTypeValidator.require_path(target_directory, "target_directory")
        resolved_dir.mkdir(parents=True, exist_ok=True)
        out_path = resolved_dir / self.ARTIFACT_FILENAME
        cfg_entry = {
            "bands": [[int(b[0]) if b[0].is_integer() else b[0], int(b[1]) if b[1].is_integer() else b[1]] for b in self._bands],
            "ea": "online",
            "train_ea": "offline",
            "ea_prior": self._ea_prior,
            "ea_mean": "euclid",
            "ea_halflife": None,
            "ea_cov": "lw",
            "ea_scale": "none",
            "ea_freeze": 0,
            "use_ts": True,
            "ts_rescale": 0.0,
            "ts_d0": None,
            "lr_C": 0.1,
            "ts_refit": 0.0,
            "use_cnn": False,
            "self_train_at": [],
        }
        np.savez(
            out_path,
            names=np.array(self._participant_labels),
            bands=np.asarray(self._bands, dtype=np.float64),
            cfgs=np.array(json.dumps([cfg_entry for _ in self._participant_labels])),
            R0=self._r0.astype(np.float64),
            scaler_mean=self._scaler_mean.astype(np.float64),
            scaler_scale=self._scaler_scale.astype(np.float64),
            coef=self._coef.astype(np.float64),
            intercept=self._intercept.astype(np.float64),
        )
        return [out_path]

    def _update_block_from_specialist_log_probs(self, spec_log_probs: np.ndarray) -> np.ndarray:
        self._last_specialist_log_probs = spec_log_probs
        class_count = spec_log_probs.shape[-1]
        rates = (self._hits + self._prior_n0 / float(class_count)) / (float(self._seen_windows) + self._prior_n0)
        sorted_rates = np.sort(rates)
        gap_factor = float(np.clip((sorted_rates[-1] - sorted_rates[-2]) / self._gap_delta, 0.0, 1.0))
        self._last_gate_weight = self._specialist_weight * gap_factor
        router_pi = CovarianceMatrixOperator.stable_softmax(self._kappa * rates)
        self._last_block_log_probs = (router_pi[:, None] * spec_log_probs).sum(axis=0)
        return CovarianceMatrixOperator.stable_softmax(self._last_block_log_probs)

    def step_causal_window(
        self,
        cleaned_window: np.ndarray,
        is_artifact_flagged: bool,
        known_covariances: dict | None = None,
    ) -> np.ndarray:
        win_f64 = np.asarray(cleaned_window, dtype=np.float64)
        demeaned_win = win_f64 - win_f64.mean(axis=-1, keepdims=True)
        covs_7b = self.window_band_covariances(
            demeaned_win,
            self._bands,
            self._sos_bank,
            known_covariances=known_covariances,
            include_broadband=True,
            sampling_rate_hz=self._sampling_rate_hz,
        )
        covs_6b = covs_7b[:-1]
        if not is_artifact_flagged:
            self._cov_sum_6b = self._cov_sum_6b + covs_6b
            self._clean_ref_count += 1

        ref_6b = (self._ea_prior * self._r0_6b + self._cov_sum_6b[None]) / (
            self._ea_prior + float(self._clean_ref_count)
        )
        whiten_6b = CovarianceMatrixOperator.inverse_matrix_square_root(ref_6b)
        aligned_6b = whiten_6b @ covs_6b[None] @ whiten_6b
        tang_raw = CovarianceMatrixOperator.project_to_tangent_vector(aligned_6b).reshape(self._spec_count, -1)
        tang_std = (tang_raw - self._scaler_mean) / self._scaler_scale
        spec_probs = CovarianceMatrixOperator.stable_softmax(
            np.einsum("qd,qkd->qk", tang_std, self._coef) + self._intercept,
            axis=-1,
        )
        raw_ls = np.log(spec_probs + 1e-12)
        norm_ls = raw_ls - np.log(np.exp(raw_ls).sum(axis=-1, keepdims=True))
        return self._update_block_from_specialist_log_probs(norm_ls)

    def step_reusing_peer_scores(self, peer_organ: ParticipantSpecialistRiemannianOrgan) -> np.ndarray:
        peer_ls = peer_organ.last_specialist_log_probs
        if peer_ls is None:
            raise RuntimeError("peer_organ must run step_causal_window before step_reusing_peer_scores")
        return self._update_block_from_specialist_log_probs(peer_ls)

    def compute_routed_term_and_update(
        self,
        base_log_prob: np.ndarray,
        balance_prior: float = 10.0,
    ) -> np.ndarray:
        if self._last_block_log_probs is None or self._last_specialist_log_probs is None:
            return np.zeros_like(base_log_prob, dtype=np.float64)
        base_lp = np.asarray(base_log_prob, dtype=np.float64)
        additive_change = self._last_gate_weight * (self._last_block_log_probs - base_lp)
        base_probs = CovarianceMatrixOperator.stable_softmax(base_lp)
        self._p0_running_sum = base_probs.copy() if self._p0_running_sum is None else (self._p0_running_sum + base_probs)
        balanced_base_argmax = int(
            self.balance_probability_step(
                base_probs,
                self._p0_running_sum,
                self._seen_windows + 1,
                prior_pseudo_count=balance_prior,
            ).argmax()
        )
        self._hits += (self._last_specialist_log_probs.argmax(axis=-1) == balanced_base_argmax).astype(np.float64)
        self._seen_windows += 1
        return additive_change
