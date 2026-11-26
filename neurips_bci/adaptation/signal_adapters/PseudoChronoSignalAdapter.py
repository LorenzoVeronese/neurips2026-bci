"""Causal B=1 PseudoChrono run-subspace Euclidean alignment, tangent projection, and BatchNorm tracker."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.signal import sosfiltfilt

from neurips_bci.adaptation.signal_adapters.SignalAdapter import SignalAdapter
from neurips_bci.container.stream_containers.WindowedCorpusStreamContainer import WindowedCorpusStreamContainer
from neurips_bci.operator.matrix_operators.CovarianceMatrixOperator import CovarianceMatrixOperator
from neurips_bci.operator.matrix_operators.SubspaceMatrixOperator import SubspaceMatrixOperator
from neurips_bci.operator.signal_operators.SpectralSignalOperator import SpectralSignalOperator
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class PseudoChronoSignalAdapter(SignalAdapter):
    """Order-invariant causal B=1 manifold aligner using a class-orthogonal non-motor run subspace."""

    _BG7_BANDS_HZ: tuple[tuple[float, float], ...] = (
        (1.0, 4.0),
        (4.0, 8.0),
        (8.0, 13.0),
        (13.0, 20.0),
        (20.0, 30.0),
        (30.0, 45.0),
        (1.0, 45.0),
    )
    _NM_BANDS_HZ: tuple[tuple[float, float], ...] = (
        (0.5, 4.0),
        (32.0, 48.0),
        (0.5, 48.0),
    )

    def __init__(
        self,
        mu_phi: np.ndarray,
        sd_phi: np.ndarray,
        u_run_basis: np.ndarray,
        feat_mode: str = "NonMotor",
        center_u: bool = False,
        ea_tau: float = 1.3,
        bn_tau: float = 1.3,
        ea_n0_prior: float = 6.0,
        ea_n0_glob: float = 2.0,
        bn_n0_prior: float = 12.0,
        bn_n0_glob: float = 3.0,
        max_history_windows: int = 512,
    ) -> None:
        super().__init__(adapter_role="pseudochrono_ea_and_bn_signal_adapter")
        self._mu_phi = np.asarray(
            ParameterTypeValidator.require_numpy_array(mu_phi, "mu_phi", expected_ndim=1),
            dtype=np.float64,
        )
        feat_dim = int(self._mu_phi.shape[0])
        self._sd_phi = np.maximum(
            np.asarray(
                ParameterTypeValidator.require_numpy_array(sd_phi, "sd_phi", expected_shape=(feat_dim,)),
                dtype=np.float64,
            ),
            1e-6,
        )
        self._u_run_basis = np.asarray(
            ParameterTypeValidator.require_numpy_array(u_run_basis, "u_run_basis", expected_ndim=2),
            dtype=np.float64,
        )
        if self._u_run_basis.shape[0] != feat_dim or self._u_run_basis.shape[1] < 1:
            raise ValueError(
                f"u_run_basis shape {self._u_run_basis.shape} incompatible with mu_phi length {feat_dim}"
            )
        self._feat_mode = ParameterTypeValidator.require_choice(
            feat_mode, ("NonMotor", "BG7", "BG7Ext"), "feat_mode"
        )
        self._center_u = ParameterTypeValidator.require_boolean(center_u, "center_u")
        self._subspace_dim = int(self._u_run_basis.shape[1])
        self._ea_tau = ParameterTypeValidator.require_float(ea_tau, "ea_tau", minimum_bound=0.0)
        self._bn_tau = ParameterTypeValidator.require_float(bn_tau, "bn_tau", minimum_bound=0.0)
        self._ea_n0_prior = ParameterTypeValidator.require_float(ea_n0_prior, "ea_n0_prior", minimum_bound=0.0)
        self._ea_n0_glob = ParameterTypeValidator.require_float(ea_n0_glob, "ea_n0_glob", minimum_bound=0.0)
        self._bn_n0_prior = ParameterTypeValidator.require_float(bn_n0_prior, "bn_n0_prior", minimum_bound=0.0)
        self._bn_n0_glob = ParameterTypeValidator.require_float(bn_n0_glob, "bn_n0_glob", minimum_bound=0.0)
        self._max_history_windows = ParameterTypeValidator.require_integer(
            max_history_windows, "max_history_windows", minimum_bound=1
        )

        self._clean_u_history: list[np.ndarray] = []
        self._clean_cov_buffer: np.ndarray | None = None
        self._clean_cov_buffer_len: int = 0
        self._clean_window_count: int = 0
        self._cov_cumulative: np.ndarray | None = None
        self._u_cumulative: np.ndarray = np.zeros(self._subspace_dim, dtype=np.float64)

        self._session_u_all: list[np.ndarray] = []
        self._session_broadband_cov_all: list[np.ndarray] = []
        self._session_bad_all: list[bool] = []
        self._retro_cache: dict[tuple[object, ...], np.ndarray] = {}

        self._current_step_id: int = 0
        self._began_current_step: bool = False
        self._cur_u: np.ndarray | None = None
        self._cur_phi_nm: np.ndarray | None = None
        self._cur_w_bn: np.ndarray | None = None
        self._bn_group_states: dict[str, dict[str, object]] = {}
        self.reset_session_state()

    @property
    def mu_phi(self) -> np.ndarray:
        return self._mu_phi

    @property
    def sd_phi(self) -> np.ndarray:
        return self._sd_phi

    @property
    def u_run_basis(self) -> np.ndarray:
        return self._u_run_basis

    @property
    def U_run(self) -> np.ndarray:
        return self._u_run_basis

    @property
    def subspace_dim(self) -> int:
        return self._subspace_dim

    @property
    def feat_mode(self) -> str:
        return self._feat_mode

    @property
    def center_u(self) -> bool:
        return self._center_u

    @property
    def ea_tau(self) -> float:
        return self._ea_tau

    @property
    def bn_tau(self) -> float:
        return self._bn_tau

    @property
    def cur_u(self) -> np.ndarray | None:
        return None if self._cur_u is None else self._cur_u.copy()

    @property
    def cur_phi_nm(self) -> np.ndarray | None:
        return None if self._cur_phi_nm is None else self._cur_phi_nm.copy()

    def reset_session_state(self) -> None:
        self._clean_u_history = []
        self._clean_cov_buffer = None
        self._clean_cov_buffer_len = 0
        self._clean_window_count = 0
        self._cov_cumulative = None
        self._u_cumulative = np.zeros(self._subspace_dim, dtype=np.float64)
        self._session_u_all = []
        self._session_broadband_cov_all = []
        self._session_bad_all = []
        self._retro_cache = {}
        self._current_step_id = 0
        self._began_current_step = False
        self._cur_u = None
        self._cur_phi_nm = None
        self._cur_w_bn = None
        self._bn_group_states = {}

    @classmethod
    def _extract_feature_vector(cls, raw_window: np.ndarray, feat_mode: str) -> np.ndarray:
        if feat_mode == "NonMotor":
            return SpectralSignalOperator.extract_non_motor_246_features(raw_window)
        centered = np.asarray(raw_window, dtype=np.float64)
        centered = centered - centered.mean(axis=-1, keepdims=True)
        car_win = centered - centered.mean(axis=0, keepdims=True)
        raw_bb_lv = np.log(np.var(centered, axis=-1) + 1e-6)
        if centered.shape[-1] == 480:
            bg7_ops = SpectralSignalOperator.precomputed_sos_operators(cls._BG7_BANDS_HZ, 480, 120.0, 4)
            bg7_lv = np.log(np.var(np.matmul(car_win[None], bg7_ops), axis=-1) + 1e-6).reshape(-1)
            if feat_mode == "BG7":
                phi_vec = np.concatenate([bg7_lv, raw_bb_lv], axis=0)
            else:
                nm_ops = SpectralSignalOperator.precomputed_sos_operators(cls._NM_BANDS_HZ, 480, 120.0, 4)
                r2_lv = np.log(np.var(np.matmul(centered[None], nm_ops[:2]), axis=-1) + 1e-6).reshape(-1)
                c_05_48 = np.log(np.var(car_win @ nm_ops[2], axis=-1) + 1e-6)
                phi_vec = np.concatenate([bg7_lv, raw_bb_lv, r2_lv, c_05_48 - raw_bb_lv], axis=0)
        else:
            sos_bg7 = SpectralSignalOperator.build_butterworth_sos_bank(list(cls._BG7_BANDS_HZ), 120.0, 4)
            bg_list = [np.log(np.var(sosfiltfilt(s_mat, car_win, axis=-1), axis=-1) + 1e-6) for s_mat in sos_bg7]
            bg_list.append(raw_bb_lv)
            if feat_mode == "BG7":
                phi_vec = np.concatenate(bg_list, axis=0)
            else:
                sos_nm = SpectralSignalOperator.build_butterworth_sos_bank(list(cls._NM_BANDS_HZ), 120.0, 4)
                r0 = np.log(np.var(sosfiltfilt(sos_nm[0], centered, axis=-1), axis=-1) + 1e-6)
                r1 = np.log(np.var(sosfiltfilt(sos_nm[1], centered, axis=-1), axis=-1) + 1e-6)
                c2 = np.log(np.var(sosfiltfilt(sos_nm[2], car_win, axis=-1), axis=-1) + 1e-6)
                phi_vec = np.concatenate(bg_list + [r0, r1, c2 - raw_bb_lv], axis=0)
        if not np.isfinite(phi_vec).all():
            phi_vec = np.nan_to_num(phi_vec, nan=0.0, posinf=0.0, neginf=0.0)
        return phi_vec

    def extract_u_coordinate(self, raw_window: np.ndarray) -> np.ndarray:
        if self._feat_mode == "NonMotor":
            phi_vec = SpectralSignalOperator.extract_non_motor_246_features(raw_window)
            self._cur_phi_nm = phi_vec
        else:
            self._cur_phi_nm = None
            phi_vec = self._extract_feature_vector(raw_window, self._feat_mode)
        u_coord = ((phi_vec - self._mu_phi) / self._sd_phi) @ self._u_run_basis
        if not np.isfinite(u_coord).all():
            u_coord = np.zeros(self._subspace_dim, dtype=np.float64)
        return u_coord

    @classmethod
    def fit_from_corpus(
        cls,
        corpus_container: WindowedCorpusStreamContainer,
        subspace_dim: int = 6,
        ea_tau: float = 1.3,
        bn_tau: float = 1.3,
        extra_task_nullspace_dims: int = 6,
        pca_components: int = 40,
        within_shrinkage: float = 0.20,
    ) -> PseudoChronoSignalAdapter:
        """Fit class-orthogonal 246-D NonMotor run/interface/rest subspace via PCA + generalized discriminant."""
        if not isinstance(corpus_container, WindowedCorpusStreamContainer):
            raise TypeError(
                f"corpus_container must be WindowedCorpusStreamContainer, got {type(corpus_container).__name__}"
            )
        ParameterTypeValidator.require_integer(subspace_dim, "subspace_dim", minimum_bound=1)
        ParameterTypeValidator.require_float(ea_tau, "ea_tau", minimum_bound=0.0)
        ParameterTypeValidator.require_float(bn_tau, "bn_tau", minimum_bound=0.0)

        task_windows = corpus_container.eeg_windows
        phi_task = np.stack(
            [SpectralSignalOperator.extract_non_motor_246_features(win_arr) for win_arr in task_windows],
            axis=0,
        )
        mu_phi = phi_task.mean(axis=0).astype(np.float64)
        sd_phi = np.maximum(phi_task.std(axis=0).astype(np.float64), 1e-5)
        z_task = (phi_task - mu_phi) / sd_phi

        labels_tr = corpus_container.class_labels
        sess_tr = corpus_container.session_ids.astype(str)
        subj_tr = corpus_container.participant_ids.astype(str)
        iface_tr = corpus_container.interface_contexts.astype(str)
        runs_tr = corpus_container.run_indices.astype(str)
        group_keys_task = np.array(
            [f"{sk}#{ifc}#{rk}" for sk, ifc, rk in zip(sess_tr, iface_tr, runs_tr)],
            dtype="U128",
        )

        if corpus_container.resting_windows is not None and corpus_container.resting_session_ids is not None and len(corpus_container.resting_windows) > 0:
            phi_rest = np.stack(
                [
                    SpectralSignalOperator.extract_non_motor_246_features(rest_win)
                    for rest_win in corpus_container.resting_windows
                ],
                axis=0,
            )
            z_rest = (phi_rest - mu_phi) / sd_phi
            sess_rest = corpus_container.resting_session_ids.astype(str)
            group_keys_rest = np.array([f"{sk}#Rest#0" for sk in sess_rest], dtype="U128")
            z_all = np.vstack([z_task, z_rest])
            sess_all = np.concatenate([sess_tr, sess_rest])
            group_keys_all = np.concatenate([group_keys_task, group_keys_rest])
        else:
            z_all = z_task
            sess_all = sess_tr
            group_keys_all = group_keys_task

        z_task_wrun = np.empty_like(z_task)
        for rk_str in sorted(set(group_keys_task.tolist())):
            rk_mask = group_keys_task == rk_str
            z_task_wrun[rk_mask] = z_task[rk_mask] - z_task[rk_mask].mean(axis=0, keepdims=True)

        unique_classes = sorted(set(labels_tr.tolist()))
        cls_means = np.stack([z_task_wrun[labels_tr == cls_id].mean(axis=0) for cls_id in unique_classes], axis=0)
        cls_means = cls_means - cls_means.mean(axis=0, keepdims=True)
        _, _, vt_cls = np.linalg.svd(cls_means, full_matrices=False)
        rank_base = min(max(1, len(unique_classes) - 1), vt_cls.shape[0])
        q_base = vt_cls[:rank_base].T

        feat_dim = z_task.shape[1]
        if extra_task_nullspace_dims > 0:
            proj_base = np.eye(feat_dim, dtype=np.float64) - q_base @ q_base.T
            subj_cls_diffs: list[np.ndarray] = []
            for pj_str in sorted(set(subj_tr.tolist())):
                mp_mask = subj_tr == pj_str
                if int(mp_mask.sum()) < 12:
                    continue
                for cls_id in unique_classes:
                    mpc_mask = mp_mask & (labels_tr == cls_id)
                    if int(mpc_mask.sum()) >= 3:
                        subj_cls_diffs.append(z_task_wrun[mpc_mask].mean(axis=0) @ proj_base)
            if subj_cls_diffs:
                _, _, vt_extra = np.linalg.svd(np.stack(subj_cls_diffs, axis=0), full_matrices=False)
                keep_extra = min(extra_task_nullspace_dims, vt_extra.shape[0])
                q_orth, _ = np.linalg.qr(np.concatenate([q_base, vt_extra[:keep_extra].T], axis=1))
            else:
                q_orth = q_base
        else:
            q_orth = q_base

        proj_orth = np.eye(feat_dim, dtype=np.float64) - q_orth @ q_orth.T

        z_sess_all = np.empty_like(z_all)
        z_wgroup_all = np.empty_like(z_all)
        for sk_str in sorted(set(sess_all.tolist())):
            sk_mask = sess_all == sk_str
            z_sess_all[sk_mask] = z_all[sk_mask] - z_all[sk_mask].mean(axis=0, keepdims=True)
        for gk_str in sorted(set(group_keys_all.tolist())):
            gk_mask = group_keys_all == gk_str
            z_wgroup_all[gk_mask] = z_all[gk_mask] - z_all[gk_mask].mean(axis=0, keepdims=True)

        z_sess_orth = z_sess_all @ proj_orth
        z_wgroup_orth = z_wgroup_all @ proj_orth

        _, _, vt_pca = np.linalg.svd(z_sess_orth, full_matrices=False)
        eff_pca = max(subspace_dim, min(pca_components, vt_pca.shape[0], feat_dim - q_orth.shape[1]))
        eff_pca = min(eff_pca, vt_pca.shape[0])
        p_pca = vt_pca[:eff_pca].T
        y_sess = z_sess_orth @ p_pca
        y_wgroup = z_wgroup_orth @ p_pca

        within_scatter = (y_wgroup.T @ y_wgroup) / float(max(1, len(y_wgroup)))
        group_means = [
            y_sess[group_keys_all == gk_str].mean(axis=0)
            for gk_str in sorted(set(group_keys_all.tolist()))
            if int(np.sum(group_keys_all == gk_str)) >= 2
        ]
        if len(group_means) >= 2:
            stacked_means = np.stack(group_means, axis=0)
            centered_means = stacked_means - stacked_means.mean(axis=0, keepdims=True)
            between_scatter = (centered_means.T @ centered_means) / float(len(centered_means))
        else:
            between_scatter = (y_sess.T @ y_sess) / float(max(1, len(y_sess)))

        disc_basis = SubspaceMatrixOperator.solve_generalized_discriminant(
            between_scatter=between_scatter,
            within_scatter=within_scatter,
            subspace_dim=subspace_dim,
            within_shrinkage=within_shrinkage,
        )
        u_run = proj_orth @ p_pca @ disc_basis
        col_norms = np.linalg.norm(u_run, axis=0, keepdims=True)
        u_run = u_run / np.maximum(col_norms, 1e-12)

        return cls(
            mu_phi=mu_phi,
            sd_phi=sd_phi,
            u_run_basis=u_run,
            feat_mode="NonMotor",
            center_u=False,
            ea_tau=ea_tau,
            bn_tau=bn_tau,
        )

    def save_npz(self, out_path: str | Path) -> Path:
        resolved_out = ParameterTypeValidator.require_path(out_path, "out_path")
        resolved_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            resolved_out,
            mu_phi=self._mu_phi.astype(np.float64),
            sd_phi=self._sd_phi.astype(np.float64),
            U_run=self._u_run_basis.astype(np.float64),
            feat_mode=np.array(self._feat_mode),
            center_u=np.array(int(self._center_u), dtype=np.int64),
        )
        return resolved_out

    @classmethod
    def load_npz(
        cls,
        npz_path: str | Path,
        ea_tau: float = 1.3,
        bn_tau: float = 1.3,
    ) -> PseudoChronoSignalAdapter:
        resolved_src = ParameterTypeValidator.require_path(npz_path, "npz_path", must_exist=True)
        with np.load(resolved_src, allow_pickle=False) as npz_archive:
            mu_phi = npz_archive["mu_phi"].astype(np.float64)
            sd_phi = npz_archive["sd_phi"].astype(np.float64)
            u_run = npz_archive["U_run"].astype(np.float64)
            feat_mode = str(npz_archive["feat_mode"].item()) if "feat_mode" in npz_archive else "NonMotor"
            center_flag = bool(int(npz_archive["center_u"].item())) if "center_u" in npz_archive else False
        return cls(
            mu_phi=mu_phi,
            sd_phi=sd_phi,
            u_run_basis=u_run,
            feat_mode=feat_mode,
            center_u=center_flag,
            ea_tau=ea_tau,
            bn_tau=bn_tau,
        )

    def weights_against_coordinates(
        self,
        u_coord: np.ndarray,
        candidate_coords: list[np.ndarray] | np.ndarray,
        tau_bw: float,
    ) -> np.ndarray:
        if tau_bw <= 0.0 or len(candidate_coords) == 0:
            return np.zeros(0, dtype=np.float64)
        coord_vec = np.asarray(u_coord, dtype=np.float64)
        coord_mat = (
            np.stack(candidate_coords, axis=0)
            if isinstance(candidate_coords, list)
            else np.asarray(candidate_coords, dtype=np.float64)
        )
        sq_dist = np.sum((coord_mat - coord_vec[None, :]) ** 2, axis=1)
        inv_2dt2 = 0.5 / (float(self._subspace_dim) * (float(tau_bw) ** 2))
        return np.exp(-inv_2dt2 * sq_dist)

    def weights_for_tau(self, u_coord: np.ndarray, tau_bw: float) -> np.ndarray:
        return self.weights_against_coordinates(u_coord, self._clean_u_history, tau_bw)

    def begin_window(self, raw_window: np.ndarray) -> np.ndarray:
        ParameterTypeValidator.require_numpy_array(raw_window, "raw_window", expected_ndim=2)
        self._current_step_id += 1
        self._began_current_step = True
        u_coord = self.extract_u_coordinate(raw_window)
        self._cur_u = u_coord
        return u_coord

    def step_window_ea_and_tangent(
        self,
        raw_window: np.ndarray,
        dc_centered_window: np.ndarray,
        is_bad_window: bool,
        is_frozen: bool,
        reference_r0: np.ndarray,
        bands_hz: list[list[float]],
        scale_mode: str = "none",
        broadband_filtered_window: np.ndarray | None = None,
        broadband_estimator: str = "lw",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute causal PseudoChrono-EA band covariances, whitened broadband window, and 5166-D tangent vector."""
        ParameterTypeValidator.require_numpy_array(raw_window, "raw_window", expected_ndim=2)
        ParameterTypeValidator.require_numpy_array(dc_centered_window, "dc_centered_window", expected_ndim=2)
        ParameterTypeValidator.require_numpy_array(reference_r0, "reference_r0", expected_ndim=3)
        ParameterTypeValidator.require_choice(scale_mode, ("none", "zstd"), "scale_mode")
        ParameterTypeValidator.require_choice(broadband_estimator, ("lw", "sample"), "broadband_estimator")

        if self._began_current_step and self._cur_u is not None:
            self._began_current_step = False
            u_coord = self._cur_u
        else:
            u_coord = self.begin_window(raw_window)
            self._began_current_step = False

        broadband_batch = (
            None
            if broadband_filtered_window is None
            else ParameterTypeValidator.require_numpy_array(
                broadband_filtered_window, "broadband_filtered_window", expected_ndim=2
            )[None, :, :]
        )
        band_covariances = SpectralSignalOperator.compute_band_covariances(
            window_batch=dc_centered_window[None, :, :],
            frequency_bands_hz=bands_hz,
            sampling_rate_hz=120.0,
            broadband_filtered_batch=broadband_batch,
            broadband_estimator=broadband_estimator,
        )[0]

        if len(self._session_u_all) < 512:
            self._session_u_all.append(u_coord)
            self._session_broadband_cov_all.append(band_covariances[-1].copy())
            self._session_bad_all.append(bool(is_bad_window or is_frozen))

        if (not is_bad_window) and (not is_frozen):
            self._u_cumulative = self._u_cumulative + u_coord
            self._cov_cumulative = (
                band_covariances.copy()
                if self._cov_cumulative is None
                else (self._cov_cumulative + band_covariances)
            )
            self._clean_window_count += 1
            self._clean_u_history.append(u_coord)
            if self._clean_cov_buffer is None:
                self._clean_cov_buffer = np.empty(
                    (self._max_history_windows, *band_covariances.shape), dtype=np.float64
                )
                self._clean_cov_buffer_len = 0
            if self._clean_cov_buffer_len < self._max_history_windows:
                self._clean_cov_buffer[self._clean_cov_buffer_len] = band_covariances
                self._clean_cov_buffer_len += 1
            else:
                self._clean_cov_buffer[:-1] = self._clean_cov_buffer[1:]
                self._clean_cov_buffer[-1] = band_covariances
            if len(self._clean_u_history) > self._max_history_windows:
                self._clean_u_history.pop(0)

        if self._bn_tau > 0.0:
            self._cur_w_bn = self.weights_for_tau(u_coord, self._bn_tau).astype(np.float32)
        else:
            self._cur_w_bn = None

        r0_f64 = np.asarray(reference_r0, dtype=np.float64)
        if self._ea_tau > 0.0:
            w_ea = self.weights_for_tau(u_coord, self._ea_tau)
            w_sum = float(np.sum(w_ea))
            if w_sum > 0.0 and self._clean_cov_buffer is not None and self._clean_cov_buffer_len > 0:
                cov_kernel = np.tensordot(
                    w_ea,
                    self._clean_cov_buffer[: self._clean_cov_buffer_len],
                    axes=(0, 0),
                )
            else:
                cov_kernel = np.zeros_like(r0_f64)
            cov_global_avg = (
                (self._cov_cumulative / float(self._clean_window_count))
                if (self._clean_window_count > 0 and self._cov_cumulative is not None)
                else r0_f64
            )
            ref_cov_t = (
                self._ea_n0_prior * r0_f64 + self._ea_n0_glob * cov_global_avg + cov_kernel
            ) / (self._ea_n0_prior + self._ea_n0_glob + w_sum)
        else:
            cov_sum = np.zeros_like(r0_f64) if self._cov_cumulative is None else self._cov_cumulative
            ref_cov_t = (self._ea_n0_prior * r0_f64 + cov_sum) / (
                self._ea_n0_prior + float(self._clean_window_count)
            )

        whitener_matrices = CovarianceMatrixOperator.compute_cnn_whitener(ref_cov_t, scale_mode=scale_mode)
        broadband_source = dc_centered_window if broadband_filtered_window is None else broadband_filtered_window
        broadband_whitened_window = whitener_matrices[-1] @ np.asarray(broadband_source, dtype=np.float64)
        aligned_band_covs = whitener_matrices[:-1] @ band_covariances[:-1] @ whitener_matrices[:-1]
        tangent_vector = CovarianceMatrixOperator.project_to_tangent_vector(aligned_band_covs).reshape(-1)
        return band_covariances, broadband_whitened_window, tangent_vector

    def retro_whiten_session_windows(
        self,
        raw_session_windows: list[np.ndarray],
        bad_flags: list[bool],
        broadband_r0: np.ndarray,
        scale_mode: str = "none",
    ) -> np.ndarray:
        """Re-whiten all accumulated prefix windows (0..N-1) at a self-training checkpoint using prefix kernel."""
        if len(raw_session_windows) == 0:
            raise ValueError("raw_session_windows must contain at least one window")
        if len(raw_session_windows) != len(bad_flags):
            raise ValueError("raw_session_windows and bad_flags must have identical length")
        ParameterTypeValidator.require_numpy_array(broadband_r0, "broadband_r0", expected_ndim=2)
        ParameterTypeValidator.require_choice(scale_mode, ("none", "zstd"), "scale_mode")

        raw_stack = np.stack([np.asarray(win_arr, dtype=np.float64) for win_arr in raw_session_windows], axis=0)
        centered_stack = raw_stack - raw_stack.mean(axis=-1, keepdims=True)
        u_prefix = np.stack([self.extract_u_coordinate(win_arr) for win_arr in raw_stack], axis=0)
        cov_prefix = CovarianceMatrixOperator.ledoit_wolf_covariance(centered_stack)
        bad_mask = np.asarray(bad_flags, dtype=bool)
        clean_mask = ~bad_mask
        r0_f64 = np.asarray(broadband_r0, dtype=np.float64)
        cov_glob = cov_prefix[clean_mask].mean(axis=0) if np.any(clean_mask) else r0_f64

        if self._ea_tau > 0.0:
            inv_2dt2 = 0.5 / (float(self._subspace_dim) * (self._ea_tau ** 2))
            sq_mat = np.sum((u_prefix[:, None, :] - u_prefix[None, :, :]) ** 2, axis=-1)
            w_mat = np.exp(-inv_2dt2 * sq_mat)
        else:
            w_mat = np.ones((len(raw_stack), len(raw_stack)), dtype=np.float64)
        w_mat[:, bad_mask] = 0.0
        w_sum = w_mat.sum(axis=1)[:, None, None]
        cov_ker = np.einsum("nm,mij->nij", w_mat, cov_prefix)
        retro_refs = (
            self._ea_n0_prior * r0_f64[None, :, :] + self._ea_n0_glob * cov_glob[None, :, :] + cov_ker
        ) / (self._ea_n0_prior + self._ea_n0_glob + w_sum)

        whiteners = CovarianceMatrixOperator.compute_cnn_whitener(
            retro_refs[:, None, :, :], scale_mode=scale_mode
        )[:, 0, :, :]
        return np.einsum("nij,njt->nit", whiteners, centered_stack)

    def compute_weighted_bn_moments(
        self,
        current_mean: np.ndarray,
        current_var: np.ndarray,
        prior_mean: np.ndarray,
        prior_var: np.ndarray,
        group_key: str,
        is_bad_or_frozen: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute causal PseudoChrono-weighted BatchNorm moments (mean, var) for one network group."""
        ParameterTypeValidator.require_non_empty_string(group_key, "group_key")
        cur_m = np.asarray(current_mean, dtype=np.float64)
        cur_v = np.asarray(current_var, dtype=np.float64)
        pri_m = np.asarray(prior_mean, dtype=np.float64)
        pri_v = np.asarray(prior_var, dtype=np.float64)

        group_state = self._bn_group_states.get(group_key)
        if group_state is None:
            group_state = {
                "u_list": [],
                "m_list": [],
                "v_list": [],
                "m_cum": np.zeros_like(pri_m, dtype=np.float64),
                "v_cum": np.zeros_like(pri_v, dtype=np.float64),
                "clean_n": 0,
                "last_step": -1,
            }
            self._bn_group_states[group_key] = group_state

        u_list: list[np.ndarray] = group_state["u_list"]  # type: ignore[assignment]
        m_list: list[np.ndarray] = group_state["m_list"]  # type: ignore[assignment]
        v_list: list[np.ndarray] = group_state["v_list"]  # type: ignore[assignment]

        if (
            (not is_bad_or_frozen)
            and int(group_state["last_step"]) != self._current_step_id  # type: ignore[arg-type]
            and self._cur_u is not None
        ):
            group_state["last_step"] = self._current_step_id
            group_state["m_cum"] = np.asarray(group_state["m_cum"], dtype=np.float64) + cur_m
            group_state["v_cum"] = np.asarray(group_state["v_cum"], dtype=np.float64) + cur_v
            group_state["clean_n"] = int(group_state["clean_n"]) + 1  # type: ignore[arg-type]
            u_list.append(self._cur_u.copy())
            m_list.append(cur_m.copy())
            v_list.append(cur_v.copy())
            if len(u_list) > self._max_history_windows:
                u_list.pop(0)
                m_list.pop(0)
                v_list.pop(0)

        clean_n = int(group_state["clean_n"])  # type: ignore[arg-type]
        m_cum = np.asarray(group_state["m_cum"], dtype=np.float64)
        v_cum = np.asarray(group_state["v_cum"], dtype=np.float64)
        n_tot_cum = self._bn_n0_prior + self._bn_n0_glob + float(clean_n)
        mu_cum = ((self._bn_n0_prior + self._bn_n0_glob) * pri_m + m_cum) / n_tot_cum
        var_cum = ((self._bn_n0_prior + self._bn_n0_glob) * pri_v + v_cum) / n_tot_cum

        if self._bn_tau > 0.0 and len(u_list) > 0 and self._cur_u is not None and clean_n > 0:
            w_bn = self.weights_against_coordinates(self._cur_u, u_list, self._bn_tau)
            w_sum = float(np.sum(w_bn))
            m_ker = np.tensordot(w_bn, np.stack(m_list, axis=0), axes=(0, 0))
            v_ker = np.tensordot(w_bn, np.stack(v_list, axis=0), axes=(0, 0))
            m_glob = m_cum / float(clean_n)
            v_glob = v_cum / float(clean_n)
            den_pc = self._bn_n0_prior + self._bn_n0_glob + w_sum
            mu_pc = (self._bn_n0_prior * pri_m + self._bn_n0_glob * m_glob + m_ker) / den_pc
            var_pc = (self._bn_n0_prior * pri_v + self._bn_n0_glob * v_glob + v_ker) / den_pc
            return mu_pc, var_pc

        return mu_cum, var_cum
