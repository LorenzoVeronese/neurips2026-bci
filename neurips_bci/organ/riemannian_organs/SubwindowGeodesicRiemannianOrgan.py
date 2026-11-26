"""Causal sub-window geodesic transport and temporal-velocity Riemannian sLDA organ (TSR4)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any
import numpy as np
from scipy.optimize import minimize

from neurips_bci.container.stream_containers.WindowedCorpusStreamContainer import WindowedCorpusStreamContainer
from neurips_bci.operator.signal_operators.SpectralSignalOperator import SpectralSignalOperator
from neurips_bci.organ.riemannian_organs.RiemannianOrgan import RiemannianOrgan
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class SubwindowGeodesicRiemannianOrgan(RiemannianOrgan):
    """3-view x 4-subwindow x 6-band Hierarchical EB-EA geodesic trajectory organ matching TSR3TaskOnlineHead."""

    ARTIFACT_FILENAME = "tsr3task.npz"
    VIEWS: tuple[str, ...] = ("WG_FP_20", "SM_SUB_20", "CAR_PCA22")
    TSR_BANDS_6_HZ: tuple[tuple[float, float], ...] = (
        (4.0, 8.0),
        (8.0, 13.0),
        (13.0, 22.0),
        (22.0, 35.0),
        (35.0, 54.0),
        (4.0, 54.0),
    )
    TSR_SEGMENTS: tuple[tuple[str, int, int], ...] = (
        ("s0_early", 0, 240),
        ("s1_mid", 120, 360),
        ("s2_late", 240, 480),
        ("s3_full", 0, 480),
    )
    TSR_SHRINKAGE = 0.10
    BUNDLED_ARMS: tuple[str, ...] = (
        "TSR4_SubWin_Vel_sLDA",
        "TSR2_SubWin_Vel_LR",
        "TSR3_SubWin_Vel_CellLR",
        "TSR5_SubWin_Ens",
    )
    WG_FP_PREFERRED_CHANNELS: tuple[str, ...] = (
        "AF3", "F5", "F3", "FC5", "C5", "TP7", "CP5", "P5",
        "AF4", "F6", "F4", "FC6", "TP8", "CP6", "P6", "Fz",
        "Fp1", "Fp2", "F7", "F8", "T7", "T8", "P7", "P8",
    )
    SM_SUB_PREFERRED_CHANNELS: tuple[str, ...] = (
        "Fz", "FCz", "Cz", "CPz", "Pz", "P3", "P4", "PO7",
        "FC3", "FC4", "C3", "C1", "C2", "C4", "CP3", "CP4",
        "FC1", "FC2", "CP1", "CP2", "F1", "F2", "Oz", "PO9",
    )
    HOMOLOGOUS_PAIRS_CANDIDATES: tuple[tuple[str, str], ...] = (
        ("AF3", "AF4"), ("F5", "F6"), ("F3", "F4"), ("F1", "F2"),
        ("FC5", "FC6"), ("FC3", "FC4"), ("FC1", "FC2"), ("C5", "C6"),
        ("C3", "C4"), ("C1", "C2"), ("TP7", "TP8"), ("CP5", "CP6"),
        ("CP3", "CP4"), ("CP1", "CP2"), ("P5", "P6"), ("P3", "P4"),
        ("P1", "P2"), ("PO7", "PO8"), ("Fp1", "Fp2"), ("AF7", "AF8"),
        ("F7", "F8"), ("T7", "T8"), ("P7", "P8"), ("PO9", "PO10"), ("O1", "O2"),
    )
    STENCIL_6X4 = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [1.0, -2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    _SYM_INDEX_CACHE: dict[int, tuple[tuple[np.ndarray, np.ndarray], np.ndarray]] = {}

    def __init__(
        self,
        weight_tensors: dict[str, np.ndarray],
        active_arm: str = "TSR4_SubWin_Vel_sLDA",
        pc_tau: float = 1.5,
        tsr_mode: str = "pc_le",
        max_history: int = 512,
        sampling_rate_hz: float = 120.0,
    ) -> None:
        super().__init__(organ_identifier="subwindow_geodesic_tsr4")
        self._weights = {k_str: np.asarray(arr).copy() for k_str, arr in weight_tensors.items()}
        self._active_arm = ParameterTypeValidator.require_non_empty_string(active_arm, "active_arm")
        self._pc_tau = ParameterTypeValidator.require_float(pc_tau, "pc_tau", 0.0)
        self._tsr_mode = str(tsr_mode)
        self._max_hist = ParameterTypeValidator.require_integer(max_history, "max_history", 8)
        self._sampling_rate_hz = ParameterTypeValidator.require_float(sampling_rate_hz, "sampling_rate_hz", 1.0)

        self._n_chans = int(self._weights["P_WG_FP_20"].shape[1])
        self._n_times = int(self._weights.get("n_times", np.array([480]))[0])
        self._spec_recenter_n0 = float(self._weights.get("spec_recenter_n0", np.array([12.0]))[0])
        self._m_sos_6 = SpectralSignalOperator.precomputed_sos_operators(
            self.TSR_BANDS_6_HZ,
            sample_count=self._n_times,
            sampling_rate_hz=self._sampling_rate_hz,
        )
        self._lr_left = np.asarray(self._weights["lr_left_idx"], dtype=np.int64)
        self._lr_right = np.asarray(self._weights["lr_right_idx"], dtype=np.int64)
        self._spec_mu0 = np.asarray(self._weights["spec_mu0"], dtype=np.float64)

        self._views_spec: dict[str, dict[str, Any]] = {}
        for view_key in self.VIEWS:
            proj_mat = np.asarray(self._weights[f"P_{view_key}"], dtype=np.float64)
            dim_d = int(proj_mat.shape[0])
            iu_idx, scale_vec = self._sym_index(dim_d)
            self._views_spec[view_key] = {
                "P": proj_mat,
                "d": dim_d,
                "d_tri": dim_d * (dim_d + 1) // 2,
                "iu": iu_idx,
                "sc": scale_vec,
                "eye_d": np.eye(dim_d, dtype=np.float64),
                "R0h": np.asarray(self._weights[f"eb_{view_key}_R0h"], dtype=np.float64),
                "R0ih": np.asarray(self._weights[f"eb_{view_key}_R0ih"], dtype=np.float64),
                "U_s": np.asarray(self._weights[f"eb_{view_key}_U_s"], dtype=np.float64),
                "lam_i": np.asarray(self._weights[f"eb_{view_key}_lam_i"], dtype=np.float64),
                "lam_w": np.asarray(self._weights[f"eb_{view_key}_lam_w"], dtype=np.float64),
                "lam_r": np.asarray(self._weights[f"eb_{view_key}_lam_r"], dtype=np.float64),
                "vi_p": np.asarray(self._weights[f"eb_{view_key}_vi_p"], dtype=np.float64),
                "vw_p": np.asarray(self._weights[f"eb_{view_key}_vw_p"], dtype=np.float64),
                "vr_p": np.asarray(self._weights[f"eb_{view_key}_vr_p"], dtype=np.float64),
            }

        self._arm_heads: dict[str, dict[str, Any]] = {}
        for cand_arm in self.BUNDLED_ARMS:
            pref = f"{cand_arm}__"
            if f"{pref}b_total" in self._weights:
                self._arm_heads[cand_arm] = {
                    "b_total": np.asarray(self._weights[f"{pref}b_total"], dtype=np.float64).copy(),
                    "W_spec_seg": np.asarray(self._weights[f"{pref}W_spec_seg"], dtype=np.float64),
                    "W_asym_seg": np.asarray(self._weights[f"{pref}W_asym_seg"], dtype=np.float64),
                    "views": {
                        view_key: {
                            "W_tan": np.asarray(self._weights[f"{pref}W_{view_key}_tan"], dtype=np.float64),
                            "W_tan_diag": np.asarray(self._weights[f"{pref}W_{view_key}_tan_diag"], dtype=np.float64),
                            "W_D": np.asarray(self._weights[f"{pref}W_{view_key}_D"], dtype=np.float64),
                            "W_eig": np.asarray(self._weights[f"{pref}W_{view_key}_eig"], dtype=np.float64),
                        }
                        for view_key in self.VIEWS
                    },
                }

        default_head = {
            "b_total": np.asarray(self._weights["b_total"], dtype=np.float64).copy(),
            "W_spec_seg": np.asarray(self._weights["W_spec_seg"], dtype=np.float64),
            "W_asym_seg": np.asarray(self._weights["W_asym_seg"], dtype=np.float64),
            "views": {
                view_key: {
                    "W_tan": np.asarray(self._weights[f"W_{view_key}_tan"], dtype=np.float64),
                    "W_tan_diag": np.asarray(self._weights[f"W_{view_key}_tan_diag"], dtype=np.float64),
                    "W_D": np.asarray(self._weights[f"W_{view_key}_D"], dtype=np.float64),
                    "W_eig": np.asarray(self._weights[f"W_{view_key}_eig"], dtype=np.float64),
                }
                for view_key in self.VIEWS
            },
        }
        if not self._arm_heads or self._active_arm not in self._arm_heads:
            self._arm_heads[self._active_arm] = default_head

        self._pseudo_chrono_ref: Any = None
        self._y_sum: dict[str, np.ndarray] = {}
        self._spec_sum: np.ndarray = np.zeros_like(self._spec_mu0)
        self._n_clean = 0
        self._u_hist: list[np.ndarray] = []
        self._lp3_hist: list[np.ndarray] = []
        self._y_hist: dict[str, list[np.ndarray]] = {}
        self._last_probs_by_arm: dict[str, np.ndarray] = {}
        self.reset_organ_state()

    @property
    def last_probabilities_by_arm(self) -> dict[str, np.ndarray]:
        return dict(self._last_probs_by_arm)

    def attach_pseudochrono_adapter(self, pseudo_chrono_adapter: Any) -> None:
        self._pseudo_chrono_ref = pseudo_chrono_adapter

    def reset_organ_state(self) -> None:
        self._y_sum = {
            view_key: np.zeros((6, self._views_spec[view_key]["d_tri"]), dtype=np.float64)
            for view_key in self.VIEWS
        }
        self._spec_sum = np.zeros_like(self._spec_mu0, dtype=np.float64)
        self._n_clean = 0
        self._u_hist = []
        self._lp3_hist = []
        self._y_hist = {view_key: [] for view_key in self.VIEWS}
        uniform_3 = np.full(3, 1.0 / 3.0, dtype=np.float64)
        self._last_probs_by_arm = {arm_k: uniform_3.copy() for arm_k in self._arm_heads}

    @classmethod
    def _sym_index(cls, dim_n: int) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        if dim_n not in cls._SYM_INDEX_CACHE:
            iu_pair = np.triu_indices(dim_n)
            scale_vec = np.where(iu_pair[0] == iu_pair[1], 1.0, np.sqrt(2.0)).astype(np.float64)
            cls._SYM_INDEX_CACHE[dim_n] = (iu_pair, scale_vec)
        return cls._SYM_INDEX_CACHE[dim_n]

    @classmethod
    def _mat2vec(cls, sym_mat: np.ndarray) -> np.ndarray:
        iu_pair, scale_vec = cls._sym_index(sym_mat.shape[-1])
        return sym_mat[..., iu_pair[0], iu_pair[1]] * scale_vec

    @classmethod
    def _vec2mat(cls, vec_in: np.ndarray, dim_n: int) -> np.ndarray:
        iu_pair, scale_vec = cls._sym_index(dim_n)
        mat_out = np.zeros(vec_in.shape[:-1] + (dim_n, dim_n), dtype=np.float64)
        unscaled = vec_in / scale_vec
        mat_out[..., iu_pair[0], iu_pair[1]] = unscaled
        mat_out[..., iu_pair[1], iu_pair[0]] = unscaled
        return mat_out

    @staticmethod
    def _eig_fn_sym(mat_in: np.ndarray, eig_scalar_fn) -> np.ndarray:
        sym_mat = 0.5 * (mat_in + np.swapaxes(mat_in, -1, -2))
        eig_vals, eig_vecs = np.linalg.eigh(sym_mat)
        return (eig_vecs * eig_scalar_fn(eig_vals)[..., None, :]) @ np.swapaxes(eig_vecs, -1, -2)

    @classmethod
    def _expm_sym(cls, mat_in: np.ndarray) -> np.ndarray:
        return cls._eig_fn_sym(mat_in, np.exp)

    @classmethod
    def _logm_sym(cls, mat_in: np.ndarray) -> np.ndarray:
        return cls._eig_fn_sym(mat_in, lambda w_arr: np.log(np.maximum(w_arr, 1e-12)))

    @classmethod
    def _inv_sqrtm_sym(cls, mat_in: np.ndarray) -> np.ndarray:
        return cls._eig_fn_sym(mat_in, lambda w_arr: 1.0 / np.sqrt(np.maximum(w_arr, 1e-12)))

    @staticmethod
    def _softmax_1d(logits_in: np.ndarray) -> np.ndarray:
        z64 = np.asarray(logits_in, dtype=np.float64)
        z64 = z64 - np.max(z64, axis=-1, keepdims=True)
        exps = np.exp(z64)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    @classmethod
    def load_from_npz(
        cls,
        tsr_npz_path: Path,
        pc_tau: float = 1.5,
        active_arm: str = "TSR4_SubWin_Vel_sLDA",
        tsr_mode: str = "pc_le",
    ) -> SubwindowGeodesicRiemannianOrgan:
        resolved_path = ParameterTypeValidator.require_path(tsr_npz_path, "tsr_npz_path", must_exist=True)
        with np.load(resolved_path, allow_pickle=False) as archive:
            loaded_weights = {key_str: archive[key_str].copy() for key_str in archive.files}
        return cls(
            weight_tensors=loaded_weights,
            active_arm=active_arm,
            pc_tau=pc_tau,
            tsr_mode=tsr_mode,
        )

    def save_organ_artifacts(self, target_directory: Path) -> list[Path]:
        resolved_dir = ParameterTypeValidator.require_path(target_directory, "target_directory")
        resolved_dir.mkdir(parents=True, exist_ok=True)
        out_path = resolved_dir / self.ARTIFACT_FILENAME
        np.savez_compressed(out_path, **self._weights)
        return [out_path]

    def _eb_reference_matrix(
        self,
        view_dict: dict[str, Any],
        y_bar: np.ndarray,
        n_override: float | None = None,
        grun_is_inv_n: bool = False,
    ) -> np.ndarray:
        n_eff = float(max(1.0, self._n_clean if n_override is None else n_override))
        inv_n = 1.0 / n_eff
        g_run = inv_n if grun_is_inv_n else (1.0 / max(1.0, n_eff / 31.0))
        noise_par = view_dict["lam_w"] * inv_n + g_run * view_dict["lam_r"] + 1e-12
        noise_perp = view_dict["vw_p"] * inv_n + g_run * view_dict["vr_p"] + 1e-12
        g_par0 = view_dict["lam_i"] / (view_dict["lam_i"] + noise_par)
        g_perp0 = view_dict["vi_p"] / (view_dict["vi_p"] + noise_perp)

        u_sub = view_dict["U_s"]
        y_par = np.einsum("bd,bdr->br", y_bar, u_sub)
        y_par_proj = np.einsum("br,bdr->bd", y_par, u_sub)
        y_par_scaled = np.einsum("br,bdr->bd", y_par * g_par0, u_sub)
        tangent_l0 = y_par_scaled + g_perp0[:, None] * (y_bar - y_par_proj)
        mat_l0 = self._vec2mat(tangent_l0, view_dict["d"])
        return view_dict["R0h"] @ self._expm_sym(mat_l0) @ view_dict["R0h"]

    @staticmethod
    def _compute_pc_weights(pc_ref: Any, cur_u: np.ndarray, u_hist: list[np.ndarray], tau: float) -> np.ndarray:
        if hasattr(pc_ref, "weights_against"):
            return pc_ref.weights_against(cur_u, u_hist, float(tau))
        if not u_hist or tau <= 0.0:
            return np.zeros(0, dtype=np.float64)
        u_mat = np.stack(u_hist, axis=0)
        sq_dist = np.sum((u_mat - cur_u[None, :]) ** 2, axis=1)
        return np.exp(-0.5 * sq_dist / (float(u_mat.shape[1]) * (tau * tau)))

    def step_causal_window(
        self,
        cleaned_window: np.ndarray,
        is_artifact_flagged: bool,
        pseudo_chrono_adapter: Any = None,
    ) -> np.ndarray:
        win_np = np.asarray(cleaned_window, dtype=np.float64)
        if not np.isfinite(win_np).all():
            win_np = np.nan_to_num(win_np, nan=0.0, posinf=500.0, neginf=-500.0)
        centered_win = win_np - np.mean(win_np, axis=-1, keepdims=True)

        filtered_6b = np.matmul(centered_win[None], self._m_sos_6)
        filtered_car = filtered_6b - np.mean(filtered_6b, axis=1, keepdims=True)

        seg_scales = float(self._n_times) / 480.0
        seg_bounds = [
            (int(round(s0 * seg_scales)), max(int(round(s0 * seg_scales)) + 2, int(round(s1 * seg_scales))))
            for (_sk, s0, s1) in self.TSR_SEGMENTS
        ]

        lp_car = np.empty((4, 6, self._n_chans), dtype=np.float64)
        for seg_idx, (s0, s1) in enumerate(seg_bounds):
            seg_c = filtered_car[..., s0:s1]
            lp_car[seg_idx] = np.log(np.maximum(np.mean(seg_c ** 2, axis=-1), 1e-6))

        s_views: dict[str, np.ndarray] = {}
        for view_key in self.VIEWS:
            proj_mat = self._views_spec[view_key]["P"]
            dim_d = self._views_spec[view_key]["d"]
            yb_v = np.matmul(proj_mat[None], filtered_6b)
            cov_seg = np.empty((4, 6, dim_d, dim_d), dtype=np.float64)
            for seg_idx, (s0, s1) in enumerate(seg_bounds):
                seg_v = yb_v[..., s0:s1]
                seg_v = seg_v - np.mean(seg_v, axis=-1, keepdims=True)
                cov_seg[seg_idx] = np.matmul(seg_v, np.swapaxes(seg_v, -1, -2)) / float(s1 - s0)
            s_views[view_key] = cov_seg

        pc_ref = self._pseudo_chrono_ref if pseudo_chrono_adapter is None else pseudo_chrono_adapter
        cur_u = None if pc_ref is None else getattr(pc_ref, "cur_u", getattr(pc_ref, "current_embedding", None))
        max_hist = self._max_hist if pc_ref is None else int(getattr(pc_ref, "max_hist", self._max_hist))
        use_pcle = (self._tsr_mode == "pc_le") and (pc_ref is not None) and (cur_u is not None) and (self._pc_tau > 0.0)

        if not is_artifact_flagged:
            self._n_clean += 1
            self._spec_sum += lp_car[3]
            if use_pcle and cur_u is not None:
                self._u_hist.append(np.asarray(cur_u, dtype=np.float64))
                self._lp3_hist.append(lp_car[3].copy())
                if len(self._u_hist) > max_hist:
                    self._u_hist.pop(0)
                    self._lp3_hist.pop(0)
        n_div = float(max(1, self._n_clean))

        if use_pcle and cur_u is not None:
            w_tsr = self._compute_pc_weights(pc_ref, np.asarray(cur_u, dtype=np.float64), self._u_hist, self._pc_tau)
            w_sum_tsr = float(w_tsr.sum())
            w_sq_tsr = float((w_tsr ** 2).sum())
            n_eff_tsr = (w_sum_tsr ** 2) / max(w_sq_tsr, 1e-12) if w_sum_tsr > 1e-12 else 1.0
            sp_ker = (
                np.tensordot(w_tsr, np.stack(self._lp3_hist, axis=0), axes=(0, 0))
                if self._lp3_hist
                else np.zeros_like(self._spec_mu0)
            )
            spec_ref = (self._spec_recenter_n0 * self._spec_mu0 + sp_ker) / (self._spec_recenter_n0 + w_sum_tsr)
        else:
            w_tsr = None
            w_sum_tsr = 0.0
            n_eff_tsr = n_div
            spec_ref = (self._spec_recenter_n0 * self._spec_mu0 + self._spec_sum) / (
                self._spec_recenter_n0 + float(self._n_clean)
            )

        lp_cent = (lp_car - spec_ref[None]).reshape(4, -1)
        asym_seg = (lp_car[..., self._lr_left] - lp_car[..., self._lr_right]).reshape(4, -1)

        logits_by_arm: dict[str, np.ndarray] = {}
        for arm_k, hd_dict in self._arm_heads.items():
            lg_vec = hd_dict["b_total"].copy()
            lg_vec += np.einsum("sd,sdc->c", lp_cent, hd_dict["W_spec_seg"])
            lg_vec += np.einsum("sd,sdc->c", asym_seg, hd_dict["W_asym_seg"])
            logits_by_arm[arm_k] = lg_vec

        for view_key in self.VIEWS:
            v_dict = self._views_spec[view_key]
            dim_d = v_dict["d"]
            iu0, iu1 = v_dict["iu"]
            sc_vec = v_dict["sc"]

            s_mat = s_views[view_key]
            s_mat = 0.5 * (s_mat + np.swapaxes(s_mat, -1, -2))
            tr_val = np.maximum(np.trace(s_mat, axis1=-2, axis2=-1) / float(dim_d), 1e-12)[..., None, None]
            c_shrunk = (1.0 - self.TSR_SHRINKAGE) * s_mat + self.TSR_SHRINKAGE * tr_val * v_dict["eye_d"]

            if not is_artifact_flagged:
                m_cur = np.matmul(np.matmul(v_dict["R0ih"], c_shrunk[3]), v_dict["R0ih"])
                y_cur = self._mat2vec(self._logm_sym(m_cur))
                self._y_sum[view_key] += y_cur
                if use_pcle:
                    self._y_hist[view_key].append(y_cur)
                    if len(self._y_hist[view_key]) > max_hist:
                        self._y_hist[view_key].pop(0)

            if use_pcle and w_tsr is not None:
                if self._y_hist[view_key]:
                    y_bar = np.tensordot(w_tsr, np.stack(self._y_hist[view_key], axis=0), axes=(0, 0)) / max(
                        w_sum_tsr, 1.0
                    )
                else:
                    y_bar = np.zeros((6, v_dict["d_tri"]), dtype=np.float64)
                g_ref = self._eb_reference_matrix(v_dict, y_bar, n_override=max(n_eff_tsr, 1.0), grun_is_inv_n=True)
            else:
                y_bar = self._y_sum[view_key] / n_div
                g_ref = self._eb_reference_matrix(
                    v_dict, y_bar, grun_is_inv_n=(self._tsr_mode == "grun1")
                )
            inv_sq_g = self._inv_sqrtm_sym(g_ref)

            m_sk = np.matmul(np.matmul(inv_sq_g[None], c_shrunk), inv_sq_g[None])
            m_sk = 0.5 * (m_sk + np.swapaxes(m_sk, -1, -2))
            tan_diag = np.log(np.maximum(np.diagonal(m_sk, axis1=-2, axis2=-1), 1e-12)).reshape(4, -1)
            ev_m, vec_m = np.linalg.eigh(m_sk)
            log_ev_m = np.log(np.maximum(ev_m, 1e-12))
            log_m = (vec_m * log_ev_m[..., None, :]) @ np.swapaxes(vec_m, -1, -2)
            tan_6 = (log_m[..., iu0, iu1] * sc_vec).reshape(4, -1)

            c_left = c_shrunk[0:1]
            c_right = c_shrunk[2:3]
            inv_sq_0 = self._inv_sqrtm_sym(c_left)
            m_0r = np.matmul(np.matmul(inv_sq_0, c_right), inv_sq_0)
            m_0r = 0.5 * (m_0r + np.swapaxes(m_0r, -1, -2))
            ev_g, vec_g = np.linalg.eigh(m_0r)
            log_ev_g = np.log(np.maximum(ev_g, 1e-12))
            log_d = (vec_g * log_ev_g[..., None, :]) @ np.swapaxes(vec_g, -1, -2)
            disp_6 = (log_d[..., iu0, iu1] * sc_vec).reshape(1, -1)

            tr_l = np.log(np.maximum(np.sum(np.diagonal(c_left, axis1=-2, axis2=-1), axis=-1), 1e-12))
            tr_r = np.log(np.maximum(np.sum(np.diagonal(c_right, axis1=-2, axis2=-1), axis=-1), 1e-12))
            eig_6 = np.concatenate(
                [log_ev_g, tr_l[..., None], tr_r[..., None], (tr_r - tr_l)[..., None]],
                axis=-1,
            ).reshape(1, -1)

            for arm_k, hd_dict in self._arm_heads.items():
                vw_dict = hd_dict["views"][view_key]
                lg_vec = logits_by_arm[arm_k]
                lg_vec += np.einsum("sd,sdc->c", tan_6, vw_dict["W_tan"])
                lg_vec += np.einsum("sd,sdc->c", tan_diag, vw_dict["W_tan_diag"])
                lg_vec += np.einsum("kd,kdc->c", disp_6, vw_dict["W_D"])
                lg_vec += np.einsum("kd,kdc->c", eig_6, vw_dict["W_eig"])

        self._last_probs_by_arm = {arm_k: self._softmax_1d(lg_vec) for arm_k, lg_vec in logits_by_arm.items()}
        return self._last_probs_by_arm[self._active_arm]

    @classmethod
    def _select_montage_views_and_pairs(
        cls,
        channel_labels: tuple[str, ...],
        s41_full: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        chan_count = len(channel_labels)
        ch_lookup = {str(lbl): idx for idx, lbl in enumerate(channel_labels)}

        def build_selector(preferred: tuple[str, ...], target_dim: int, fallback_start: int) -> np.ndarray:
            chosen = [ch_lookup[ch_str] for ch_str in preferred if ch_str in ch_lookup][:target_dim]
            if len(chosen) < target_dim:
                for cand_i in range(chan_count):
                    shifted = (fallback_start + cand_i) % chan_count
                    if shifted not in chosen:
                        chosen.append(shifted)
                        if len(chosen) == target_dim:
                            break
            proj = np.zeros((len(chosen), chan_count), dtype=np.float64)
            for row_i, col_i in enumerate(chosen):
                proj[row_i, col_i] = 1.0
            return proj

        target_reg_dim = min(16, chan_count)
        p_wg = build_selector(cls.WG_FP_PREFERRED_CHANNELS, target_reg_dim, 0)
        p_sm = build_selector(cls.SM_SUB_PREFERRED_CHANNELS, target_reg_dim, chan_count // 3)

        pca_dim = min(18, max(2, chan_count - 1))
        h_car = np.eye(chan_count, dtype=np.float64) - np.full(
            (chan_count, chan_count), 1.0 / float(chan_count), dtype=np.float64
        )
        c_bar = np.mean(s41_full, axis=(0, 1))
        c_car = 0.5 * (h_car @ c_bar @ h_car.T + (h_car @ c_bar @ h_car.T).T)
        eig_w, eig_v = np.linalg.eigh(c_car)
        top_v = eig_v[:, np.argsort(eig_w)[::-1][:pca_dim]]
        p_pca = (top_v.T @ h_car).astype(np.float64)

        left_list, right_list = [], []
        for left_ch, right_ch in cls.HOMOLOGOUS_PAIRS_CANDIDATES:
            if left_ch in ch_lookup and right_ch in ch_lookup and len(left_list) < 18:
                left_list.append(ch_lookup[left_ch])
                right_list.append(ch_lookup[right_ch])
        if not left_list:
            half_c = max(1, min(18, chan_count // 2))
            left_list = list(range(half_c))
            right_list = list(range(chan_count - half_c, chan_count))

        views_p = {"WG_FP_20": p_wg, "SM_SUB_20": p_sm, "CAR_PCA22": p_pca}
        return views_p, np.asarray(left_list, dtype=np.int64), np.asarray(right_list, dtype=np.int64)

    @classmethod
    def _fit_eb_prior_6b(
        cls,
        c_tr_full: np.ndarray,
        sessions: np.ndarray,
        interfaces: np.ndarray,
        runs: np.ndarray,
        r_max: int = 12,
    ) -> dict[str, np.ndarray]:
        n_tr, band_cnt, dim_d, _ = c_tr_full.shape
        uniq_sess = list(dict.fromkeys(sessions.tolist()))
        sess_means = np.stack([c_tr_full[sessions == sk].mean(axis=0) for sk in uniq_sess], axis=0)
        r0_mat = sess_means.mean(axis=0)
        r0_half = np.empty_like(r0_mat)
        r0_inv_half = np.empty_like(r0_mat)
        for b_idx in range(band_cnt):
            ev_b, vec_b = np.linalg.eigh(0.5 * (r0_mat[b_idx] + r0_mat[b_idx].T))
            ev_b = np.maximum(ev_b, 1e-12)
            r0_half[b_idx] = (vec_b * np.sqrt(ev_b)[None, :]) @ vec_b.T
            r0_inv_half[b_idx] = (vec_b * (1.0 / np.sqrt(ev_b))[None, :]) @ vec_b.T

        d_tri = dim_d * (dim_d + 1) // 2
        y_tr = np.empty((n_tr, band_cnt, d_tri), dtype=np.float64)
        for b_idx in range(band_cnt):
            m_b = r0_inv_half[b_idx][None] @ c_tr_full[:, b_idx] @ r0_inv_half[b_idx][None]
            y_tr[:, b_idx] = cls._mat2vec(cls._logm_sym(m_b))

        r_sub = min(r_max, max(1, len(uniq_sess) - 1), d_tri)
        u_s = np.zeros((band_cnt, d_tri, r_sub), dtype=np.float64)
        lam_i = np.zeros((band_cnt, r_sub), dtype=np.float64)
        lam_w = np.zeros((band_cnt, r_sub), dtype=np.float64)
        lam_r = np.zeros((band_cnt, r_sub), dtype=np.float64)
        vi_p = np.zeros(band_cnt, dtype=np.float64)
        vw_p = np.zeros(band_cnt, dtype=np.float64)
        vr_p = np.zeros(band_cnt, dtype=np.float64)

        for b_idx in range(band_cnt):
            yb = y_tr[:, b_idx]
            s_bars = np.stack([yb[sessions == sk].mean(axis=0) for sk in uniq_sess], axis=0)
            s_cent = s_bars - s_bars.mean(axis=0, keepdims=True)
            _, _, vt_mat = np.linalg.svd(s_cent, full_matrices=False)
            u_b = vt_mat[:r_sub].T
            if u_b.shape[1] < r_sub:
                pad_cols = np.eye(d_tri, r_sub - u_b.shape[1], dtype=np.float64)
                u_b, _ = np.linalg.qr(np.concatenate([u_b, pad_cols], axis=1))
            u_s[b_idx] = u_b
            s_proj = s_cent @ u_b
            ddof_s = 1 if len(s_bars) > 1 else 0
            lam_i[b_idx] = np.maximum(np.var(s_proj, axis=0, ddof=ddof_s), 1e-6)
            s_perp = s_cent - s_proj @ u_b.T
            vi_p[b_idx] = max(float(np.mean(np.sum(s_perp ** 2, axis=1)) / max(1, d_tri - r_sub)), 1e-6)

            w_res, r_bars = [], []
            for sk in uniq_sess:
                m_s = sessions == sk
                ys = yb[m_s]
                sb = ys.mean(axis=0, keepdims=True)
                w_res.append(ys - sb)
                for iv in np.unique(interfaces[m_s]):
                    for rv in np.unique(runs[m_s & (interfaces == iv)]):
                        m_sr = m_s & (interfaces == iv) & (runs == rv)
                        if int(m_sr.sum()) >= 4:
                            r_bars.append(yb[m_sr].mean(axis=0) - sb[0])
            w_all = np.concatenate(w_res, axis=0)
            w_proj = w_all @ u_b
            lam_w[b_idx] = np.maximum(np.var(w_proj, axis=0, ddof=1 if len(w_all) > 1 else 0), 1e-6)
            w_perp = w_all - w_proj @ u_b.T
            vw_p[b_idx] = max(float(np.mean(np.sum(w_perp ** 2, axis=1)) / max(1, d_tri - r_sub)), 1e-6)

            if len(r_bars) >= 2:
                r_all = np.stack(r_bars, axis=0)
                r_proj = r_all @ u_b
                lam_r[b_idx] = np.maximum(np.var(r_proj, axis=0, ddof=1), 1e-6)
                r_perp = r_all - r_proj @ u_b.T
                vr_p[b_idx] = max(float(np.mean(np.sum(r_perp ** 2, axis=1)) / max(1, d_tri - r_sub)), 1e-6)
            else:
                lam_r[b_idx] = 0.25 * lam_w[b_idx]
                vr_p[b_idx] = 0.25 * vw_p[b_idx]

        return {
            "R0h": r0_half,
            "R0ih": r0_inv_half,
            "U_s": u_s,
            "lam_i": lam_i,
            "lam_w": lam_w,
            "lam_r": lam_r,
            "vi_p": vi_p,
            "vw_p": vw_p,
            "vr_p": vr_p,
        }

    @classmethod
    def _extract_session_raw_blocks(
        cls,
        s41_sess: np.ndarray,
        lp_car_sess: np.ndarray,
        views_p: dict[str, np.ndarray],
        eb_priors: dict[str, dict[str, np.ndarray]],
        spec_mu0: np.ndarray,
        lr_left: np.ndarray,
        lr_right: np.ndarray,
        spec_recenter_n0: float = 12.0,
    ) -> dict[str, np.ndarray]:
        win_cnt = len(s41_sess)
        lp_cent_out = np.empty((win_cnt, 4, 6 * s41_sess.shape[-1]), dtype=np.float64)
        asym_out = np.empty((win_cnt, 4, 6 * len(lr_left)), dtype=np.float64)
        spec_sum = np.zeros_like(spec_mu0, dtype=np.float64)
        for i_win in range(win_cnt):
            spec_sum += lp_car_sess[i_win, 3].astype(np.float64)
            spec_ref = (spec_recenter_n0 * spec_mu0 + spec_sum) / (spec_recenter_n0 + float(i_win + 1))
            lpi = lp_car_sess[i_win].astype(np.float64)
            lp_cent_out[i_win] = (lpi - spec_ref[None]).reshape(4, -1)
            asym_out[i_win] = (lpi[..., lr_left] - lpi[..., lr_right]).reshape(4, -1)

        blocks_out: dict[str, np.ndarray] = {"spec_seg": lp_cent_out, "asym_seg": asym_out}
        for view_key, proj_mat in views_p.items():
            eb_dict = eb_priors[view_key]
            dim_d = proj_mat.shape[0]
            d_tri = dim_d * (dim_d + 1) // 2
            (iu0, iu1), sc_vec = cls._sym_index(dim_d)
            eye_d = np.eye(dim_d, dtype=np.float64)

            s_mat = np.matmul(np.matmul(proj_mat, s41_sess.astype(np.float64)), proj_mat.T)
            s_mat = 0.5 * (s_mat + np.swapaxes(s_mat, -1, -2))
            tr_val = np.maximum(np.trace(s_mat, axis1=-2, axis2=-1) / float(dim_d), 1e-12)[..., None, None]
            c_all = (1.0 - cls.TSR_SHRINKAGE) * s_mat + cls.TSR_SHRINKAGE * tr_val * eye_d

            c_left = c_all[:, 0:1]
            c_right = c_all[:, 2:3]
            inv_sq_0 = cls._inv_sqrtm_sym(c_left)
            m_0r = np.matmul(np.matmul(inv_sq_0, c_right), inv_sq_0)
            m_0r = 0.5 * (m_0r + np.swapaxes(m_0r, -1, -2))
            ev_g, vec_g = np.linalg.eigh(m_0r)
            log_ev_g = np.log(np.maximum(ev_g, 1e-12))
            log_d = (vec_g * log_ev_g[..., None, :]) @ np.swapaxes(vec_g, -1, -2)
            disp_6 = (log_d[..., iu0, iu1] * sc_vec).reshape(win_cnt, 1, -1)

            tr_l = np.log(np.maximum(np.sum(np.diagonal(c_left, axis1=-2, axis2=-1), axis=-1), 1e-12))
            tr_r = np.log(np.maximum(np.sum(np.diagonal(c_right, axis1=-2, axis2=-1), axis=-1), 1e-12))
            eig_6 = np.concatenate(
                [log_ev_g, tr_l[..., None], tr_r[..., None], (tr_r - tr_l)[..., None]],
                axis=-1,
            ).reshape(win_cnt, 1, -1)

            m_full_0 = np.matmul(np.matmul(eb_dict["R0ih"][None], c_all[:, 3]), eb_dict["R0ih"][None])
            y_full_0 = cls._mat2vec(cls._logm_sym(m_full_0))

            g_refs = np.empty((win_cnt, 6, dim_d, dim_d), dtype=np.float64)
            y_sum = np.zeros((6, d_tri), dtype=np.float64)
            u_sub = eb_dict["U_s"]
            for i_win in range(win_cnt):
                y_sum += y_full_0[i_win]
                n_f = float(i_win + 1)
                yb = y_sum / n_f
                inv_n = 1.0 / n_f
                g_run = 1.0 / max(1.0, n_f / 31.0)
                noise_par = eb_dict["lam_w"] * inv_n + g_run * eb_dict["lam_r"] + 1e-12
                noise_perp = eb_dict["vw_p"] * inv_n + g_run * eb_dict["vr_p"] + 1e-12
                g_par0 = eb_dict["lam_i"] / (eb_dict["lam_i"] + noise_par)
                g_perp0 = eb_dict["vi_p"] / (eb_dict["vi_p"] + noise_perp)
                y_par = np.einsum("bd,bdr->br", yb, u_sub)
                y_par_proj = np.einsum("br,bdr->bd", y_par, u_sub)
                y_par_scaled = np.einsum("br,bdr->bd", y_par * g_par0, u_sub)
                l0_vec = y_par_scaled + g_perp0[:, None] * (yb - y_par_proj)
                g_refs[i_win] = eb_dict["R0h"] @ cls._expm_sym(cls._vec2mat(l0_vec, dim_d)) @ eb_dict["R0h"]

            inv_sq_g = cls._inv_sqrtm_sym(g_refs)
            m_sk = np.matmul(np.matmul(inv_sq_g[:, None], c_all), inv_sq_g[:, None])
            m_sk = 0.5 * (m_sk + np.swapaxes(m_sk, -1, -2))
            tan_diag = np.log(np.maximum(np.diagonal(m_sk, axis1=-2, axis2=-1), 1e-12)).reshape(win_cnt, 4, -1)
            ev_m, vec_m = np.linalg.eigh(m_sk)
            log_ev_m = np.log(np.maximum(ev_m, 1e-12))
            log_m = (vec_m * log_ev_m[..., None, :]) @ np.swapaxes(vec_m, -1, -2)
            tan_6 = (log_m[..., iu0, iu1] * sc_vec).reshape(win_cnt, 4, -1)

            blocks_out[f"{view_key}_tan"] = tan_6
            blocks_out[f"{view_key}_tan_diag"] = tan_diag
            blocks_out[f"{view_key}_D"] = disp_6
            blocks_out[f"{view_key}_eig"] = eig_6
        return blocks_out

    @classmethod
    def _build_linear_terms(cls, raw_blocks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        terms: dict[str, np.ndarray] = {}
        spec_6 = np.einsum("ks,nsd->nkd", cls.STENCIL_6X4, raw_blocks["spec_seg"])
        asym_6 = np.einsum("ks,nsd->nkd", cls.STENCIL_6X4, raw_blocks["asym_seg"])
        row_cnt = len(spec_6)
        terms["spec_stat"] = spec_6[:, :2].reshape(row_cnt, -1)
        terms["spec_vel"] = spec_6[:, 2:].reshape(row_cnt, -1)
        terms["asym_stat"] = asym_6[:, :2].reshape(row_cnt, -1)
        terms["asym_vel"] = asym_6[:, 2:].reshape(row_cnt, -1)
        for view_key in cls.VIEWS:
            tan_6 = np.einsum("ks,nsd->nkd", cls.STENCIL_6X4, raw_blocks[f"{view_key}_tan"])
            td_6 = np.einsum("ks,nsd->nkd", cls.STENCIL_6X4, raw_blocks[f"{view_key}_tan_diag"])
            terms[f"{view_key}_tan_stat"] = tan_6[:, :2].reshape(row_cnt, -1)
            terms[f"{view_key}_tan_vel"] = tan_6[:, 2:].reshape(row_cnt, -1)
            terms[f"{view_key}_tdiag_stat"] = td_6[:, :2].reshape(row_cnt, -1)
            terms[f"{view_key}_tdiag_vel"] = td_6[:, 2:].reshape(row_cnt, -1)
            terms[f"{view_key}_D"] = raw_blocks[f"{view_key}_D"].reshape(row_cnt, -1)
            terms[f"{view_key}_eig"] = raw_blocks[f"{view_key}_eig"].reshape(row_cnt, -1)
        return terms

    @staticmethod
    def _is_velocity_block(term_key: str) -> bool:
        return ("_vel" in term_key) or term_key.endswith("_D") or term_key.endswith("_eig") or ("asym" in term_key)

    @classmethod
    def _block_weight(cls, term_key: str, vel_boost: float = 1.25) -> float:
        if "_tan_stat" in term_key:
            base_w = 1.0
        elif "_tan_vel" in term_key or term_key.endswith("_D"):
            base_w = 0.70
        elif term_key.endswith("_eig"):
            base_w = 0.40
        elif "asym" in term_key:
            base_w = 0.45
        else:
            base_w = 0.35
        return base_w * (vel_boost if cls._is_velocity_block(term_key) else 1.0)

    @staticmethod
    def _solve_dual_newton_3class(
        gram_tr: np.ndarray,
        y_tr: np.ndarray,
        c_reg: float,
        sample_weights: np.ndarray | None = None,
        max_iter: int = 80,
    ) -> tuple[np.ndarray, np.ndarray]:
        row_cnt = len(y_tr)
        one_hot_y = np.eye(3, dtype=np.float64)[y_tr]
        sw = np.ones(row_cnt, dtype=np.float64) if sample_weights is None else np.asarray(sample_weights, dtype=np.float64)
        reg_lam = 1.0 / max(float(c_reg), 1e-6)

        k_sym = 0.5 * (gram_tr + gram_tr.T) + 1e-8 * np.eye(row_cnt, dtype=np.float64)
        ev, vecs = np.linalg.eigh(k_sym)
        keep = ev > (1e-10 * max(float(ev[-1]), 1e-10))
        ev_r = ev[keep]
        vecs_r = vecs[:, keep]
        phi_mat = vecs_r * np.sqrt(ev_r)[None, :]
        phi_inv_sqrt = vecs_r * (1.0 / np.sqrt(ev_r))[None, :]
        r_dim = phi_mat.shape[1]

        def obj_and_grad(flat_params: np.ndarray) -> tuple[float, np.ndarray]:
            th = flat_params.reshape(3, r_dim + 1)
            w_mat = th[:, :r_dim]
            b_vec = th[:, r_dim]
            logits = phi_mat @ w_mat.T + b_vec[None, :]
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_z = np.exp(logits)
            sum_exp = exp_z.sum(axis=1, keepdims=True)
            probs = exp_z / sum_exp
            log_p = logits - np.log(sum_exp)
            obj_val = -float(np.sum(sw[:, None] * one_hot_y * log_p)) + 0.5 * reg_lam * float(np.sum(w_mat * w_mat))
            grad_z = (probs - one_hot_y) * sw[:, None]
            grad_w = grad_z.T @ phi_mat + reg_lam * w_mat
            grad_b = grad_z.sum(axis=0)
            grad_b = grad_b - grad_b.mean()
            return obj_val, np.concatenate([grad_w, grad_b[:, None]], axis=1).reshape(-1)

        opt_out = minimize(
            obj_and_grad,
            np.zeros(3 * (r_dim + 1), dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iter, "gtol": 1e-6},
        )
        th_opt = opt_out.x.reshape(3, r_dim + 1)
        b_opt = th_opt[:, r_dim].copy()
        b_opt -= b_opt.mean()
        alpha_opt = phi_inv_sqrt @ th_opt[:, :r_dim].T
        return alpha_opt, b_opt

    @staticmethod
    def _solve_dual_slda(gram_tr: np.ndarray, y_tr: np.ndarray, shrink: float = 0.18) -> tuple[np.ndarray, np.ndarray]:
        row_cnt = len(y_tr)
        k_sym = 0.5 * (gram_tr + gram_tr.T) + 1e-8 * np.eye(row_cnt, dtype=np.float64)
        ev, vecs = np.linalg.eigh(k_sym)
        ev = np.maximum(ev, 1e-10)
        phi_mat = vecs * np.sqrt(ev)[None, :]
        phi_inv_sqrt = vecs * (1.0 / np.sqrt(ev))[None, :]

        mu_c = np.stack(
            [phi_mat[y_tr == c_idx].mean(axis=0) if np.any(y_tr == c_idx) else phi_mat.mean(axis=0) for c_idx in range(3)],
            axis=0,
        )
        centered = phi_mat - mu_c[y_tr]
        s_within = (centered.T @ centered) / float(row_cnt)
        tr_s = float(np.trace(s_within)) / float(row_cnt)
        s_reg = (1.0 - shrink) * s_within + (shrink * tr_s) * np.eye(row_cnt, dtype=np.float64)
        w_phi = np.linalg.solve(s_reg, mu_c.T).T
        b_vec = -0.5 * np.sum(mu_c * w_phi, axis=1)
        b_vec -= b_vec.mean()
        alpha_vec = phi_inv_sqrt @ w_phi.T
        return alpha_vec, b_vec

    @classmethod
    def _collapse_dual_to_primal(
        cls,
        terms_tr: dict[str, np.ndarray],
        stats_vel: dict[str, tuple[np.ndarray, np.ndarray, float]],
        alpha_vec: np.ndarray,
        b_vec: np.ndarray,
        views_p: dict[str, np.ndarray],
        eb_priors: dict[str, dict[str, np.ndarray]],
        spec_mu0: np.ndarray,
        lr_left: np.ndarray,
        lr_right: np.ndarray,
        n_times: int,
        vel_boost: float,
    ) -> dict[str, np.ndarray]:
        w_terms: dict[str, np.ndarray] = {}
        b_total = np.asarray(b_vec, dtype=np.float64).copy()
        for term_k, x_k in terms_tr.items():
            mu_k, sd_k, rms_k = stats_vel[term_k]
            w_blk = cls._block_weight(term_k, vel_boost=vel_boost)
            z_k = ((x_k - mu_k) / (sd_k * rms_k)) * np.sqrt(w_blk)
            w_z = z_k.T @ alpha_vec
            w_x = w_z * (np.sqrt(w_blk) / (sd_k * rms_k))[:, None]
            b_total -= (mu_k[:, None] * w_x).sum(axis=0)
            w_terms[term_k] = w_x

        chan_count = views_p["WG_FP_20"].shape[1]
        w_sp_6 = np.concatenate(
            [w_terms["spec_stat"].reshape(2, 6 * chan_count, 3), w_terms["spec_vel"].reshape(4, 6 * chan_count, 3)],
            axis=0,
        )
        w_spec_seg = np.einsum("ks,kdc->sdc", cls.STENCIL_6X4, w_sp_6)
        w_as_6 = np.concatenate(
            [
                w_terms["asym_stat"].reshape(2, 6 * len(lr_left), 3),
                w_terms["asym_vel"].reshape(4, 6 * len(lr_left), 3),
            ],
            axis=0,
        )
        w_asym_seg = np.einsum("ks,kdc->sdc", cls.STENCIL_6X4, w_as_6)

        collapsed: dict[str, np.ndarray] = {
            "b_total": b_total,
            "n_times": np.array([int(n_times)], dtype=np.int64),
            "spec_recenter_n0": np.array([12.0], dtype=np.float64),
            "lr_left_idx": lr_left.astype(np.int64),
            "lr_right_idx": lr_right.astype(np.int64),
            "spec_mu0": spec_mu0.astype(np.float64),
            "W_spec_seg": w_spec_seg,
            "W_asym_seg": w_asym_seg,
        }
        for view_key, proj_mat in views_p.items():
            dim_d = proj_mat.shape[0]
            d_tri = dim_d * (dim_d + 1) // 2
            w_t_6 = np.concatenate(
                [
                    w_terms[f"{view_key}_tan_stat"].reshape(2, 6 * d_tri, 3),
                    w_terms[f"{view_key}_tan_vel"].reshape(4, 6 * d_tri, 3),
                ],
                axis=0,
            )
            w_td_6 = np.concatenate(
                [
                    w_terms[f"{view_key}_tdiag_stat"].reshape(2, 6 * dim_d, 3),
                    w_terms[f"{view_key}_tdiag_vel"].reshape(4, 6 * dim_d, 3),
                ],
                axis=0,
            )
            collapsed[f"P_{view_key}"] = proj_mat.astype(np.float64)
            collapsed[f"W_{view_key}_tan"] = np.einsum("ks,kdc->sdc", cls.STENCIL_6X4, w_t_6)
            collapsed[f"W_{view_key}_tan_diag"] = np.einsum("ks,kdc->sdc", cls.STENCIL_6X4, w_td_6)
            collapsed[f"W_{view_key}_D"] = w_terms[f"{view_key}_D"].reshape(1, 6 * d_tri, 3)
            collapsed[f"W_{view_key}_eig"] = w_terms[f"{view_key}_eig"].reshape(1, 6 * (dim_d + 3), 3)
            for eb_k, eb_v in eb_priors[view_key].items():
                collapsed[f"eb_{view_key}_{eb_k}"] = eb_v.astype(np.float64)
        return collapsed

    @classmethod
    def fit_from_corpus(
        cls,
        corpus_container: WindowedCorpusStreamContainer,
        random_seed: int = 0,
        pc_tau: float = 1.5,
    ) -> SubwindowGeodesicRiemannianOrgan:
        _ = ParameterTypeValidator.require_integer(random_seed, "random_seed", 0)
        windows_x = np.asarray(corpus_container.eeg_windows, dtype=np.float64)
        win_cnt, chan_cnt, sample_cnt = windows_x.shape
        y_tr = np.asarray(corpus_container.class_labels, dtype=np.int64)
        sessions = np.asarray(corpus_container.session_ids)
        participants = np.asarray(corpus_container.participant_ids)
        interfaces = np.asarray(corpus_container.interface_contexts)
        runs = np.asarray(corpus_container.run_indices)

        m_sos_6 = SpectralSignalOperator.precomputed_sos_operators(
            cls.TSR_BANDS_6_HZ,
            sample_count=sample_cnt,
            sampling_rate_hz=corpus_container.sampling_rate_hz,
        )
        seg_scales = float(sample_cnt) / 480.0
        seg_bounds = [
            (int(round(s0 * seg_scales)), max(int(round(s0 * seg_scales)) + 2, int(round(s1 * seg_scales))))
            for (_sk, s0, s1) in cls.TSR_SEGMENTS
        ]

        s41 = np.empty((win_cnt, 4, 6, chan_cnt, chan_cnt), dtype=np.float32)
        lp_car = np.empty((win_cnt, 4, 6, chan_cnt), dtype=np.float32)
        for i0 in range(0, win_cnt, 128):
            i1 = min(win_cnt, i0 + 128)
            xb = windows_x[i0:i1] - windows_x[i0:i1].mean(axis=-1, keepdims=True)
            yb = np.matmul(xb[:, None], m_sos_6[None])
            yb_car = yb - np.mean(yb, axis=2, keepdims=True)
            for seg_idx, (s0, s1) in enumerate(seg_bounds):
                seg = yb[..., s0:s1]
                seg = seg - np.mean(seg, axis=-1, keepdims=True)
                s41[i0:i1, seg_idx] = (np.matmul(seg, np.swapaxes(seg, -1, -2)) / float(s1 - s0)).astype(np.float32)
                lp_car[i0:i1, seg_idx] = np.log(np.maximum(np.mean(yb_car[..., s0:s1] ** 2, axis=-1), 1e-6)).astype(
                    np.float32
                )

        views_p, lr_left, lr_right = cls._select_montage_views_and_pairs(
            corpus_container.channel_labels, s41[:, 3]
        )
        eb_priors: dict[str, dict[str, np.ndarray]] = {}
        for view_key, proj_mat in views_p.items():
            dim_d = proj_mat.shape[0]
            eye_d = np.eye(dim_d, dtype=np.float64)
            s_v = np.matmul(np.matmul(proj_mat, s41[:, 3].astype(np.float64)), proj_mat.T)
            s_v = 0.5 * (s_v + np.swapaxes(s_v, -1, -2))
            tr_v = np.maximum(np.trace(s_v, axis1=-2, axis2=-1) / float(dim_d), 1e-12)[..., None, None]
            c_v_full = (1.0 - cls.TSR_SHRINKAGE) * s_v + cls.TSR_SHRINKAGE * tr_v * eye_d
            eb_priors[view_key] = cls._fit_eb_prior_6b(c_v_full, sessions, interfaces, runs, r_max=12)

        uniq_sess = list(dict.fromkeys(sessions.tolist()))
        spec_mu0 = np.stack([lp_car[sessions == sk, 3].mean(axis=0) for sk in uniq_sess], axis=0).mean(axis=0).astype(
            np.float64
        )

        raw_tr_lists: dict[str, list[np.ndarray]] = {}
        idx_order = []
        for sk in uniq_sess:
            idx_s = np.flatnonzero(sessions == sk)
            idx_order.append(idx_s)
            rb = cls._extract_session_raw_blocks(
                s41[idx_s], lp_car[idx_s], views_p, eb_priors, spec_mu0, lr_left, lr_right
            )
            for blk_k, blk_v in rb.items():
                raw_tr_lists.setdefault(blk_k, []).append(blk_v)
        inv_ord = np.argsort(np.concatenate(idx_order, axis=0))
        raw_tr = {blk_k: np.concatenate(v_list, axis=0)[inv_ord] for blk_k, v_list in raw_tr_lists.items()}

        terms_vel = cls._build_linear_terms(raw_tr)
        stats_vel: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        z_norm_vel: dict[str, np.ndarray] = {}
        k_base = np.zeros((win_cnt, win_cnt), dtype=np.float64)
        k_vel = np.zeros((win_cnt, win_cnt), dtype=np.float64)
        for term_k, x_k in terms_vel.items():
            mu_k = x_k.mean(axis=0)
            sd_k = np.maximum(x_k.std(axis=0), 1e-4)
            z0 = (x_k - mu_k) / sd_k
            rms_k = float(np.sqrt(np.maximum(np.mean(np.sum(z0 ** 2, axis=1)), 1e-8)))
            stats_vel[term_k] = (mu_k, sd_k, rms_k)
            zk = z0 / rms_k
            z_norm_vel[term_k] = zk
            gram_k = zk @ zk.T
            if cls._is_velocity_block(term_k):
                k_vel += cls._block_weight(term_k, vel_boost=1.0) * gram_k
            else:
                k_base += cls._block_weight(term_k, vel_boost=1.0) * gram_k

        uniq_subjs = list(dict.fromkeys(participants.tolist()))
        best_c, best_vb = 0.05, 1.35
        if len(uniq_subjs) >= 2:
            best_score = -1.0
            for vb_cand in (1.0, 1.35):
                k_cand = k_base + vb_cand * k_vel
                for c_cand in (0.02, 0.05, 0.15):
                    fold_accs = []
                    for subj_id in uniq_subjs[:5]:
                        m_va = participants == subj_id
                        m_in = ~m_va
                        if not np.any(m_va) or len(np.unique(y_tr[m_in])) < 2:
                            continue
                        idx_in = np.flatnonzero(m_in)
                        idx_va = np.flatnonzero(m_va)
                        al_in, b_in = cls._solve_dual_newton_3class(k_cand[np.ix_(idx_in, idx_in)], y_tr[idx_in], c_cand)
                        p_va = cls._softmax_1d(k_cand[np.ix_(idx_va, idx_in)] @ al_in + b_in[None, :])
                        fold_accs.append(float((p_va.argmax(axis=1) == y_tr[idx_va]).mean()))
                    if fold_accs and float(np.mean(fold_accs)) > best_score:
                        best_score = float(np.mean(fold_accs))
                        best_c, best_vb = c_cand, vb_cand

        k_full = k_base + best_vb * k_vel
        alpha_s2, b_s2 = cls._solve_dual_newton_3class(k_full, y_tr, best_c)
        w_tsr2 = cls._collapse_dual_to_primal(
            terms_vel, stats_vel, alpha_s2, b_s2, views_p, eb_priors, spec_mu0, lr_left, lr_right, sample_cnt, best_vb
        )

        sw_cell = np.ones(win_cnt, dtype=np.float64)
        m_gr = np.array(["Graz" in str(ctx) for ctx in interfaces], dtype=bool)
        if np.any(m_gr) and np.any(~m_gr):
            sw_cell[m_gr] = 0.5 * win_cnt / float(m_gr.sum())
            sw_cell[~m_gr] = 0.5 * win_cnt / float((~m_gr).sum())
        sw_cell /= max(float(sw_cell.mean()), 1e-8)
        alpha_s3, b_s3 = cls._solve_dual_newton_3class(k_full, y_tr, best_c, sample_weights=sw_cell)
        w_tsr3 = cls._collapse_dual_to_primal(
            terms_vel, stats_vel, alpha_s3, b_s3, views_p, eb_priors, spec_mu0, lr_left, lr_right, sample_cnt, best_vb
        )

        alpha_s4, b_s4 = cls._solve_dual_slda(k_full, y_tr, shrink=0.18)
        w_tsr4 = cls._collapse_dual_to_primal(
            terms_vel, stats_vel, alpha_s4, b_s4, views_p, eb_priors, spec_mu0, lr_left, lr_right, sample_cnt, best_vb
        )

        w_tsr5 = copy.deepcopy(w_tsr2)
        for key_str in w_tsr5:
            if key_str.startswith("W_") or key_str == "b_total":
                w_tsr5[key_str] = 0.45 * w_tsr2[key_str] + 0.30 * w_tsr3[key_str] + 0.25 * w_tsr4[key_str]

        bundled_weights = copy.deepcopy(w_tsr4)
        for arm_label, arm_dict in (
            ("TSR2_SubWin_Vel_LR", w_tsr2),
            ("TSR3_SubWin_Vel_CellLR", w_tsr3),
            ("TSR4_SubWin_Vel_sLDA", w_tsr4),
            ("TSR5_SubWin_Ens", w_tsr5),
        ):
            for key_str, arr_val in arm_dict.items():
                if key_str.startswith("W_") or key_str == "b_total":
                    bundled_weights[f"{arm_label}__{key_str}"] = arr_val.copy()

        return cls(
            weight_tensors=bundled_weights,
            active_arm="TSR4_SubWin_Vel_sLDA",
            pc_tau=pc_tau,
            sampling_rate_hz=corpus_container.sampling_rate_hz,
        )
