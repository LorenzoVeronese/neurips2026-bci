"""ANCHOR-MM multimodal biometric margin-gated and NIW prototype specialist organ."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt
import torch

from neurips_bci.organ.foundation_organs.FoundationOrgan import FoundationOrgan
from neurips_bci.organ.multimodal_organs.MultimodalOrgan import MultimodalOrgan
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class AnchorSpecialistMultimodalOrgan(MultimodalOrgan):
    """Causal `B=1` float64 CPU/GPU `AnchorMMOnlineHead` (`256-D z_sr` + `1536-D z_ee` + `16-D` open-set ID margin)."""

    SPEC_BANDS_7: tuple[tuple[int, int], ...] = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (1, 45))
    RIEM_BANDS_6: tuple[tuple[int, int], ...] = ((4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (1, 45))
    MORTP_BANDS_9: tuple[tuple[int, int], ...] = (
        (1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (4, 12), (12, 30), (4, 40)
    )
    MORTP_FULL_IDX: tuple[int, ...] = (1, 2, 3, 4, 5, 8)
    MORTP_DIFF_IDX: tuple[int, ...] = (6, 7, 8)
    MORTP_SHRINK: float = 0.10
    SLICE_EARLY: slice = slice(0, 192)
    SLICE_LATE: slice = slice(264, 480)

    _SYM_INDEX_NP: dict[int, tuple[tuple[np.ndarray, np.ndarray], np.ndarray]] = {}
    _SYM_INDEX_TORCH: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def __init__(
        self,
        weights_dict: dict[str, np.ndarray],
        execution_device: str | torch.device = "cpu",
        zsr_pc_tau: float = 1.5,
        rec_mode: str = "pc",
        pc_tau: float = 1.5,
        n0_glob: float = 0.0,
        n0_prior: float = 8.0,
        proto_mode: str = "cum",
        bal_mode: str = "raw_u",
        min_router_n: int = 4,
        ramp_n0: float = 6.0,
        max_pc_history: int = 512,
    ) -> None:
        super().__init__(organ_identifier="anchor_specialist_multimodal_organ")
        self._weights_dict = weights_dict
        self._n_chans = int(weights_dict["n_chans"][0]) if "n_chans" in weights_dict else 41
        self._n_times = int(weights_dict["n_times"][0]) if "n_times" in weights_dict else 480
        self._n_lat = int(weights_dict["n_lat"][0]) if "n_lat" in weights_dict else 8
        self._n_rms = int(weights_dict["n_rms"][0]) if "n_rms" in weights_dict else 3

        eye_t = np.eye(self._n_times, dtype=np.float64)
        self._m_sos_7 = np.stack(
            [sosfiltfilt(butter(4, b_hz, btype="bandpass", fs=120.0, output="sos"), eye_t, axis=-1) for b_hz in self.SPEC_BANDS_7],
            axis=0,
        )
        self._m_sos_6 = np.stack(
            [sosfiltfilt(butter(4, b_hz, btype="bandpass", fs=120.0, output="sos"), eye_t, axis=-1) for b_hz in self.RIEM_BANDS_6],
            axis=0,
        )
        self._m_sos_9 = np.stack(
            [sosfiltfilt(butter(4, b_hz, btype="bandpass", fs=120.0, output="sos"), eye_t, axis=-1) for b_hz in self.MORTP_BANDS_9],
            axis=0,
        )

        self.w0_52 = np.asarray(weights_dict["W0_52"], dtype=np.float64)
        self.inv_std0_id = np.asarray(weights_dict["inv_std0_id"], dtype=np.float64)
        self.w0_6b_id = np.asarray(weights_dict["W0_6b_id"], dtype=np.float64)
        self.tg_mu = np.asarray(weights_dict["tg_mu"], dtype=np.float64)
        self.tg_sd = np.asarray(weights_dict["tg_sd"], dtype=np.float64)
        self.p_tg = np.asarray(weights_dict["P_tg"], dtype=np.float64)

        self.has_pop_eegpt = "P_ep" in weights_dict and weights_dict["P_ep"].size > 0
        self.ep_mu = np.asarray(weights_dict["ep_mu"], dtype=np.float64) if self.has_pop_eegpt else np.zeros(0, dtype=np.float64)
        self.ep_sd = np.asarray(weights_dict["ep_sd"], dtype=np.float64) if self.has_pop_eegpt else np.zeros(0, dtype=np.float64)
        self.p_ep = np.asarray(weights_dict["P_ep"], dtype=np.float64) if self.has_pop_eegpt else np.zeros((0, 0), dtype=np.float64)

        self.has_pop_c4 = "P_c4_pop" in weights_dict and weights_dict["P_c4_pop"].size > 0
        self.c4_pop_mu = np.asarray(weights_dict["c4_pop_mu"], dtype=np.float64) if self.has_pop_c4 else np.zeros(0, dtype=np.float64)
        self.c4_pop_sd = np.asarray(weights_dict["c4_pop_sd"], dtype=np.float64) if self.has_pop_c4 else np.zeros(0, dtype=np.float64)
        self.p_c4_pop = np.asarray(weights_dict["P_c4_pop"], dtype=np.float64) if self.has_pop_c4 else np.zeros((0, 0), dtype=np.float64)

        self.id_mu_glob = np.asarray(weights_dict["id_mu_glob"], dtype=np.float64)
        self.id_sd_glob = np.asarray(weights_dict["id_sd_glob"], dtype=np.float64)
        self.id_m_lda = np.asarray(weights_dict["id_M_lda"], dtype=np.float64)
        self.emb_p_mat = np.asarray(weights_dict["emb_cen_p"], dtype=np.float64)
        self.cov_w_inv = np.asarray(weights_dict["cov_w_inv"], dtype=np.float64)
        self.tau_mrg = float(np.asarray(weights_dict["tau_mrg"]).reshape(-1)[0])
        self.id_dim = int(self.id_m_lda.shape[1])

        self.p_car40 = np.asarray(weights_dict["P_car40"], dtype=np.float64)
        self.r0_m1_f = np.asarray(weights_dict["R0_m1_f"], dtype=np.float64)
        self.r0_m1_d = np.asarray(weights_dict["R0_m1_d"], dtype=np.float64)
        self.r0_car_f = np.asarray(weights_dict["R0_car_f"], dtype=np.float64)
        self.sr_mu = np.asarray(weights_dict["sr_mu"], dtype=np.float64)
        self.sr_sd = np.asarray(weights_dict["sr_sd"], dtype=np.float64)
        self.p_sr = np.asarray(weights_dict["P_sr"], dtype=np.float64)

        self._zsr_pc_tau = float(zsr_pc_tau)
        self._rec_mode = str(rec_mode)
        self._pc_tau = float(pc_tau)
        self._n0_glob = float(n0_glob)
        self._n0_prior = float(n0_prior)
        self._proto_mode = str(proto_mode)
        self._bal_mode = str(bal_mode)
        self._min_router_n = int(min_router_n)
        self._ramp_n0 = float(ramp_n0)
        self._max_pc_history = int(max_pc_history)

        self._branches: dict[str, dict[str, Any]] = {}
        for branch_key in ("sr", "ee", "full"):
            if f"{branch_key}__W_pop" not in weights_dict:
                continue
            self._branches[branch_key] = self._build_branch_spec(branch_key, weights_dict)

        self.has_ee = "ee" in self._branches
        self.has_full = "full" in self._branches
        self.scale_ee = float(np.sqrt(float(self._branches["ee"]["D"]))) if self.has_ee else 1.0
        self.scale_sr = float(np.sqrt(float(self._branches["sr"]["D"]))) if "sr" in self._branches else 1.0

        if "ee_slice_mode" in weights_dict:
            raw_sl = weights_dict["ee_slice_mode"]
            self.ee_slice_mode = str(raw_sl.item() if raw_sl.ndim > 0 else raw_sl)
        elif self.has_ee and int(self._branches["ee"]["D"]) == 1536:
            self.ee_slice_mode = "lat_rms_c4"
        else:
            self.ee_slice_mode = "lat_rms"
        if self.has_ee and int(self._branches["ee"]["D"]) == 1536 and self.ee_slice_mode == "lat_rms":
            self.ee_slice_mode = "lat_rms_c4"

        self._gpu_consts: dict[str, Any] | None = None
        self._active_pc_u: np.ndarray | None = None
        self.enable_cuda(execution_device)

    @staticmethod
    def _build_branch_spec(branch_key: str, weights_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        mu_z = np.asarray(weights_dict[f"{branch_key}__mu_z"], dtype=np.float64)
        sd_z = np.asarray(weights_dict[f"{branch_key}__sd_z"], dtype=np.float64)
        w_pop = np.asarray(weights_dict[f"{branch_key}__W_pop"], dtype=np.float64)
        b_pop = np.asarray(weights_dict[f"{branch_key}__b_pop"], dtype=np.float64)
        w_stack = np.asarray(weights_dict[f"{branch_key}__W_p"], dtype=np.float64)
        b_stack = np.asarray(weights_dict[f"{branch_key}__b_p"], dtype=np.float64)
        mu_pop_c = np.asarray(weights_dict[f"{branch_key}__mu_pop_c"], dtype=np.float64)
        mu_p_c_stack = np.asarray(weights_dict[f"{branch_key}__mu_p_c"], dtype=np.float64)
        prec_mat = np.asarray(weights_dict[f"{branch_key}__Prec"], dtype=np.float64)
        gamma_lda = float(np.asarray(weights_dict[f"{branch_key}__gamma_lda"]).reshape(-1)[0])
        feat_d = int(mu_z.shape[0])
        spec_k = int(w_stack.shape[0])
        return {
            "branch_key": branch_key,
            "mu_z": mu_z,
            "sd_z": sd_z,
            "D": feat_d,
            "W_pop": w_pop,
            "b_pop": b_pop,
            "W_stack": w_stack,
            "b_stack": b_stack,
            "K": spec_k,
            "mu_pop_c": mu_pop_c,
            "mu_p_c_stack": mu_p_c_stack,
            "Prec": prec_mat,
            "Prec_f32": np.ascontiguousarray(prec_mat, dtype=np.float32),
            "P_mu_pop_c": mu_pop_c @ prec_mat,
            "P_mu_p_c": mu_p_c_stack @ prec_mat,
            "gamma_lda": gamma_lda,
            "nu0": 24.0,
            "alpha_proto": 0.40,
            "conf_min": 0.45,
        }

    def configure_r32_adaptation(
        self,
        zsr_pc_tau: float = 1.5,
        rec_mode: str = "pc",
        pc_tau: float = 1.5,
        n0_glob: float = 0.0,
        n0_prior: float = 8.0,
        proto_mode: str = "cum",
        bal_mode: str = "raw_u",
        min_router_n: int = 4,
        ramp_n0: float = 6.0,
        max_pc_history: int = 512,
    ) -> None:
        self._zsr_pc_tau = float(zsr_pc_tau)
        self._rec_mode = str(rec_mode)
        self._pc_tau = float(pc_tau)
        self._n0_glob = float(n0_glob)
        self._n0_prior = float(n0_prior)
        self._proto_mode = str(proto_mode)
        self._bal_mode = str(bal_mode)
        self._min_router_n = int(min_router_n)
        self._ramp_n0 = float(ramp_n0)
        self._max_pc_history = int(max_pc_history)
        self.reset_organ_state()

    def set_pseudo_chrono_vector(self, current_u_vector: np.ndarray | None, max_history: int | None = None) -> None:
        self._active_pc_u = None if current_u_vector is None else np.asarray(current_u_vector, dtype=np.float64)
        if max_history is not None:
            self._max_pc_history = int(max_history)

    def enable_cuda(self, execution_device: str | torch.device) -> None:
        try:
            dev = torch.device(execution_device)
            want_cuda = dev.type == "cuda" and torch.cuda.is_available()
        except Exception:
            want_cuda = False
            dev = torch.device("cpu")
        if not want_cuda:
            self._gpu_consts = None
            self.reset_organ_state()
            return

        def _f64_dev(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(np.ascontiguousarray(np.asarray(arr, dtype=np.float64)), dtype=torch.float64, device=dev)

        gpu_map: dict[str, Any] = {
            "dev": dev,
            "_m_sos_7": _f64_dev(self._m_sos_7),
            "_m_sos_6": _f64_dev(self._m_sos_6),
            "_m_sos_9": _f64_dev(self._m_sos_9),
            "w0_6b_id": _f64_dev(self.w0_6b_id),
            "tg_mu": _f64_dev(self.tg_mu),
            "tg_sd": _f64_dev(self.tg_sd),
            "p_tg": _f64_dev(self.p_tg),
            "id_mu_glob": _f64_dev(self.id_mu_glob),
            "id_sd_glob": _f64_dev(self.id_sd_glob),
            "id_m_lda": _f64_dev(self.id_m_lda),
            "p_car40": _f64_dev(self.p_car40),
            "r0_m1_f": _f64_dev(self.r0_m1_f),
            "r0_m1_d": _f64_dev(self.r0_m1_d),
            "r0_car_f": _f64_dev(self.r0_car_f),
            "sr_mu": _f64_dev(self.sr_mu),
            "sr_sd": _f64_dev(self.sr_sd),
            "p_sr": _f64_dev(self.p_sr),
        }
        if self.has_pop_eegpt:
            gpu_map["ep_mu"] = _f64_dev(self.ep_mu)
            gpu_map["ep_sd"] = _f64_dev(self.ep_sd)
            gpu_map["p_ep"] = _f64_dev(self.p_ep)
        if self.has_pop_c4:
            gpu_map["c4_pop_mu"] = _f64_dev(self.c4_pop_mu)
            gpu_map["c4_pop_sd"] = _f64_dev(self.c4_pop_sd)
            gpu_map["p_c4_pop"] = _f64_dev(self.p_c4_pop)
        self._gpu_consts = gpu_map
        self.reset_organ_state()

    @staticmethod
    def _make_balancer_state(bal_mode: str) -> dict[str, Any]:
        if bal_mode.startswith("balstd"):
            pr_val = float(bal_mode.split("_")[1]) if "_" in bal_mode else 10.0
            return {"kind": "std", "prior": pr_val, "sum": np.zeros(3, dtype=np.float64), "n": 0}
        q_drift = float(bal_mode.split("_")[1]) / 1000.0 if (bal_mode.startswith("kalman_") and bal_mode != "kalman_008") else 0.008
        return {"kind": "kalman", "q_drift": q_drift, "u": np.full(3, 1.0 / 3.0, dtype=np.float64), "P": 1.0 / 5.0}

    @staticmethod
    def _step_balancer_state(bal_st: dict[str, Any], u_vec: np.ndarray) -> np.ndarray:
        ua = np.asarray(u_vec, dtype=np.float64)
        if bal_st["kind"] == "std":
            bal_st["n"] = int(bal_st["n"]) + 1
            bal_st["sum"] = bal_st["sum"] + ua
            pr = float(bal_st["prior"])
            p_bar = np.clip((pr / 3.0 + bal_st["sum"]) / (pr + float(bal_st["n"])), 1e-6, 1.0)
            q_out = ua / p_bar
            return q_out / float(np.sum(q_out))
        p_pred = float(bal_st["P"]) + float(bal_st["q_drift"])
        gain_k = p_pred / (p_pred + 1.0)
        bal_st["u"] = bal_st["u"] + gain_k * (ua - bal_st["u"])
        bal_st["P"] = (1.0 - gain_k) * p_pred
        q_out = ua / np.clip(bal_st["u"], 1e-6, 1.0)
        return q_out / float(np.sum(q_out))

    def reset_organ_state(self) -> None:
        self.n_ok_sr = 0
        if self._gpu_consts is not None:
            dev = self._gpu_consts["dev"]
            self._sum_m1_f: Any = torch.zeros(self.r0_m1_f.shape, dtype=torch.float64, device=dev)
            self._sum_m1_d: Any = torch.zeros(self.r0_m1_d.shape, dtype=torch.float64, device=dev)
            self._sum_car_f: Any = torch.zeros(self.r0_car_f.shape, dtype=torch.float64, device=dev)
        else:
            self._sum_m1_f = np.zeros_like(self.r0_m1_f)
            self._sum_m1_d = np.zeros_like(self.r0_m1_d)
            self._sum_car_f = np.zeros_like(self.r0_car_f)

        self.z_id_sum = np.zeros(self.id_dim, dtype=np.float64)
        self._pending_id_emb: np.ndarray | None = None
        self._pending_sr_covs: tuple[Any, Any, Any] | None = None
        self._cached_z_sr: tuple[np.ndarray, Any] | None = None
        self.last_z_sr: np.ndarray | None = None
        self.last_z_ee: np.ndarray | None = None
        self._active_pc_u = None
        self._sr_u_hist: list[np.ndarray] = []
        self._n_sr_buf = 0
        self._m1f_buf_np: np.ndarray | None = None
        self._m1d_buf_np: np.ndarray | None = None
        self._carf_buf_np: np.ndarray | None = None
        self._m1f_hist_gpu: list[torch.Tensor] = []
        self._m1d_hist_gpu: list[torch.Tensor] = []
        self._carf_hist_gpu: list[torch.Tensor] = []

        self.last_res_dualmod = np.zeros(3, dtype=np.float64)
        self.last_res_joint = np.zeros(3, dtype=np.float64)
        self.last_res_proto_st = np.zeros(3, dtype=np.float64)
        self.last_p_dualmod = np.full(3, 1.0 / 3.0, dtype=np.float64)

        for br_spec in self._branches.values():
            feat_d = int(br_spec["D"])
            spec_k = int(br_spec["K"])
            br_spec["bal_anc"] = self._make_balancer_state(self._bal_mode)
            br_spec["bal_pop"] = self._make_balancer_state(self._bal_mode)
            br_spec["bal_pop_static"] = self._make_balancer_state(self._bal_mode)
            br_spec["n_ok"] = 0
            br_spec["z_sum"] = np.zeros(feat_d, dtype=np.float64)
            br_spec["ll_excess"] = np.zeros(spec_k, dtype=np.float64)
            br_spec["class_z_sum"] = np.zeros((3, feat_d), dtype=np.float64)
            br_spec["P_class_z_sum"] = np.zeros((3, feat_d), dtype=np.float64)
            br_spec["class_w_sum"] = np.zeros(3, dtype=np.float64)
            br_spec["u_ema_self"] = np.full(3, 1.0 / 3.0, dtype=np.float64)
            br_spec["last_obs_cache"] = None
            br_spec["u_hist"] = []
            br_spec["n_br"] = 0
            br_spec["z_buf"] = None
            br_spec["zrec_buf"] = None
            br_spec["wcls_buf"] = None
            br_spec["last_w_sp"] = 0.0
            br_spec["last_kb"] = 0

    @property
    def last_specialist_weight(self) -> float:
        first_branch = next(iter(self._branches.values()), None)
        return float(first_branch["last_w_sp"]) if first_branch is not None else 0.0

    def compute_excess_biometric_margin(self, min_clean_windows: int = 8) -> float:
        if self.n_ok_sr < int(min_clean_windows):
            return -999.0
        z_mean = self.z_id_sum / float(self.n_ok_sr)
        diffs = self.emb_p_mat - z_mean[None, :]
        d2_sorted = np.sort(np.einsum("ki,ij,kj->k", diffs, self.cov_w_inv, diffs))
        raw_margin = float(d2_sorted[1] - d2_sorted[0]) if len(d2_sorted) >= 2 else 0.0
        return raw_margin - self.tau_mrg

    @staticmethod
    def _pc_rbf_weights(u_cur: np.ndarray, u_hist: list[np.ndarray], tau: float) -> np.ndarray:
        if not u_hist or tau <= 0.0:
            return np.zeros(0, dtype=np.float64)
        u_mat = np.stack(u_hist, axis=0)
        sq_d = np.sum((u_mat - u_cur[None, :]) ** 2, axis=1)
        return np.exp(-0.5 * sq_d / (float(u_cur.shape[0]) * (tau * tau)))

    @classmethod
    def _sym_index_np(cls, dim_n: int) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        if dim_n not in cls._SYM_INDEX_NP:
            iu = np.triu_indices(dim_n)
            sc = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0)).astype(np.float64)
            cls._SYM_INDEX_NP[dim_n] = (iu, sc)
        return cls._SYM_INDEX_NP[dim_n]

    @classmethod
    def _mat2vec_np(cls, mat: np.ndarray, scale_mult: float = 1.0) -> np.ndarray:
        iu, sc = cls._sym_index_np(mat.shape[-1])
        return mat[..., iu[0], iu[1]] * (sc * scale_mult)

    @staticmethod
    def _eig_fn_np(mat: np.ndarray, scalar_fn: Any) -> np.ndarray:
        m_sym = 0.5 * (mat + np.swapaxes(mat, -1, -2))
        eig_vals, eig_vecs = np.linalg.eigh(m_sym)
        out_m = (eig_vecs * scalar_fn(eig_vals)[..., None, :]) @ np.swapaxes(eig_vecs, -1, -2)
        return 0.5 * (out_m + np.swapaxes(out_m, -1, -2))

    @classmethod
    def _logm_np(cls, mat: np.ndarray) -> np.ndarray:
        return cls._eig_fn_np(mat, lambda w_v: np.log(np.maximum(w_v, 1e-12)))

    @classmethod
    def _inv_sqrtm_np(cls, mat: np.ndarray) -> np.ndarray:
        return cls._eig_fn_np(mat, lambda w_v: 1.0 / np.sqrt(np.maximum(w_v, 1e-12)))

    @staticmethod
    def _lw_cov_single_np(win_x: np.ndarray) -> np.ndarray:
        xc = win_x - np.mean(win_x, axis=-1, keepdims=True)
        ch_cnt, t_cnt = xc.shape
        s_mat = (xc @ xc.T) / float(t_cnt)
        eye_c = np.eye(ch_cnt, dtype=np.float64)
        mu_tr = float(np.trace(s_mat)) / float(ch_cnt)
        delta_t = float(np.sum((s_mat - mu_tr * eye_c) ** 2)) / float(ch_cnt)
        x2 = xc ** 2
        beta_t = float(np.sum(((x2 @ x2.T) / float(t_cnt) - s_mat ** 2))) / float(ch_cnt * t_cnt)
        beta_t = min(beta_t, delta_t)
        shrink_s = (beta_t / max(delta_t, 1e-300)) if delta_t > 0.0 else 0.0
        return (1.0 - shrink_s) * s_mat + (shrink_s * mu_tr) * eye_c

    @classmethod
    def _project_and_shrink_np(cls, s41: np.ndarray, proj_p: np.ndarray | None, shrink: float = 0.10) -> np.ndarray:
        s_proj = s41 if proj_p is None else np.matmul(np.matmul(proj_p, s41), proj_p.T)
        s_proj = 0.5 * (s_proj + np.swapaxes(s_proj, -1, -2))
        dim_d = s_proj.shape[-1]
        tr_mean = np.maximum(np.trace(s_proj, axis1=-2, axis2=-1) / float(dim_d), 1e-12)[..., None, None]
        return (1.0 - shrink) * s_proj + shrink * tr_mean * np.eye(dim_d, dtype=np.float64)

    @classmethod
    def _sym_index_torch(cls, dim_d: int, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (int(dim_d), str(dev))
        if key not in cls._SYM_INDEX_TORCH:
            iu = np.triu_indices(int(dim_d))
            sc = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0)).astype(np.float64)
            cls._SYM_INDEX_TORCH[key] = (
                torch.as_tensor(iu[0], dtype=torch.long, device=dev),
                torch.as_tensor(iu[1], dtype=torch.long, device=dev),
                torch.as_tensor(sc, dtype=torch.float64, device=dev),
            )
        return cls._SYM_INDEX_TORCH[key]

    @classmethod
    def _mat2vec_torch(cls, mat_t: torch.Tensor, scale_mult: float | None = None) -> torch.Tensor:
        iu0, iu1, sc = cls._sym_index_torch(mat_t.shape[-1], mat_t.device)
        return mat_t[..., iu0, iu1] * (sc if scale_mult is None else sc * scale_mult)

    @staticmethod
    def _eigh_many_torch(mats: list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if not mats:
            return []
        dim_d = mats[0].shape[-1]
        flat = torch.cat([m_item.reshape(-1, dim_d, dim_d) for m_item in mats], dim=0)
        if flat.is_cuda and os.environ.get("R31_FORCE_DET", "0") == "1":
            w_cpu, v_cpu = torch.linalg.eigh(flat.cpu())
            w_all, v_all = w_cpu.to(flat.device), v_cpu.to(flat.device)
        else:
            w_all, v_all = torch.linalg.eigh(flat)
        out_list: list[tuple[torch.Tensor, torch.Tensor]] = []
        offset_k = 0
        for m_item in mats:
            n_items = int(np.prod(m_item.shape[:-2])) if m_item.dim() > 2 else 1
            out_list.append(
                (w_all[offset_k : offset_k + n_items].reshape(m_item.shape[:-1]), v_all[offset_k : offset_k + n_items].reshape(m_item.shape))
            )
            offset_k += n_items
        return out_list

    @staticmethod
    def _eig_fn_torch(eig_vals: torch.Tensor, eig_vecs: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == "log":
            transformed = torch.log(torch.clamp(eig_vals, min=1e-12))
        elif kind == "isqrt":
            transformed = 1.0 / torch.sqrt(torch.clamp(eig_vals, min=1e-12))
        else:
            transformed = torch.exp(eig_vals)
        reconstructed = (eig_vecs * transformed.unsqueeze(-2)) @ eig_vecs.transpose(-1, -2)
        return 0.5 * (reconstructed + reconstructed.transpose(-1, -2))

    @staticmethod
    def _lw_cov_single_torch(win_t: torch.Tensor) -> torch.Tensor:
        xc = win_t - torch.mean(win_t, dim=-1, keepdim=True)
        ch_cnt, t_cnt = xc.shape
        s_mat = (xc @ xc.T) / float(t_cnt)
        eye_c = torch.eye(ch_cnt, dtype=torch.float64, device=win_t.device)
        mu_tr = float(torch.trace(s_mat)) / float(ch_cnt)
        delta_t = float(torch.sum((s_mat - mu_tr * eye_c) ** 2)) / float(ch_cnt)
        x2 = xc ** 2
        beta_t = float(torch.sum(((x2 @ x2.T) / float(t_cnt) - s_mat ** 2))) / float(ch_cnt * t_cnt)
        beta_t = min(beta_t, delta_t)
        shrink_s = (beta_t / max(delta_t, 1e-300)) if delta_t > 0.0 else 0.0
        return (1.0 - shrink_s) * s_mat + (shrink_s * mu_tr) * eye_c

    @staticmethod
    def _project_and_shrink_torch(s41: torch.Tensor, proj_p: torch.Tensor | None, shrink: float = 0.10) -> torch.Tensor:
        s_proj = s41 if proj_p is None else torch.matmul(torch.matmul(proj_p, s41), proj_p.T)
        s_proj = 0.5 * (s_proj + s_proj.transpose(-1, -2))
        dim_d = s_proj.shape[-1]
        tr_mean = torch.clamp(torch.diagonal(s_proj, dim1=-2, dim2=-1).sum(-1) / float(dim_d), min=1e-12)[..., None, None]
        return (1.0 - shrink) * s_proj + shrink * tr_mean * torch.eye(dim_d, dtype=torch.float64, device=s_proj.device)

    @staticmethod
    def collapse_eegpt_features(
        all_f: np.ndarray | None = None,
        n_lat: int = 8,
        n_clean4: int = 0,
        n_rms4c: int = 0,
        f_lat: np.ndarray | None = None,
        f_rms: np.ndarray | None = None,
        f_c4: np.ndarray | None = None,
        ee_slice_mode: str = "lat_rms",
    ) -> np.ndarray:
        fc4_arr = f_c4
        if all_f is not None:
            f_arr = np.asarray(all_f, dtype=np.float64)
            fl = f_arr[: int(n_lat)]
            if int(n_clean4) > 0:
                fc4_arr = f_arr[int(n_lat) : int(n_lat) + int(n_clean4)]
            fr = f_arr[int(n_lat) + int(n_clean4) + int(n_rms4c) :]
        else:
            if f_lat is None or f_rms is None:
                raise ValueError("Provide either all_f or both f_lat and f_rms")
            fl = np.asarray(f_lat, dtype=np.float64)
            fr = np.asarray(f_rms, dtype=np.float64)
        fl_v = fl.mean(axis=0) if fl.ndim == 2 else fl.reshape(-1)
        fr_v = fr.mean(axis=0) if fr.ndim == 2 else fr.reshape(-1)
        if ee_slice_mode == "lat_c4" and fc4_arr is not None and len(fc4_arr) > 0:
            fc4_v = np.asarray(fc4_arr, dtype=np.float64)
            fc4_v = fc4_v.mean(axis=0) if fc4_v.ndim == 2 else fc4_v.reshape(-1)
            return np.concatenate([fl_v, fc4_v], axis=0)
        if ee_slice_mode == "lat_rms_c4" and fc4_arr is not None and len(fc4_arr) > 0:
            fc4_v = np.asarray(fc4_arr, dtype=np.float64)
            fc4_v = fc4_v.mean(axis=0) if fc4_v.ndim == 2 else fc4_v.reshape(-1)
            return np.concatenate([fl_v, fr_v, fc4_v], axis=0)
        return np.concatenate([fl_v, fr_v], axis=0)

    def prepare_pop_eegpt_inputs(
        self,
        cleaned_window_f64: np.ndarray,
        fir52: np.ndarray,
        w0_52_override: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        xb = np.asarray(cleaned_window_f64, dtype=np.float64)
        if not np.isfinite(xb).all():
            xb = np.nan_to_num(xb, nan=0.0, posinf=500.0, neginf=-500.0)
        xz = xb - np.mean(xb, axis=-1, keepdims=True)
        xcar = xz - np.mean(xz, axis=0, keepdims=True)
        xpop = xcar * self.inv_std0_id[:, None]
        xc52 = fftconvolve(xz, np.asarray(fir52, dtype=np.float64).reshape(1, -1), mode="same", axes=-1)
        w52 = self.w0_52 if w0_52_override is None else np.asarray(w0_52_override, dtype=np.float64)
        return (w52 @ xc52).astype(np.float32), xpop.astype(np.float32)

    def _extract_id_emb(
        self,
        cleaned_window_f64: np.ndarray,
        z_pop_ee: np.ndarray | None,
        known_covs: dict[Any, np.ndarray] | None = None,
        z_pop_c4_512: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._gpu_consts is not None:
            return self._extract_id_emb_cuda(cleaned_window_f64, z_pop_ee, known_covs, z_pop_c4_512)
        xb = np.asarray(cleaned_window_f64, dtype=np.float64)
        if not np.isfinite(xb).all():
            xb = np.nan_to_num(xb, nan=0.0, posinf=500.0, neginf=-500.0)
        xz = xb - np.mean(xb, axis=-1, keepdims=True)
        xcar = xz - np.mean(xz, axis=0, keepdims=True)
        xpop = xcar * self.inv_std0_id[:, None]

        xf_7 = np.matmul(xpop, self._m_sos_7)
        lp = np.log(np.maximum(np.mean(xf_7 ** 2, axis=-1), 1e-6)).reshape(-1)
        ve = np.log(np.maximum(np.mean(xf_7[:6, :, self.SLICE_EARLY] ** 2, axis=-1), 1e-6))
        vl = np.log(np.maximum(np.mean(xf_7[:6, :, self.SLICE_LATE] ** 2, axis=-1), 1e-6))
        spec_533 = np.concatenate([lp, (vl - ve).reshape(-1)], axis=0)

        known = known_covs or {}
        cb6 = np.empty((6, self._n_chans, self._n_chans), dtype=np.float64)
        missing = [bi for bi, bk in enumerate(self.RIEM_BANDS_6) if bk not in known]
        if missing:
            xf_6 = np.matmul(xz, self._m_sos_6[missing])
            for m_i, bi in enumerate(missing):
                cb6[bi] = self._lw_cov_single_np(xf_6[m_i])
        for bi, bk in enumerate(self.RIEM_BANDS_6):
            if bk in known:
                cb6[bi] = np.asarray(known[bk], dtype=np.float64)

        a6 = np.matmul(np.matmul(self.w0_6b_id, cb6), self.w0_6b_id)
        tang_raw = self._mat2vec_np(self._logm_np(a6)).reshape(-1)
        parts = [spec_533, ((tang_raw - self.tg_mu) / self.tg_sd) @ self.p_tg]
        if self.has_pop_eegpt:
            if z_pop_ee is None:
                raise ValueError("AnchorSpecialistMultimodalOrgan requires all_f_pop for full Z_id")
            parts.append(((np.asarray(z_pop_ee, dtype=np.float64) - self.ep_mu) / self.ep_sd) @ self.p_ep)
        if self.has_pop_c4:
            if z_pop_c4_512 is None:
                raise ValueError("AnchorSpecialistMultimodalOrgan requires z_pop_c4_512 when P_c4_pop is present")
            parts.append(((np.asarray(z_pop_c4_512, dtype=np.float64) - self.c4_pop_mu) / self.c4_pop_sd) @ self.p_c4_pop)
        z_id = np.concatenate(parts, axis=0)
        return ((z_id - self.id_mu_glob) / self.id_sd_glob) @ self.id_m_lda

    def _extract_id_emb_cuda(
        self,
        cleaned_window_f64: np.ndarray,
        z_pop_ee: np.ndarray | None,
        known_covs: dict[Any, np.ndarray] | None,
        z_pop_c4_512: np.ndarray | None,
    ) -> np.ndarray:
        assert self._gpu_consts is not None
        g = self._gpu_consts
        dev = g["dev"]
        xb = np.asarray(cleaned_window_f64, dtype=np.float64)
        if not np.isfinite(xb).all():
            xb = np.nan_to_num(xb, nan=0.0, posinf=500.0, neginf=-500.0)
        xz = xb - np.mean(xb, axis=-1, keepdims=True)
        xcar = xz - np.mean(xz, axis=0, keepdims=True)
        xpop = xcar * self.inv_std0_id[:, None]
        with torch.no_grad():
            xf_7 = torch.matmul(torch.as_tensor(np.ascontiguousarray(xpop), dtype=torch.float64, device=dev), g["_m_sos_7"])
            lp = torch.log(torch.clamp(torch.mean(xf_7 ** 2, dim=-1), min=1e-6)).reshape(-1)
            ve = torch.log(torch.clamp(torch.mean(xf_7[:6, :, self.SLICE_EARLY] ** 2, dim=-1), min=1e-6))
            vl = torch.log(torch.clamp(torch.mean(xf_7[:6, :, self.SLICE_LATE] ** 2, dim=-1), min=1e-6))
            spec_533 = torch.cat([lp, (vl - ve).reshape(-1)], dim=0)

            known = known_covs or {}
            missing = [bi for bi, bk in enumerate(self.RIEM_BANDS_6) if bk not in known]
            cb6_list: list[torch.Tensor | None] = [None] * 6
            if missing:
                xf_6 = torch.matmul(
                    torch.as_tensor(np.ascontiguousarray(xz), dtype=torch.float64, device=dev), g["_m_sos_6"][missing]
                )
                for m_i, bi in enumerate(missing):
                    cb6_list[bi] = self._lw_cov_single_torch(xf_6[m_i])
            have = [bi for bi, bk in enumerate(self.RIEM_BANDS_6) if bk in known]
            if have:
                kn = torch.as_tensor(
                    np.ascontiguousarray(np.stack([np.asarray(known[self.RIEM_BANDS_6[bi]], dtype=np.float64) for bi in have])),
                    dtype=torch.float64,
                    device=dev,
                )
                for j_i, bi in enumerate(have):
                    cb6_list[bi] = kn[j_i]
            cb6 = torch.stack([item_t for item_t in cb6_list if item_t is not None], dim=0)
            a6 = torch.matmul(torch.matmul(g["w0_6b_id"], cb6), g["w0_6b_id"])
            w_eig, v_eig = self._eigh_many_torch([0.5 * (a6 + a6.transpose(-1, -2))])[0]
            tang_raw = self._mat2vec_torch(self._eig_fn_torch(w_eig, v_eig, "log")).reshape(-1)
            parts = [spec_533, ((tang_raw - g["tg_mu"]) / g["tg_sd"]) @ g["p_tg"]]
            if self.has_pop_eegpt:
                if z_pop_ee is None:
                    raise ValueError("AnchorSpecialistMultimodalOrgan requires all_f_pop for full Z_id")
                z_ee_t = torch.as_tensor(np.ascontiguousarray(z_pop_ee, dtype=np.float64), dtype=torch.float64, device=dev)
                parts.append(((z_ee_t - g["ep_mu"]) / g["ep_sd"]) @ g["p_ep"])
            if self.has_pop_c4:
                if z_pop_c4_512 is None:
                    raise ValueError("AnchorSpecialistMultimodalOrgan requires z_pop_c4_512 when P_c4_pop is present")
                z_c4_t = torch.as_tensor(np.ascontiguousarray(z_pop_c4_512, dtype=np.float64), dtype=torch.float64, device=dev)
                parts.append(((z_c4_t - g["c4_pop_mu"]) / g["c4_pop_sd"]) @ g["p_c4_pop"])
            z_id = torch.cat(parts, dim=0)
            return (((z_id - g["id_mu_glob"]) / g["id_sd_glob"]) @ g["id_m_lda"]).cpu().numpy()

    def _extract_sr_feat_cuda(
        self, raw_window_f64: np.ndarray, s_41: np.ndarray | torch.Tensor | None, cur_u: np.ndarray | None
    ) -> np.ndarray:
        assert self._gpu_consts is not None
        g = self._gpu_consts
        dev = g["dev"]
        with torch.no_grad():
            if s_41 is None:
                x_np = np.asarray(raw_window_f64, dtype=np.float64)
                if not np.isfinite(x_np).all():
                    x_np = np.nan_to_num(x_np, nan=0.0, posinf=500.0, neginf=-500.0)
                xc = x_np - np.mean(x_np, axis=-1, keepdims=True)
                yb = torch.matmul(torch.as_tensor(np.ascontiguousarray(xc), dtype=torch.float64, device=dev), g["_m_sos_9"])

                def _seg_cov(band_t: torch.Tensor, s0: int, s1: int) -> torch.Tensor:
                    seg = band_t[..., s0:s1]
                    seg = seg - torch.mean(seg, dim=-1, keepdim=True)
                    return torch.matmul(seg, seg.transpose(-1, -2)) / float(s1 - s0)

                s_base_diff = torch.stack([_seg_cov(yb[bi], 0, 120) for bi in self.MORTP_DIFF_IDX], dim=0)
                s_motor_diff = torch.stack([_seg_cov(yb[bi], 180, 480) for bi in self.MORTP_DIFF_IDX], dim=0)
                s_full_6 = torch.stack([_seg_cov(yb[bi], 0, self._n_times) for bi in self.MORTP_FULL_IDX], dim=0)
            else:
                s41_t = s_41 if torch.is_tensor(s_41) else torch.as_tensor(np.ascontiguousarray(s_41, dtype=np.float64), device=dev)
                s_full_6 = s41_t[3, list(self.MORTP_FULL_IDX)]
                s_base_diff = s41_t[0, list(self.MORTP_DIFF_IDX)]
                s_motor_diff = s41_t[2, list(self.MORTP_DIFF_IDX)]

            m1_f = self._project_and_shrink_torch(s_full_6, None, self.MORTP_SHRINK)
            m1_db = self._project_and_shrink_torch(s_base_diff, None, self.MORTP_SHRINK)
            m1_dm = self._project_and_shrink_torch(s_motor_diff, None, self.MORTP_SHRINK)
            car_f = self._project_and_shrink_torch(s_full_6[:5], g["p_car40"], self.MORTP_SHRINK)

            if self._zsr_pc_tau > 0.0 and cur_u is not None and self._sr_u_hist:
                w_sr = self._pc_rbf_weights(cur_u, self._sr_u_hist, self._zsr_pc_tau)
                wt_sr = torch.as_tensor(w_sr, dtype=torch.float64, device=dev)
                denom = 10.0 + float(w_sr.sum())
                gm1 = (10.0 * g["r0_m1_f"] + torch.tensordot(wt_sr, torch.stack(self._m1f_hist_gpu, dim=0), dims=([0], [0]))) / denom
                gd = (10.0 * g["r0_m1_d"] + torch.tensordot(wt_sr, torch.stack(self._m1d_hist_gpu, dim=0), dims=([0], [0]))) / denom
                gcar = (10.0 * g["r0_car_f"] + torch.tensordot(wt_sr, torch.stack(self._carf_hist_gpu, dim=0), dims=([0], [0]))) / denom
            else:
                denom = 10.0 + float(self.n_ok_sr)
                gm1 = (10.0 * g["r0_m1_f"] + self._sum_m1_f) / denom
                gd = (10.0 * g["r0_m1_d"] + self._sum_m1_d) / denom
                gcar = (10.0 * g["r0_car_f"] + self._sum_car_f) / denom

            r1 = self._eigh_many_torch([0.5 * (gm1 + gm1.transpose(-1, -2)), 0.5 * (gd + gd.transpose(-1, -2))])
            r1c = self._eigh_many_torch([0.5 * (gcar + gcar.transpose(-1, -2))])
            w_m1 = self._eig_fn_torch(*r1[0], "isqrt")
            w_d = self._eig_fn_torch(*r1[1], "isqrt")
            w_car = self._eig_fn_torch(*r1c[0], "isqrt")

            a_m1 = torch.matmul(torch.matmul(w_m1, m1_f), w_m1)
            a_dm = torch.matmul(torch.matmul(w_d, m1_dm), w_d)
            a_db = torch.matmul(torch.matmul(w_d, m1_db), w_d)
            a_car = torch.matmul(torch.matmul(w_car, car_f), w_car)
            r2 = self._eigh_many_torch(
                [
                    0.5 * (a_m1 + a_m1.transpose(-1, -2)),
                    0.5 * (a_dm + a_dm.transpose(-1, -2)),
                    0.5 * (a_db + a_db.transpose(-1, -2)),
                ]
            )
            r2c = self._eigh_many_torch([0.5 * (a_car + a_car.transpose(-1, -2))])
            t_m1 = self._mat2vec_torch(self._eig_fn_torch(*r2[0], "log"), scale_mult=1.0 / np.sqrt(6.0)).reshape(-1)
            t_dm = self._mat2vec_torch(self._eig_fn_torch(*r2[1], "log"), scale_mult=1.0 / np.sqrt(3.0)).reshape(-1)
            t_db = self._mat2vec_torch(self._eig_fn_torch(*r2[2], "log"), scale_mult=1.0 / np.sqrt(3.0)).reshape(-1)
            t_car = self._mat2vec_torch(self._eig_fn_torch(*r2c[0], "log"), scale_mult=1.0 / np.sqrt(5.0)).reshape(-1)
            t_sr = torch.cat([t_m1, t_dm - t_db, t_car], dim=0)
            self._pending_sr_covs = (m1_f, 0.5 * (m1_db + m1_dm), car_f)
            return (((t_sr - g["sr_mu"]) / g["sr_sd"]) @ g["p_sr"]).cpu().numpy()

    def _extract_sr_feat(
        self, raw_window_f64: np.ndarray, s_41: np.ndarray | torch.Tensor | None, cur_u: np.ndarray | None
    ) -> np.ndarray:
        if self._gpu_consts is not None:
            return self._extract_sr_feat_cuda(raw_window_f64, s_41, cur_u)
        if torch.is_tensor(s_41):
            s_41 = s_41.cpu().numpy()
        if s_41 is None:
            x_np = np.asarray(raw_window_f64, dtype=np.float64)
            if not np.isfinite(x_np).all():
                x_np = np.nan_to_num(x_np, nan=0.0, posinf=500.0, neginf=-500.0)
            xc = x_np - np.mean(x_np, axis=-1, keepdims=True)
            yb = np.matmul(xc, self._m_sos_9)
            s_base_diff = np.empty((3, self._n_chans, self._n_chans), dtype=np.float64)
            s_motor_diff = np.empty((3, self._n_chans, self._n_chans), dtype=np.float64)
            for k_i, bi in enumerate(self.MORTP_DIFF_IDX):
                sb = yb[bi, :, 0:120]
                sb = sb - np.mean(sb, axis=-1, keepdims=True)
                s_base_diff[k_i] = (sb @ sb.T) / 120.0
                sm = yb[bi, :, 180:480]
                sm = sm - np.mean(sm, axis=-1, keepdims=True)
                s_motor_diff[k_i] = (sm @ sm.T) / 300.0
            s_full_6 = np.empty((6, self._n_chans, self._n_chans), dtype=np.float64)
            for k_i, bi in enumerate(self.MORTP_FULL_IDX):
                sf = yb[bi] - np.mean(yb[bi], axis=-1, keepdims=True)
                s_full_6[k_i] = (sf @ sf.T) / float(self._n_times)
        else:
            s41_np = np.asarray(s_41, dtype=np.float64)
            s_full_6 = s41_np[3, self.MORTP_FULL_IDX]
            s_base_diff = s41_np[0, self.MORTP_DIFF_IDX]
            s_motor_diff = s41_np[2, self.MORTP_DIFF_IDX]

        m1_f = self._project_and_shrink_np(s_full_6, None, self.MORTP_SHRINK)
        m1_db = self._project_and_shrink_np(s_base_diff, None, self.MORTP_SHRINK)
        m1_dm = self._project_and_shrink_np(s_motor_diff, None, self.MORTP_SHRINK)
        car_f = self._project_and_shrink_np(s_full_6[:5], self.p_car40, self.MORTP_SHRINK)

        if self._zsr_pc_tau > 0.0 and cur_u is not None and self._n_sr_buf > 0 and self._m1f_buf_np is not None:
            w_sr = self._pc_rbf_weights(cur_u, self._sr_u_hist, self._zsr_pc_tau)
            denom = 10.0 + float(w_sr.sum())
            assert self._m1d_buf_np is not None and self._carf_buf_np is not None
            gm1 = (10.0 * self.r0_m1_f + np.tensordot(w_sr, self._m1f_buf_np[: self._n_sr_buf], axes=(0, 0))) / denom
            gd = (10.0 * self.r0_m1_d + np.tensordot(w_sr, self._m1d_buf_np[: self._n_sr_buf], axes=(0, 0))) / denom
            gcar = (10.0 * self.r0_car_f + np.tensordot(w_sr, self._carf_buf_np[: self._n_sr_buf], axes=(0, 0))) / denom
        else:
            denom = 10.0 + float(self.n_ok_sr)
            gm1 = (10.0 * self.r0_m1_f + self._sum_m1_f) / denom
            gd = (10.0 * self.r0_m1_d + self._sum_m1_d) / denom
            gcar = (10.0 * self.r0_car_f + self._sum_car_f) / denom

        w_m1 = self._inv_sqrtm_np(gm1)
        w_d = self._inv_sqrtm_np(gd)
        w_car = self._inv_sqrtm_np(gcar)

        t_m1 = self._mat2vec_np(self._logm_np(np.matmul(np.matmul(w_m1, m1_f), w_m1)), scale_mult=1.0 / np.sqrt(6.0)).reshape(-1)
        t_dm = self._mat2vec_np(self._logm_np(np.matmul(np.matmul(w_d, m1_dm), w_d)), scale_mult=1.0 / np.sqrt(3.0)).reshape(-1)
        t_db = self._mat2vec_np(self._logm_np(np.matmul(np.matmul(w_d, m1_db), w_d)), scale_mult=1.0 / np.sqrt(3.0)).reshape(-1)
        t_car = self._mat2vec_np(self._logm_np(np.matmul(np.matmul(w_car, car_f), w_car)), scale_mult=1.0 / np.sqrt(5.0)).reshape(-1)

        t_sr = np.concatenate([t_m1, t_dm - t_db, t_car], axis=0)
        self._pending_sr_covs = (m1_f, 0.5 * (m1_db + m1_dm), car_f)
        return ((t_sr - self.sr_mu) / self.sr_sd) @ self.p_sr

    def precompute_sr_cpu(self, raw_window_f64: np.ndarray, pseudo_chrono_u: np.ndarray | None = None) -> None:
        if self._gpu_consts is not None:
            return
        cur_u = np.asarray(pseudo_chrono_u, dtype=np.float64) if pseudo_chrono_u is not None else self._active_pc_u
        z_sr = self._extract_sr_feat(raw_window_f64, None, cur_u)
        self._cached_z_sr = (z_sr, self._pending_sr_covs)

    def _score_branch(
        self, br: dict[str, Any], z_feat: np.ndarray, cur_u: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z_t = (np.asarray(z_feat, dtype=np.float64) - br["mu_z"]) / br["sd_z"]
        w_pc_past: np.ndarray | None = None
        n_br = int(br["n_br"])
        n_ok = int(br["n_ok"])

        if self._rec_mode == "pc" and n_ok > 0 and cur_u is not None and n_br > 0 and br["z_buf"] is not None:
            w_pc_past = self._pc_rbf_weights(cur_u, br["u_hist"], self._pc_tau)
            w_pc_sum = float(w_pc_past.sum())
            z_glob = (self._n0_glob * (br["z_sum"] / float(n_ok))) if self._n0_glob > 0.0 else 0.0
            g_den = self._n0_glob if self._n0_glob > 0.0 else 0.0
            z_rec = z_t - ((w_pc_past @ br["z_buf"][:n_br]) + z_glob) / (w_pc_sum + g_den + self._n0_prior)
        elif n_ok > 0:
            gamma_rec = float(n_ok) / (float(n_ok) + 12.0)
            z_rec = z_t - gamma_rec * (br["z_sum"] / float(n_ok))
        else:
            z_rec = z_t

        spec_k = int(br["K"])
        v_spec = np.zeros(spec_k + 1, dtype=np.float64)
        kb = 0
        w_sp = 0.0
        if n_ok >= self._min_router_n:
            z_id_mean = self.z_id_sum / float(n_ok)
            diffs = self.emb_p_mat - z_id_mean[None, :]
            d2_vec = np.einsum("ki,ij,kj->k", diffs, self.cov_w_inv, diffs)
            ord_k = np.argsort(d2_vec)
            kb = int(ord_k[0])
            d2_second = float(d2_vec[ord_k[1]]) if spec_k >= 2 else float(d2_vec[kb] + 100.0)
            mrg = d2_second - float(d2_vec[kb])
            ramp = float(n_ok) / (float(n_ok) + self._ramp_n0)
            gate = min(1.0, (mrg - self.tau_mrg) / 16.0) if mrg > self.tau_mrg else 0.0
            w_sp = ramp * gate
            v_spec[0] = 1.0 - w_sp
            v_spec[1 + kb] = w_sp
        else:
            v_spec[0] = 1.0
        br["last_w_sp"] = float(w_sp)
        br["last_kb"] = int(kb)

        lg_pop_lr = z_rec @ br["W_pop"].T + br["b_pop"]
        if w_sp > 0.0:
            w_anc = v_spec[0] * br["W_pop"] + w_sp * br["W_stack"][kb]
            b_anc = v_spec[0] * br["b_pop"] + w_sp * br["b_stack"][kb]
            lg_lr = z_rec @ w_anc.T + b_anc
        else:
            lg_lr = lg_pop_lr

        nu0_eff = float(br["nu0"])
        if self._proto_mode == "pc" and cur_u is not None and n_br > 0 and br["zrec_buf"] is not None:
            if w_pc_past is None:
                w_pc_past = self._pc_rbf_weights(cur_u, br["u_hist"], self._pc_tau)
            kp = 0.5 + 0.5 * w_pc_past
            wk = br["wcls_buf"][:n_br] * kp[:, None]
            cz = wk.T @ br["zrec_buf"][:n_br]
            pcz = (cz.astype(np.float32) @ br["Prec_f32"]).astype(np.float64)
            cw = wk.sum(axis=0)
        else:
            cz = br["class_z_sum"]
            pcz = br["P_class_z_sum"]
            cw = br["class_w_sum"]

        denom_cls = (nu0_eff + cw)[:, None]
        mu_live_pop = (nu0_eff * br["mu_pop_c"] + cz) / denom_cls
        p_mu_pop = (nu0_eff * br["P_mu_pop_c"] + pcz) / denom_cls
        lp_pop_p = (p_mu_pop @ z_rec) - 0.5 * np.sum(p_mu_pop * mu_live_pop, axis=1)
        lp_pop_p -= np.mean(lp_pop_p)
        alpha_pr = float(br["alpha_proto"])
        gamma_lda = float(br["gamma_lda"])
        lg_pop_tot = (1.0 - alpha_pr) * lg_pop_lr + alpha_pr * gamma_lda * lp_pop_p

        if w_sp > 0.0:
            mu_anc_c = v_spec[0] * br["mu_pop_c"] + w_sp * br["mu_p_c_stack"][kb]
            mu_live_anc = (nu0_eff * mu_anc_c + cz) / denom_cls
            p_mu_anc_c = v_spec[0] * br["P_mu_pop_c"] + w_sp * br["P_mu_p_c"][kb]
            p_mu_anc = (nu0_eff * p_mu_anc_c + pcz) / denom_cls
            lp_anc_p = (p_mu_anc @ z_rec) - 0.5 * np.sum(p_mu_anc * mu_live_anc, axis=1)
            lp_anc_p -= np.mean(lp_anc_p)
            lg_tot = (1.0 - alpha_pr) * lg_lr + alpha_pr * gamma_lda * lp_anc_p
        else:
            lg_tot = lg_pop_tot

        u_pop_t = self.softmax_1d(lg_pop_tot)
        u_anc_t = self.softmax_1d(lg_tot)
        p_pop_lr_t = self.softmax_1d(lg_pop_lr)
        if self._bal_mode == "raw_u":
            p_anc_bal = u_anc_t
            p_pop_bal = u_pop_t
            p_pop_static_bal = p_pop_lr_t
        else:
            p_anc_bal = self._step_balancer_state(br["bal_anc"], u_anc_t)
            p_pop_bal = self._step_balancer_state(br["bal_pop"], u_pop_t)
            p_pop_static_bal = self._step_balancer_state(br["bal_pop_static"], p_pop_lr_t)

        res_lp = self.centered_log_prob_1d(p_anc_bal) - self.centered_log_prob_1d(p_pop_bal)
        br["last_obs_cache"] = (z_t, z_rec, p_pop_lr_t, u_anc_t)
        return res_lp, u_anc_t, u_pop_t, v_spec, p_anc_bal, p_pop_bal, p_pop_static_bal

    def _observe_branch(
        self, br: dict[str, Any], is_bad_bool: bool, p_tchr: np.ndarray | None, cur_u: np.ndarray | None
    ) -> None:
        if br["last_obs_cache"] is None or is_bad_bool:
            br["last_obs_cache"] = None
            return
        z_t, z_rec, _, u_anc_t = br["last_obs_cache"]
        br["last_obs_cache"] = None

        conf_min = float(br["conf_min"])
        if p_tchr is not None:
            q_obs = np.asarray(p_tchr, dtype=np.float64)
            c_min_use = conf_min
        else:
            q_debias = u_anc_t / np.maximum(3.0 * br["u_ema_self"], 0.35)
            q_obs = q_debias / np.sum(q_debias)
            br["u_ema_self"] = 0.92 * br["u_ema_self"] + 0.08 * u_anc_t
            c_min_use = max(conf_min, 0.52)

        br["n_ok"] = int(br["n_ok"]) + 1
        br["z_sum"] = br["z_sum"] + z_t

        conf_q = float(np.max(q_obs))
        w_vec = np.zeros(3, dtype=np.float64)
        if conf_q >= c_min_use:
            wt_c = min(1.0, (conf_q - 0.33) / 0.40)
            w_vec = wt_c * q_obs
            br["class_z_sum"] += np.outer(w_vec, z_rec)
            pz_rec = (z_rec.astype(np.float32) @ br["Prec_f32"]).astype(np.float64)
            br["P_class_z_sum"] += np.outer(w_vec, pz_rec)
            br["class_w_sum"] += w_vec

        if cur_u is not None:
            mh = self._max_pc_history
            feat_d = int(br["D"])
            if br["z_buf"] is None or br["z_buf"].shape[0] != mh:
                br["z_buf"] = np.empty((mh, feat_d), dtype=np.float64)
                br["zrec_buf"] = np.empty((mh, feat_d), dtype=np.float64)
                br["wcls_buf"] = np.empty((mh, 3), dtype=np.float64)
                br["n_br"] = 0
            br["u_hist"].append(cur_u)
            nb = int(br["n_br"])
            if nb < mh:
                br["z_buf"][nb] = z_t
                br["zrec_buf"][nb] = z_rec
                br["wcls_buf"][nb] = w_vec
                br["n_br"] = nb + 1
            else:
                br["z_buf"][:-1] = br["z_buf"][1:]
                br["zrec_buf"][:-1] = br["zrec_buf"][1:]
                br["wcls_buf"][:-1] = br["wcls_buf"][1:]
                br["z_buf"][-1] = z_t
                br["zrec_buf"][-1] = z_rec
                br["wcls_buf"][-1] = w_vec
            if len(br["u_hist"]) > mh:
                br["u_hist"].pop(0)

    def score_window(
        self,
        raw_window_f64: np.ndarray | None = None,
        cleaned_window_f64: np.ndarray | None = None,
        all_f_causal: np.ndarray | None = None,
        all_f_pop: np.ndarray | None = None,
        n_lat: int | None = None,
        n_clean4: int = 0,
        n_rms4c: int = 0,
        f_lat: np.ndarray | None = None,
        f_rms: np.ndarray | None = None,
        f_pop_lat: np.ndarray | None = None,
        f_pop_rms: np.ndarray | None = None,
        s_41: np.ndarray | None = None,
        known_covs: dict[Any, np.ndarray] | None = None,
        z_ee_pre: np.ndarray | None = None,
        z_sr_pre: np.ndarray | None = None,
        z_id_pre: np.ndarray | None = None,
        pseudo_chrono_u: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        cur_u = np.asarray(pseudo_chrono_u, dtype=np.float64) if pseudo_chrono_u is not None else self._active_pc_u
        nl = self._n_lat if n_lat is None else int(n_lat)

        if z_sr_pre is not None:
            z_sr = np.asarray(z_sr_pre, dtype=np.float64)
            self._pending_sr_covs = None
            self._cached_z_sr = None
        elif s_41 is None and self._cached_z_sr is not None:
            z_sr, self._pending_sr_covs = self._cached_z_sr
            self._cached_z_sr = None
        else:
            if raw_window_f64 is None and s_41 is None:
                raise ValueError("Provide raw_window_f64 or s_41 (or z_sr_pre)")
            z_sr = self._extract_sr_feat(
                raw_window_f64 if raw_window_f64 is not None else np.empty(0), s_41, cur_u
            )

        if z_id_pre is not None:
            z_id_emb = ((np.asarray(z_id_pre, dtype=np.float64) - self.id_mu_glob) / self.id_sd_glob) @ self.id_m_lda
        else:
            if cleaned_window_f64 is None:
                raise ValueError("Provide cleaned_window_f64 (or z_id_pre)")
            z_pop_ee = None
            z_pop_c4 = None
            if self.has_pop_eegpt:
                pop_slice_mode = "lat_rms_c4" if self.ep_mu.shape[0] == 1536 else "lat_rms"
                z_pop_ee = self.collapse_eegpt_features(
                    all_f=all_f_pop,
                    n_lat=nl,
                    n_clean4=n_clean4,
                    n_rms4c=n_rms4c,
                    f_lat=f_pop_lat,
                    f_rms=f_pop_rms,
                    ee_slice_mode=pop_slice_mode,
                )
            if self.has_pop_c4 and all_f_pop is not None and int(n_clean4) > 0:
                z_pop_c4 = np.asarray(all_f_pop[int(nl) : int(nl) + int(n_clean4)], dtype=np.float64).mean(axis=0)
            z_id_emb = self._extract_id_emb(cleaned_window_f64, z_pop_ee, known_covs=known_covs, z_pop_c4_512=z_pop_c4)
        self._pending_id_emb = z_id_emb

        z_ee: np.ndarray | None = None
        if self.has_ee:
            if z_ee_pre is not None:
                z_ee = np.asarray(z_ee_pre, dtype=np.float64)
            else:
                z_ee = self.collapse_eegpt_features(
                    all_f=all_f_causal,
                    n_lat=nl,
                    n_clean4=n_clean4,
                    n_rms4c=n_rms4c,
                    f_lat=f_lat,
                    f_rms=f_rms,
                    ee_slice_mode=self.ee_slice_mode,
                )
            expected_d = int(self._branches["ee"]["D"])
            if z_ee.shape[0] != expected_d:
                raise ValueError(f"z_ee dim {z_ee.shape[0]} != ee branch D {expected_d}")

        self.last_z_sr = z_sr
        self.last_z_ee = z_ee

        res_sr, u_sr, u_pop_sr, v_sr, p_sr_bal, p_sr_proto, p_sr_static = self._score_branch(
            self._branches["sr"], z_sr, cur_u
        )
        scored_out: dict[str, np.ndarray] = {
            "res_sr": res_sr,
            "u_anc_sr": u_sr,
            "u_pop_sr": u_pop_sr,
            "p_anc_sr": p_sr_bal,
            "v_spec": v_sr,
        }

        if self.has_ee and z_ee is not None:
            res_ee, u_ee, u_pop_ee, _, p_ee_bal, _, _ = self._score_branch(self._branches["ee"], z_ee, cur_u)
            res_dm = 0.5 * (res_ee + res_sr)
            scored_out["res_ee"] = res_ee
            scored_out["u_anc_ee"] = u_ee
            scored_out["u_pop_ee"] = u_pop_ee
            scored_out["p_anc_ee"] = p_ee_bal
            scored_out["res_dualmod"] = res_dm
            scored_out["p_dualmod"] = self.softmax_1d(
                0.5 * self.centered_log_prob_1d(p_ee_bal) + 0.5 * self.centered_log_prob_1d(p_sr_bal)
            )
        else:
            scored_out["res_dualmod"] = res_sr
            scored_out["p_dualmod"] = p_sr_bal

        if self.has_full and z_ee is not None:
            z_mm = np.concatenate([z_ee / self.scale_ee, z_sr / self.scale_sr], axis=0)
            res_fl, u_fl, u_pop_fl, _, p_fl_bal, p_proto_only, p_pop_static = self._score_branch(
                self._branches["full"], z_mm, cur_u
            )
            res_proto_st = self.centered_log_prob_1d(p_proto_only) - self.centered_log_prob_1d(p_pop_static)
            scored_out["res_joint"] = res_fl
            scored_out["u_anc_joint"] = u_fl
            scored_out["u_pop_joint"] = u_pop_fl
            scored_out["p_tchr_s7"] = p_fl_bal
            scored_out["p_proto_only"] = p_proto_only
            scored_out["p_pop_static"] = p_pop_static
            scored_out["res_proto_st"] = res_proto_st
        else:
            res_proto_st = self.centered_log_prob_1d(p_sr_proto) - self.centered_log_prob_1d(p_sr_static)
            scored_out["res_joint"] = scored_out["res_dualmod"]
            scored_out["p_tchr_s7"] = scored_out["p_dualmod"]
            scored_out["p_proto_only"] = p_sr_proto
            scored_out["p_pop_static"] = p_sr_static
            scored_out["res_proto_st"] = res_proto_st

        scored_out["res_dm80_protost20"] = 0.80 * scored_out["res_dualmod"] + 0.20 * scored_out["res_proto_st"]
        scored_out["res_dm85_protost15"] = 0.85 * scored_out["res_dualmod"] + 0.15 * scored_out["res_proto_st"]
        scored_out["res_dm70_tchr30"] = 0.70 * scored_out["res_dualmod"] + 0.30 * scored_out["res_joint"]
        self.last_res_dualmod = scored_out["res_dualmod"].copy()
        self.last_res_joint = scored_out["res_joint"].copy()
        self.last_res_proto_st = scored_out["res_proto_st"].copy()
        self.last_p_dualmod = scored_out["p_dualmod"].copy()
        return scored_out

    def observe_window(
        self,
        is_artifact_flagged: bool = False,
        teacher_probabilities: np.ndarray | None = None,
        pseudo_chrono_u: np.ndarray | None = None,
    ) -> None:
        cur_u = np.asarray(pseudo_chrono_u, dtype=np.float64) if pseudo_chrono_u is not None else self._active_pc_u
        bad_flag = bool(is_artifact_flagged)
        if not bad_flag:
            if self._pending_id_emb is not None:
                self.z_id_sum += self._pending_id_emb
            if self._pending_sr_covs is not None:
                m1_f, m1_d, car_f = self._pending_sr_covs
                self._sum_m1_f = self._sum_m1_f + m1_f
                self._sum_m1_d = self._sum_m1_d + m1_d
                self._sum_car_f = self._sum_car_f + car_f
                if self._zsr_pc_tau > 0.0 and cur_u is not None:
                    self._sr_u_hist.append(cur_u)
                    mh = self._max_pc_history
                    if self._gpu_consts is not None:
                        dev_sr = self._gpu_consts["dev"]
                        self._m1f_hist_gpu.append(
                            m1_f.detach().clone()
                            if torch.is_tensor(m1_f)
                            else torch.as_tensor(m1_f, dtype=torch.float64, device=dev_sr)
                        )
                        self._m1d_hist_gpu.append(
                            m1_d.detach().clone()
                            if torch.is_tensor(m1_d)
                            else torch.as_tensor(m1_d, dtype=torch.float64, device=dev_sr)
                        )
                        self._carf_hist_gpu.append(
                            car_f.detach().clone()
                            if torch.is_tensor(car_f)
                            else torch.as_tensor(car_f, dtype=torch.float64, device=dev_sr)
                        )
                        if len(self._sr_u_hist) > mh:
                            self._sr_u_hist.pop(0)
                            self._m1f_hist_gpu.pop(0)
                            self._m1d_hist_gpu.pop(0)
                            self._carf_hist_gpu.pop(0)
                    else:
                        m1f_np = m1_f.detach().cpu().numpy() if torch.is_tensor(m1_f) else np.asarray(m1_f, dtype=np.float64)
                        m1d_np = m1_d.detach().cpu().numpy() if torch.is_tensor(m1_d) else np.asarray(m1_d, dtype=np.float64)
                        car_np = car_f.detach().cpu().numpy() if torch.is_tensor(car_f) else np.asarray(car_f, dtype=np.float64)
                        if self._m1f_buf_np is None or self._m1f_buf_np.shape[0] != mh:
                            self._m1f_buf_np = np.empty((mh, *m1f_np.shape), dtype=np.float64)
                            self._m1d_buf_np = np.empty((mh, *m1d_np.shape), dtype=np.float64)
                            self._carf_buf_np = np.empty((mh, *car_np.shape), dtype=np.float64)
                            self._n_sr_buf = 0
                        assert self._m1d_buf_np is not None and self._carf_buf_np is not None
                        nsr = int(self._n_sr_buf)
                        if nsr < mh:
                            self._m1f_buf_np[nsr] = m1f_np
                            self._m1d_buf_np[nsr] = m1d_np
                            self._carf_buf_np[nsr] = car_np
                            self._n_sr_buf = nsr + 1
                        else:
                            self._m1f_buf_np[:-1] = self._m1f_buf_np[1:]
                            self._m1d_buf_np[:-1] = self._m1d_buf_np[1:]
                            self._carf_buf_np[:-1] = self._carf_buf_np[1:]
                            self._m1f_buf_np[-1] = m1f_np
                            self._m1d_buf_np[-1] = m1d_np
                            self._carf_buf_np[-1] = car_np
                        if len(self._sr_u_hist) > mh:
                            self._sr_u_hist.pop(0)
                self.n_ok_sr += 1
        self._pending_id_emb = None
        self._pending_sr_covs = None
        for br_spec in self._branches.values():
            self._observe_branch(br_spec, bad_flag, teacher_probabilities, cur_u)

    def step_multimodal(
        self,
        raw_window_f64: np.ndarray | None = None,
        cleaned_window_f64: np.ndarray | None = None,
        is_artifact_flagged: bool = False,
        teacher_probabilities: np.ndarray | None = None,
        all_f_causal: np.ndarray | None = None,
        all_f_pop: np.ndarray | None = None,
        n_lat: int | None = None,
        n_clean4: int = 0,
        n_rms4c: int = 0,
        s_41: np.ndarray | None = None,
        known_covs: dict[Any, np.ndarray] | None = None,
        pseudo_chrono_u: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        scored_out = self.score_window(
            raw_window_f64=raw_window_f64,
            cleaned_window_f64=cleaned_window_f64,
            all_f_causal=all_f_causal,
            all_f_pop=all_f_pop,
            n_lat=n_lat,
            n_clean4=n_clean4,
            n_rms4c=n_rms4c,
            s_41=s_41,
            known_covs=known_covs,
            pseudo_chrono_u=pseudo_chrono_u,
        )
        self.observe_window(
            is_artifact_flagged=is_artifact_flagged,
            teacher_probabilities=teacher_probabilities,
            pseudo_chrono_u=pseudo_chrono_u,
        )
        return scored_out

    def step_causal_window(self, cleaned_window: np.ndarray, is_artifact_flagged: bool) -> np.ndarray:
        if self.has_ee or self.has_pop_eegpt:
            return self.last_p_dualmod.copy()
        scored_out = self.step_multimodal(
            raw_window_f64=cleaned_window,
            cleaned_window_f64=cleaned_window,
            is_artifact_flagged=is_artifact_flagged,
        )
        return scored_out["p_dualmod"]

    @classmethod
    def load_from_npz(
        cls,
        anchor_npz_path: str | Path,
        execution_device: str | torch.device = "cpu",
        zsr_pc_tau: float = 1.5,
        rec_mode: str = "pc",
        pc_tau: float = 1.5,
        n0_glob: float = 0.0,
        n0_prior: float = 8.0,
        proto_mode: str = "cum",
        bal_mode: str = "raw_u",
        min_router_n: int = 4,
        ramp_n0: float = 6.0,
    ) -> AnchorSpecialistMultimodalOrgan:
        resolved_npz = ParameterTypeValidator.require_path(anchor_npz_path, "anchor_npz_path", must_exist=True)
        with np.load(resolved_npz, allow_pickle=False) as archive:
            weights_dict = {
                k_str: (archive[k_str].astype(np.float64) if archive[k_str].dtype == np.float16 else archive[k_str].copy())
                for k_str in archive.files
            }
        return cls(
            weights_dict=weights_dict,
            execution_device=execution_device,
            zsr_pc_tau=zsr_pc_tau,
            rec_mode=rec_mode,
            pc_tau=pc_tau,
            n0_glob=n0_glob,
            n0_prior=n0_prior,
            proto_mode=proto_mode,
            bal_mode=bal_mode,
            min_router_n=min_router_n,
            ramp_n0=ramp_n0,
        )

    @classmethod
    def fit_from_features(
        cls,
        windows: np.ndarray,
        class_labels: np.ndarray,
        subject_ids: np.ndarray,
        z_ee_causal: np.ndarray | None = None,
        z_ee_pop: np.ndarray | None = None,
        sr_dim: int = 256,
        id_dim: int = 16,
        margin_safety_offset: float = 8.0,
        ee_slice_mode: str = "lat_rms_c4",
        execution_device: str | torch.device = "cpu",
    ) -> AnchorSpecialistMultimodalOrgan:
        """Fit `ANCHOR-MM` biometric ID LDA (`tau_mrg = max_lopo_mrg + 8.0`), SpecRiem PCA, and `sr`/`ee`/`full` heads."""
        x_all = np.asarray(windows, dtype=np.float64)
        y_all = np.asarray(class_labels, dtype=np.int64)
        subj_all = np.asarray(subject_ids)
        unique_subjs = np.unique(subj_all)
        n_samples, n_chans, n_times = x_all.shape

        eye_t = np.eye(n_times, dtype=np.float64)
        m_sos_7 = np.stack([sosfiltfilt(butter(4, b, btype="bandpass", fs=120.0, output="sos"), eye_t, axis=-1) for b in cls.SPEC_BANDS_7], axis=0)
        m_sos_6 = np.stack([sosfiltfilt(butter(4, b, btype="bandpass", fs=120.0, output="sos"), eye_t, axis=-1) for b in cls.RIEM_BANDS_6], axis=0)
        m_sos_9 = np.stack([sosfiltfilt(butter(4, b, btype="bandpass", fs=120.0, output="sos"), eye_t, axis=-1) for b in cls.MORTP_BANDS_9], axis=0)

        xz_all = x_all - x_all.mean(axis=-1, keepdims=True)
        xcar_all = xz_all - xz_all.mean(axis=1, keepdims=True)
        inv_std0_id = 1.0 / np.sqrt(np.maximum((xcar_all ** 2).mean(axis=(0, 2)), 1e-6))

        eye_c = np.eye(n_chans, dtype=np.float64)
        p_car = eye_c - np.full((n_chans, n_chans), 1.0 / float(n_chans), dtype=np.float64)
        _, _, vh_car = np.linalg.svd(p_car, full_matrices=False)
        p_car40 = vh_car[: n_chans - 1]

        cb6_all = np.empty((n_samples, 6, n_chans, n_chans), dtype=np.float64)
        m1f_all = np.empty((n_samples, 6, n_chans, n_chans), dtype=np.float64)
        m1db_all = np.empty((n_samples, 3, n_chans, n_chans), dtype=np.float64)
        m1dm_all = np.empty((n_samples, 3, n_chans, n_chans), dtype=np.float64)
        carf_all = np.empty((n_samples, 5, n_chans - 1, n_chans - 1), dtype=np.float64)
        spec533_all = np.empty((n_samples, 7 * n_chans + 6 * n_chans), dtype=np.float64)

        for i_row in range(n_samples):
            xpop_i = xcar_all[i_row] * inv_std0_id[:, None]
            xf7 = np.matmul(xpop_i, m_sos_7)
            lp7 = np.log(np.maximum(np.mean(xf7 ** 2, axis=-1), 1e-6)).reshape(-1)
            ve6 = np.log(np.maximum(np.mean(xf7[:6, :, cls.SLICE_EARLY] ** 2, axis=-1), 1e-6))
            vl6 = np.log(np.maximum(np.mean(xf7[:6, :, cls.SLICE_LATE] ** 2, axis=-1), 1e-6))
            spec533_all[i_row] = np.concatenate([lp7, (vl6 - ve6).reshape(-1)], axis=0)

            xf6 = np.matmul(xz_all[i_row], m_sos_6)
            for b_i in range(6):
                cb6_all[i_row, b_i] = cls._lw_cov_single_np(xf6[b_i])

            yb9 = np.matmul(xz_all[i_row], m_sos_9)
            s_bd = np.empty((3, n_chans, n_chans), dtype=np.float64)
            s_md = np.empty((3, n_chans, n_chans), dtype=np.float64)
            for k_i, bi in enumerate(cls.MORTP_DIFF_IDX):
                sb = yb9[bi, :, 0:120] - np.mean(yb9[bi, :, 0:120], axis=-1, keepdims=True)
                s_bd[k_i] = (sb @ sb.T) / 120.0
                sm = yb9[bi, :, 180:480] - np.mean(yb9[bi, :, 180:480], axis=-1, keepdims=True)
                s_md[k_i] = (sm @ sm.T) / 300.0
            s_f6 = np.empty((6, n_chans, n_chans), dtype=np.float64)
            for k_i, bi in enumerate(cls.MORTP_FULL_IDX):
                sf = yb9[bi] - np.mean(yb9[bi], axis=-1, keepdims=True)
                s_f6[k_i] = (sf @ sf.T) / float(n_times)

            m1f_all[i_row] = cls._project_and_shrink_np(s_f6, None, cls.MORTP_SHRINK)
            m1db_all[i_row] = cls._project_and_shrink_np(s_bd, None, cls.MORTP_SHRINK)
            m1dm_all[i_row] = cls._project_and_shrink_np(s_md, None, cls.MORTP_SHRINK)
            carf_all[i_row] = cls._project_and_shrink_np(s_f6[:5], p_car40, cls.MORTP_SHRINK)

        r0_6b = cb6_all.mean(axis=0)
        w0_6b_id = cls._inv_sqrtm_np(r0_6b)
        tang_rows = np.stack(
            [cls._mat2vec_np(cls._logm_np(np.matmul(np.matmul(w0_6b_id, cb6_all[i_r]), w0_6b_id))).reshape(-1) for i_r in range(n_samples)],
            axis=0,
        )
        tg_mu = tang_rows.mean(axis=0)
        tg_sd = np.maximum(tang_rows.std(axis=0), 1e-6)
        tg_std = (tang_rows - tg_mu[None, :]) / tg_sd[None, :]
        _, _, vh_tg = np.linalg.svd(tg_std, full_matrices=False)
        k_tg = min(64, vh_tg.shape[0])
        p_tg = vh_tg[:k_tg].T

        r0_m1_f = m1f_all.mean(axis=0)
        r0_m1_d = (0.5 * (m1db_all + m1dm_all)).mean(axis=0)
        r0_car_f = carf_all.mean(axis=0)
        wm1_0 = cls._inv_sqrtm_np(r0_m1_f)
        wd_0 = cls._inv_sqrtm_np(r0_m1_d)
        wcar_0 = cls._inv_sqrtm_np(r0_car_f)

        tsr_rows = []
        for i_r in range(n_samples):
            tm1 = cls._mat2vec_np(cls._logm_np(np.matmul(np.matmul(wm1_0, m1f_all[i_r]), wm1_0)), 1.0 / np.sqrt(6.0)).reshape(-1)
            tdm = cls._mat2vec_np(cls._logm_np(np.matmul(np.matmul(wd_0, m1dm_all[i_r]), wd_0)), 1.0 / np.sqrt(3.0)).reshape(-1)
            tdb = cls._mat2vec_np(cls._logm_np(np.matmul(np.matmul(wd_0, m1db_all[i_r]), wd_0)), 1.0 / np.sqrt(3.0)).reshape(-1)
            tcr = cls._mat2vec_np(cls._logm_np(np.matmul(np.matmul(wcar_0, carf_all[i_r]), wcar_0)), 1.0 / np.sqrt(5.0)).reshape(-1)
            tsr_rows.append(np.concatenate([tm1, tdm - tdb, tcr], axis=0))
        tsr_mat = np.stack(tsr_rows, axis=0)
        sr_mu = tsr_mat.mean(axis=0)
        sr_sd = np.maximum(tsr_mat.std(axis=0), 1e-6)
        tsr_std = (tsr_mat - sr_mu[None, :]) / sr_sd[None, :]
        _, _, vh_sr = np.linalg.svd(tsr_std, full_matrices=False)
        eff_sr_d = min(int(sr_dim), vh_sr.shape[0])
        p_sr = vh_sr[:eff_sr_d].T
        z_sr_all = tsr_std @ p_sr

        id_parts = [spec533_all, tg_std @ p_tg]
        ep_mu = np.zeros(0, dtype=np.float64)
        ep_sd = np.zeros(0, dtype=np.float64)
        p_ep = np.zeros((0, 0), dtype=np.float64)
        if z_ee_pop is not None:
            zp_f64 = np.asarray(z_ee_pop, dtype=np.float64)
            ep_mu = zp_f64.mean(axis=0)
            ep_sd = np.maximum(zp_f64.std(axis=0), 1e-6)
            zp_std = (zp_f64 - ep_mu[None, :]) / ep_sd[None, :]
            _, _, vh_ep = np.linalg.svd(zp_std, full_matrices=False)
            k_ep = min(64, vh_ep.shape[0])
            p_ep = vh_ep[:k_ep].T
            id_parts.append(zp_std @ p_ep)

        z_id_cat = np.concatenate(id_parts, axis=1)
        id_mu_glob = z_id_cat.mean(axis=0)
        id_sd_glob = np.maximum(z_id_cat.std(axis=0), 1e-6)
        z_id_std = (z_id_cat - id_mu_glob[None, :]) / id_sd_glob[None, :]
        eff_id_d = min(int(id_dim), max(1, len(unique_subjs) - 1), z_id_std.shape[1])
        _, _, vh_id = np.linalg.svd(z_id_std, full_matrices=False)
        id_m_lda = vh_id[:eff_id_d].T
        emb_all = z_id_std @ id_m_lda

        emb_cen_p = np.stack([emb_all[subj_all == s_key].mean(axis=0) for s_key in unique_subjs], axis=0)
        within_res = np.concatenate([emb_all[subj_all == s_key] - emb_cen_p[idx_s][None, :] for idx_s, s_key in enumerate(unique_subjs)], axis=0)
        cov_w = (within_res.T @ within_res) / max(float(len(within_res)), 1.0) + 0.10 * np.eye(eff_id_d, dtype=np.float64)
        cov_w_inv = np.linalg.inv(cov_w)

        # Inner-LOPO calibration of `tau_mrg = max_lopo_mrg + margin_safety_offset`
        max_lopo_mrg = 0.0
        if len(unique_subjs) >= 3:
            for idx_s in range(len(unique_subjs)):
                others = np.delete(emb_cen_p, idx_s, axis=0)
                diffs_o = others - emb_cen_p[idx_s][None, :]
                d2_o = np.sort(np.einsum("ki,ij,kj->k", diffs_o, cov_w_inv, diffs_o))
                if len(d2_o) >= 2:
                    max_lopo_mrg = max(max_lopo_mrg, float(d2_o[1] - d2_o[0]))
        tau_mrg = max_lopo_mrg + float(margin_safety_offset)

        weights_out: dict[str, np.ndarray] = {
            "n_chans": np.array([n_chans], dtype=np.int64),
            "n_times": np.array([n_times], dtype=np.int64),
            "n_lat": np.array([8], dtype=np.int64),
            "n_rms": np.array([3], dtype=np.int64),
            "W0_52": cls._inv_sqrtm_np(cls._lw_cov_single_np(xz_all.transpose(1, 0, 2).reshape(n_chans, -1))),
            "inv_std0_id": inv_std0_id,
            "W0_6b_id": w0_6b_id,
            "tg_mu": tg_mu,
            "tg_sd": tg_sd,
            "P_tg": p_tg,
            "ep_mu": ep_mu,
            "ep_sd": ep_sd,
            "P_ep": p_ep,
            "id_mu_glob": id_mu_glob,
            "id_sd_glob": id_sd_glob,
            "id_M_lda": id_m_lda,
            "emb_cen_p": emb_cen_p,
            "cov_w_inv": cov_w_inv,
            "tau_mrg": np.array([tau_mrg], dtype=np.float64),
            "P_car40": p_car40,
            "R0_m1_f": r0_m1_f,
            "R0_m1_d": r0_m1_d,
            "R0_car_f": r0_car_f,
            "sr_mu": sr_mu,
            "sr_sd": sr_sd,
            "P_sr": p_sr,
            "ee_slice_mode": np.array(ee_slice_mode),
        }

        def _fit_branch_weights(branch_tag: str, z_raw_b: np.ndarray) -> None:
            mu_z = z_raw_b.mean(axis=0)
            sd_z = np.maximum(z_raw_b.std(axis=0), 1e-6)
            z_std_b = (z_raw_b - mu_z[None, :]) / sd_z[None, :]
            z_rec_b = z_std_b.copy()
            for s_key in unique_subjs:
                m_mask = subj_all == s_key
                z_rec_b[m_mask] = z_std_b[m_mask] - z_std_b[m_mask].mean(axis=0, keepdims=True)
            w_pop, b_pop = FoundationOrgan.fit_multinomial_logistic_lbfgs(z_rec_b, y_all, inverse_reg_c=0.10, pull_strength=1.0)
            w_p_list, b_p_list, mu_p_c_list = [], [], []
            mu_pop_c = np.stack(
                [z_rec_b[y_all == c_i].mean(axis=0) if np.any(y_all == c_i) else np.zeros(z_rec_b.shape[1]) for c_i in range(3)],
                axis=0,
            )
            for s_key in unique_subjs:
                m_mask = subj_all == s_key
                ws, bs = FoundationOrgan.fit_multinomial_logistic_lbfgs(
                    z_rec_b[m_mask],
                    y_all[m_mask],
                    inverse_reg_c=0.25,
                    anchor_weights=w_pop,
                    anchor_intercept=b_pop,
                    pull_strength=2.0,
                )
                w_p_list.append(ws)
                b_p_list.append(bs)
                mu_s_c = np.stack(
                    [
                        (24.0 * mu_pop_c[c_i] + z_rec_b[m_mask & (y_all == c_i)].sum(axis=0))
                        / (24.0 + float(np.sum(m_mask & (y_all == c_i))))
                        for c_i in range(3)
                    ],
                    axis=0,
                )
                mu_p_c_list.append(mu_s_c)
            cls_res = z_rec_b - mu_pop_c[y_all]
            cov_cls = (cls_res.T @ cls_res) / max(float(len(cls_res)), 1.0)
            tr_c = max(float(np.trace(cov_cls)) / float(cov_cls.shape[0]), 1e-6)
            prec_mat = np.linalg.inv(0.75 * cov_cls + 0.25 * tr_c * np.eye(cov_cls.shape[0], dtype=np.float64))

            weights_out[f"{branch_tag}__mu_z"] = mu_z
            weights_out[f"{branch_tag}__sd_z"] = sd_z
            weights_out[f"{branch_tag}__W_pop"] = w_pop
            weights_out[f"{branch_tag}__b_pop"] = b_pop
            weights_out[f"{branch_tag}__W_p"] = np.stack(w_p_list, axis=0)
            weights_out[f"{branch_tag}__b_p"] = np.stack(b_p_list, axis=0)
            weights_out[f"{branch_tag}__mu_pop_c"] = mu_pop_c
            weights_out[f"{branch_tag}__mu_p_c"] = np.stack(mu_p_c_list, axis=0)
            weights_out[f"{branch_tag}__Prec"] = prec_mat
            weights_out[f"{branch_tag}__gamma_lda"] = np.array([0.50], dtype=np.float64)

        _fit_branch_weights("sr", z_sr_all)
        if z_ee_causal is not None:
            z_ee_f64 = np.asarray(z_ee_causal, dtype=np.float64)
            _fit_branch_weights("ee", z_ee_f64)
            z_full = np.concatenate(
                [z_ee_f64 / np.sqrt(float(z_ee_f64.shape[1])), z_sr_all / np.sqrt(float(z_sr_all.shape[1]))], axis=1
            )
            _fit_branch_weights("full", z_full)

        return cls(weights_dict=weights_out, execution_device=execution_device)

    def save_organ_artifacts(self, target_directory: Path) -> list[Path]:
        resolved_dir = ParameterTypeValidator.require_path(target_directory, "target_directory")
        resolved_dir.mkdir(parents=True, exist_ok=True)
        fp64_protected_prefixes = ("R0_", "W0_", "id_", "cov_w_inv", "emb_cen_p", "tau_mrg", "inv_std0_id", "P_car40")
        fp16_projection_keys = {"P_sr", "P_tg", "P_ep", "P_c4_pop"}
        compressed_payload: dict[str, np.ndarray] = {}
        for key_str, arr_raw in self._weights_dict.items():
            arr_np = np.asarray(arr_raw)
            if not np.issubdtype(arr_np.dtype, np.floating):
                compressed_payload[key_str] = arr_np
            elif key_str.startswith(fp64_protected_prefixes):
                compressed_payload[key_str] = arr_np.astype(np.float64)
            elif key_str in fp16_projection_keys or arr_np.nbytes >= 200_000:
                compressed_payload[key_str] = arr_np.astype(np.float16)
            else:
                compressed_payload[key_str] = arr_np.astype(np.float32)
        out_path = resolved_dir / "anchor_mm.npz"
        np.savez_compressed(out_path, **compressed_payload)
        return [out_path]
