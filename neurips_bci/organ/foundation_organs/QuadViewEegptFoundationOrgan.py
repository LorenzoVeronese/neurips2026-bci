"""Quad-view EEGPT causal streaming foundation organ (`lat52` + `clean4` + `rms4c` + `rms56`)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from neurips_bci.backbone.foundation_backbones.BatchedEegptFoundationBackbone import (
    BatchedEegptFoundationBackbone,
)
from neurips_bci.operator.matrix_operators.CovarianceMatrixOperator import CovarianceMatrixOperator
from neurips_bci.operator.signal_operators.SpectralSignalOperator import SpectralSignalOperator
from neurips_bci.organ.foundation_organs.FoundationOrgan import FoundationOrgan
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class QuadViewEegptFoundationOrgan(FoundationOrgan):
    """Causal streaming quad-view EEGPT bank (`ee7_pcearms_pcrec_t58`) with Newton-Cholesky head refit."""

    def __init__(
        self,
        batched_backbone: BatchedEegptFoundationBackbone,
        lat_mus: Sequence[np.ndarray],
        lat_reps: Sequence[tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]],
        rms_mus: Sequence[np.ndarray],
        rms_reps: Sequence[tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]],
        r0_52_broad: np.ndarray,
        r0_var: np.ndarray,
        execution_device: str | torch.device = "cpu",
        ea_prior: float = 8.0,
        recenter_n0: float = 10.0,
        refit_at: Sequence[int] = (66, 126),
        clean4_mus: Sequence[np.ndarray] | None = None,
        clean4_reps: Sequence[tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]] | None = None,
        alpha_clean4: float = 0.35,
        rms4c_mus: Sequence[np.ndarray] | None = None,
        rms4c_reps: Sequence[tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]] | None = None,
        alpha_rms4c: float = 0.65,
        eegpt_mode: str = "ee7_pcearms_pcrec_t58",
        rms4c_pc_recenter: bool = False,
        max_pc_history: int = 512,
        base_fp16_tensors: dict[str, np.ndarray] | None = None,
        seed_npz_payloads: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> None:
        super().__init__(
            organ_identifier="quad_view_eegpt_foundation_organ",
            backbone=batched_backbone,
            execution_device=execution_device,
        )
        self._batched_backbone = batched_backbone
        self._lat_mus = [np.asarray(mu_item, dtype=np.float64) for mu_item in lat_mus]
        self._lat_reps = list(lat_reps)
        self._clean4_mus = [np.asarray(mu_item, dtype=np.float64) for mu_item in (clean4_mus or ())]
        self._clean4_reps = list(clean4_reps or ())
        self._rms4c_mus = [np.asarray(mu_item, dtype=np.float64) for mu_item in (rms4c_mus or ())]
        self._rms4c_reps = list(rms4c_reps or ())
        self._rms_mus = [np.asarray(mu_item, dtype=np.float64) for mu_item in rms_mus]
        self._rms_reps = list(rms_reps)

        self._alpha_clean4 = float(alpha_clean4)
        self._alpha_rms4c = float(alpha_rms4c)
        self._r0_52_broad = np.asarray(r0_52_broad, dtype=np.float64)
        self._r0_var = np.asarray(r0_var, dtype=np.float64)
        self._w0_52 = CovarianceMatrixOperator.inverse_matrix_square_root(self._r0_52_broad)
        self._inv_std0_rms = 1.0 / np.sqrt(np.maximum(self._r0_var, 1e-6))
        self._fir52 = SpectralSignalOperator.design_bandpass_fir((4.0, 52.0), tap_count=121, sampling_rate_hz=120.0)
        self._ea_prior = float(ea_prior)
        self._recenter_n0 = float(recenter_n0)
        self._refit_at = tuple(int(step_idx) for step_idx in refit_at)
        self._eegpt_mode = str(eegpt_mode)
        self._rms4c_pc_recenter = bool(rms4c_pc_recenter)
        self._max_pc_history = ParameterTypeValidator.require_integer(max_pc_history, "max_pc_history", 8)
        self._base_fp16_tensors = base_fp16_tensors
        self._seed_npz_payloads = seed_npz_payloads

        self._gpu_refit_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        combined_reps = self._lat_reps + self._clean4_reps + self._rms4c_reps + self._rms_reps
        if self._execution_device.type == "cuda" and len(combined_reps) > 0:
            self._gpu_refit_cache = (
                torch.as_tensor(
                    np.stack([rep_item[0] for rep_item in combined_reps], axis=0),
                    dtype=torch.float64,
                    device=self._execution_device,
                ),
                torch.as_tensor(
                    np.stack([rep_item[1] for rep_item in combined_reps], axis=0),
                    dtype=torch.long,
                    device=self._execution_device,
                ),
                torch.as_tensor(
                    np.stack([rep_item[2][0] for rep_item in combined_reps], axis=0),
                    dtype=torch.float64,
                    device=self._execution_device,
                ),
                torch.as_tensor(
                    np.stack([rep_item[2][1] for rep_item in combined_reps], axis=0),
                    dtype=torch.float64,
                    device=self._execution_device,
                ),
            )

        self._pin52_1: object = None
        self._pin56_1: object = None
        self._pin52_2: object = None
        self._pin56_2: object = None
        self._active_pc_u: np.ndarray | None = None
        self.reset_organ_state()

    @property
    def lat_seed_count(self) -> int:
        return len(self._lat_mus)

    @property
    def clean4_seed_count(self) -> int:
        return len(self._clean4_mus)

    @property
    def rms4c_seed_count(self) -> int:
        return len(self._rms4c_mus)

    @property
    def rms_seed_count(self) -> int:
        return len(self._rms_mus)

    @property
    def fir52(self) -> np.ndarray:
        return self._fir52

    @property
    def w0_52(self) -> np.ndarray:
        return self._w0_52

    @property
    def eegpt_mode(self) -> str:
        return self._eegpt_mode

    def set_eegpt_mode(self, mode_label: str, rms4c_pc_recenter: bool = False) -> None:
        self._eegpt_mode = str(mode_label)
        self._rms4c_pc_recenter = bool(rms4c_pc_recenter)

    def set_pseudo_chrono_vector(self, current_u_vector: np.ndarray | None, max_history: int | None = None) -> None:
        self._active_pc_u = None if current_u_vector is None else np.asarray(current_u_vector, dtype=np.float64)
        if max_history is not None:
            self._max_pc_history = int(max_history)

    def reset_organ_state(self) -> None:
        self._step_index = 0
        self._ok_hist: list[bool] = []
        self._tchr_hist: list[np.ndarray] = []
        self._tchr_hist_b: list[np.ndarray] = []

        self._cov52_sum = np.zeros_like(self._r0_52_broad, dtype=np.float64)
        self._n_ok = 0
        self._w52_cur = self._w0_52.copy()

        self._var_rms_sum = np.zeros_like(self._r0_var, dtype=np.float64)
        self._inv_std_cur = self._inv_std0_rms.copy()

        self._lat_f_sums = [np.zeros_like(mu_v, dtype=np.float64) for mu_v in self._lat_mus]
        self._lat_f_hist: list[list[np.ndarray]] = [[] for _ in self._lat_mus]
        self._lat_w = [rep_item[2][0].copy() for rep_item in self._lat_reps]
        self._lat_b = [rep_item[2][1].copy() for rep_item in self._lat_reps]
        self._lat_w_b = [rep_item[2][0].copy() for rep_item in self._lat_reps]
        self._lat_b_b = [rep_item[2][1].copy() for rep_item in self._lat_reps]

        self._c4_f_sums = [np.zeros_like(mu_v, dtype=np.float64) for mu_v in self._clean4_mus]
        self._c4_f_hist: list[list[np.ndarray]] = [[] for _ in self._clean4_mus]
        self._c4_w = [rep_item[2][0].copy() for rep_item in self._clean4_reps]
        self._c4_b = [rep_item[2][1].copy() for rep_item in self._clean4_reps]
        self._c4_w_b = [rep_item[2][0].copy() for rep_item in self._clean4_reps]
        self._c4_b_b = [rep_item[2][1].copy() for rep_item in self._clean4_reps]

        self._r4c_f_sums = [np.zeros_like(mu_v, dtype=np.float64) for mu_v in self._rms4c_mus]
        self._r4c_f_hist: list[list[np.ndarray]] = [[] for _ in self._rms4c_mus]
        self._r4c_w = [rep_item[2][0].copy() for rep_item in self._rms4c_reps]
        self._r4c_b = [rep_item[2][1].copy() for rep_item in self._rms4c_reps]
        self._r4c_w_b = [rep_item[2][0].copy() for rep_item in self._rms4c_reps]
        self._r4c_b_b = [rep_item[2][1].copy() for rep_item in self._rms4c_reps]

        self._rms_f_sums = [np.zeros_like(mu_v, dtype=np.float64) for mu_v in self._rms_mus]
        self._rms_f_hist: list[list[np.ndarray]] = [[] for _ in self._rms_mus]
        self._rms_w = [rep_item[2][0].copy() for rep_item in self._rms_reps]
        self._rms_b = [rep_item[2][1].copy() for rep_item in self._rms_reps]
        self._rms_w_b = [rep_item[2][0].copy() for rep_item in self._rms_reps]
        self._rms_b_b = [rep_item[2][1].copy() for rep_item in self._rms_reps]

        uniform_3 = np.full(3, 1.0 / 3.0, dtype=np.float64)
        self.last_u_lat_a = uniform_3.copy()
        self.last_u_rms_a = uniform_3.copy()
        self.last_u_lat_b = uniform_3.copy()
        self.last_u_rms_b = uniform_3.copy()
        self.last_u_rms_c4_a = uniform_3.copy()
        self.last_u_rms_c4_b = uniform_3.copy()
        self.last_u_rms_leg_a = uniform_3.copy()
        self.last_u_rms_leg_b = uniform_3.copy()
        self.last_u_rms_prot_a = uniform_3.copy()
        self.last_all_f_causal: np.ndarray | None = None
        self.last_all_f_pop: np.ndarray | None = None

        self._pending_eegpt: tuple[bool, bool, torch.Tensor, np.ndarray | None] | None = None
        self._active_pc_u = None
        self._ee_u_hist: list[np.ndarray] = []
        self._ee_cov52_hist: list[np.ndarray] = []
        self._ee_var_hist: list[np.ndarray] = []
        self._ee_lat_raw: list[np.ndarray] = []
        self._ee_rms_raw: list[np.ndarray] = []
        self._ee_r4c_raw: list[np.ndarray] = []

    @staticmethod
    def _pc_rbf_weights(u_cur: np.ndarray, u_hist: list[np.ndarray], tau: float) -> np.ndarray:
        if not u_hist or tau <= 0.0:
            return np.zeros(0, dtype=np.float64)
        u_mat = np.stack(u_hist, axis=0)
        sq_dist = np.sum((u_mat - u_cur[None, :]) ** 2, axis=1)
        inv_2dt2 = 0.5 / (float(u_cur.shape[0]) * (tau * tau))
        return np.exp(-inv_2dt2 * sq_dist)

    @staticmethod
    def _refit_newton_mixed_cuda(
        w0_batch: torch.Tensor,
        b0_batch: torch.Tensor,
        f_all: torch.Tensor,
        y_all: torch.Tensor,
        wt_all: torch.Tensor,
        inverse_c: float = 1.0,
        pull_strength: float = 3.0,
        max_iter: int = 16,
        grad_tol: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_cnt, class_cnt, feat_dim = w0_batch.shape
        sample_cnt = f_all.shape[1]
        dp1 = feat_dim + 1
        param_dim = class_cnt * dp1
        reg_lam = float(pull_strength) / float(inverse_c)
        dev = w0_batch.device
        one_hot_y = F.one_hot(y_all, num_classes=class_cnt).to(dtype=torch.float64)
        wt_mat = wt_all.expand(model_cnt, sample_cnt)
        wt_col = wt_mat.unsqueeze(-1)
        wt_y = wt_col * one_hot_y

        ones_col = torch.ones((model_cnt, sample_cnt, 1), dtype=torch.float64, device=dev)
        x_aug = torch.cat([f_all, ones_col], dim=2)
        x_aug_t = x_aug.transpose(1, 2).contiguous()

        theta = torch.cat([w0_batch, b0_batch.unsqueeze(2)], dim=2).clone()
        reg_diag = torch.full((dp1,), float(reg_lam), dtype=torch.float64, device=dev)
        reg_diag[feat_dim] = 0.0
        idx_diag = torch.arange(dp1, device=dev)

        def _eval_loss_grad(th_cur: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            logits = torch.bmm(x_aug, th_cur.transpose(1, 2))
            logits = logits - logits.amax(dim=-1, keepdim=True)
            probs = torch.softmax(logits, dim=-1)
            lse = torch.log(torch.exp(logits).sum(dim=-1, keepdim=True))
            log_p = logits - lse
            diff_w = th_cur[:, :, :feat_dim] - w0_batch
            loss_vec = -(wt_y * log_p).sum(dim=(1, 2)) + 0.5 * reg_lam * (diff_w * diff_w).sum(dim=(1, 2))
            grad_z = (probs - one_hot_y) * wt_col
            grad_th = torch.bmm(grad_z.transpose(1, 2), x_aug)
            grad_th[:, :, :feat_dim] = grad_th[:, :, :feat_dim] + reg_lam * diff_w
            return loss_vec, grad_th, probs

        hess_f64 = torch.empty((model_cnt, class_cnt, dp1, class_cnt, dp1), dtype=torch.float64, device=dev)
        loss_vec, grad_th, probs = _eval_loss_grad(theta)
        for _ in range(max_iter):
            if float(grad_th.abs().max().item()) <= grad_tol:
                break
            hess_f64.zero_()
            for k_cls in range(class_cnt):
                pk = probs[:, :, k_cls]
                for l_cls in range(k_cls, class_cnt):
                    w_kl = (wt_mat * pk * (1.0 - pk)) if (k_cls == l_cls) else (-wt_mat * pk * probs[:, :, l_cls])
                    blk = torch.bmm(x_aug_t, x_aug * w_kl.unsqueeze(-1))
                    if k_cls == l_cls:
                        blk[:, idx_diag, idx_diag] += reg_diag
                    hess_f64[:, k_cls, :, l_cls, :] = blk
                    if k_cls != l_cls:
                        hess_f64[:, l_cls, :, k_cls, :] = blk
            hess_f64[:, :, feat_dim, :, feat_dim] += 1e-6
            h_mat = hess_f64.reshape(model_cnt, param_dim, param_dim)
            g_vec = grad_th.reshape(model_cnt, param_dim, 1)
            l_chol, chol_info = torch.linalg.cholesky_ex(h_mat)
            if bool((chol_info != 0).any()):
                eye_p = torch.eye(param_dim, dtype=torch.float64, device=dev).unsqueeze(0)
                l_chol = torch.linalg.cholesky(h_mat + 1e-4 * eye_p)
            delta = -torch.cholesky_solve(g_vec, l_chol).reshape(model_cnt, class_cnt, dp1)
            delta[:, :, feat_dim] = delta[:, :, feat_dim] - delta[:, :, feat_dim].mean(dim=1, keepdim=True)
            gtd = (grad_th * delta).sum(dim=(1, 2))
            bad_d = gtd >= 0
            if bool(bad_d.any()):
                delta = torch.where(bad_d.view(model_cnt, 1, 1), -grad_th, delta)
                gtd = (grad_th * delta).sum(dim=(1, 2))
            step_sz = torch.ones((model_cnt,), dtype=torch.float64, device=dev)
            loss_try, grad_try, probs_try = _eval_loss_grad(theta + step_sz.view(model_cnt, 1, 1) * delta)
            for _ in range(15):
                bad_step = (~torch.isfinite(loss_try)) | (loss_try > loss_vec + 1e-4 * step_sz * gtd)
                if not bool(bad_step.any()):
                    break
                step_sz = torch.where(bad_step, step_sz * 0.5, step_sz)
                loss_try, grad_try, probs_try = _eval_loss_grad(theta + step_sz.view(model_cnt, 1, 1) * delta)
            theta = theta + step_sz.view(model_cnt, 1, 1) * delta
            loss_vec, grad_th, probs = loss_try, grad_try, probs_try
        if float(grad_th.abs().max().item()) >= 1e-2:
            raise RuntimeError("newton_grad")
        return theta[:, :, :feat_dim], theta[:, :, feat_dim]

    def _refit_head_set(
        self,
        tchr_hist: list[np.ndarray],
        lat_w: list[np.ndarray],
        lat_b: list[np.ndarray],
        c4_w: list[np.ndarray],
        c4_b: list[np.ndarray],
        r4c_w: list[np.ndarray],
        r4c_b: list[np.ndarray],
        rms_w: list[np.ndarray],
        rms_b: list[np.ndarray],
    ) -> None:
        cur_t = self._step_index
        if len(tchr_hist) < cur_t:
            return
        p_past = np.stack(tchr_hist[:cur_t], axis=0)
        ok_past = np.asarray(self._ok_hist[:cur_t], dtype=bool)
        conf_tc = p_past.max(axis=1)
        cls_tc = p_past.argmax(axis=1)
        keep_tc = np.flatnonzero((conf_tc >= 0.55) & ok_past)
        if len(keep_tc) < 15 or len(np.unique(cls_tc[keep_tc])) < 2:
            return
        w_sess = 2.5 * (conf_tc[keep_tc] - 0.33)
        y_sess = cls_tc[keep_tc]

        if self._execution_device.type == "cuda" and self._gpu_refit_cache is not None:
            try:
                f_src_g, y_src_g, w0_g, b0_g = self._gpu_refit_cache
                n_lat = len(self._lat_reps)
                n_c4 = len(self._clean4_reps)
                n_r4c = len(self._rms4c_reps)
                n_rms = len(self._rms_reps)
                n_tot = n_lat + n_c4 + n_r4c + n_rms
                f_sess_list = (
                    [np.stack([self._lat_f_hist[s_i][k_i] for k_i in keep_tc], axis=0) for s_i in range(n_lat)]
                    + [np.stack([self._c4_f_hist[s_i][k_i] for k_i in keep_tc], axis=0) for s_i in range(n_c4)]
                    + [np.stack([self._r4c_f_hist[s_i][k_i] for k_i in keep_tc], axis=0) for s_i in range(n_r4c)]
                    + [np.stack([self._rms_f_hist[s_i][k_i] for k_i in keep_tc], axis=0) for s_i in range(n_rms)]
                )
                with torch.no_grad():
                    f_sess_g = torch.as_tensor(np.stack(f_sess_list, axis=0), dtype=torch.float64, device=self._execution_device)
                    f_all_g = torch.cat([f_src_g, f_sess_g], dim=1)
                    y_sess_g = torch.as_tensor(y_sess, dtype=torch.long, device=self._execution_device).unsqueeze(0).expand(n_tot, -1)
                    y_all_g = torch.cat([y_src_g, y_sess_g], dim=1)
                    wt_all_g = torch.cat(
                        [
                            torch.ones(f_src_g.shape[1], dtype=torch.float64, device=self._execution_device),
                            torch.as_tensor(w_sess, dtype=torch.float64, device=self._execution_device),
                        ],
                        dim=0,
                    ).unsqueeze(0)
                    wg_out, bg_out = self._refit_newton_mixed_cuda(w0_g, b0_g, f_all_g, y_all_g, wt_all_g, 1.0, 3.0)
                    wg_np, bg_np = wg_out.cpu().numpy(), bg_out.cpu().numpy()
                for s_i in range(n_lat):
                    lat_w[s_i], lat_b[s_i] = wg_np[s_i].copy(), bg_np[s_i].copy()
                for s_i in range(n_c4):
                    c4_w[s_i], c4_b[s_i] = wg_np[n_lat + s_i].copy(), bg_np[n_lat + s_i].copy()
                for s_i in range(n_r4c):
                    r4c_w[s_i], r4c_b[s_i] = wg_np[n_lat + n_c4 + s_i].copy(), bg_np[n_lat + n_c4 + s_i].copy()
                for s_i in range(n_rms):
                    rms_w[s_i], rms_b[s_i] = (
                        wg_np[n_lat + n_c4 + n_r4c + s_i].copy(),
                        bg_np[n_lat + n_c4 + n_r4c + s_i].copy(),
                    )
                return
            except Exception:
                pass

        for stream_reps, stream_hist, stream_w, stream_b in (
            (self._lat_reps, self._lat_f_hist, lat_w, lat_b),
            (self._clean4_reps, self._c4_f_hist, c4_w, c4_b),
            (self._rms4c_reps, self._r4c_f_hist, r4c_w, r4c_b),
            (self._rms_reps, self._rms_f_hist, rms_w, rms_b),
        ):
            for s_i, rep_item in enumerate(stream_reps):
                f_src, y_src, (w0_s, b0_s) = rep_item
                f_sess = np.stack([stream_hist[s_i][k_i] for k_i in keep_tc], axis=0)
                x_fit = np.concatenate([f_src, f_sess], axis=0)
                y_fit = np.concatenate([y_src, y_sess], axis=0)
                w_fit = np.concatenate([np.ones(len(y_src), dtype=np.float64), w_sess], axis=0)
                stream_w[s_i], stream_b[s_i] = self.fit_multinomial_logistic_lbfgs(
                    x_fit,
                    y_fit,
                    inverse_reg_c=1.0,
                    sample_weights=w_fit,
                    anchor_weights=w0_s,
                    anchor_intercept=b0_s,
                    pull_strength=3.0,
                )

    def _maybe_refit_heads(self) -> None:
        if self._step_index not in self._refit_at:
            return
        if self._eegpt_mode == "ee7_pcearms_pcrec_t58":
            n_past = min(len(self._tchr_hist), len(self._ok_hist))
            if n_past < 8:
                return
            t_arr = np.stack(self._tchr_hist[:n_past], axis=0).astype(np.float64)
            ok_p = np.asarray(self._ok_hist[:n_past], dtype=bool)
            t_use = t_arr[ok_p] if np.any(ok_p) else t_arr
            if float(np.mean(np.max(t_use, axis=-1))) < 0.58:
                return
            self._refit_head_set(
                self._tchr_hist,
                self._lat_w,
                self._lat_b,
                self._c4_w,
                self._c4_b,
                self._r4c_w,
                self._r4c_b,
                self._rms_w,
                self._rms_b,
            )
            self._lat_w_b = [w_m.copy() for w_m in self._lat_w]
            self._lat_b_b = [b_v.copy() for b_v in self._lat_b]
            self._c4_w_b = [w_m.copy() for w_m in self._c4_w]
            self._c4_b_b = [b_v.copy() for b_v in self._c4_b]
            self._r4c_w_b = [w_m.copy() for w_m in self._r4c_w]
            self._r4c_b_b = [b_v.copy() for b_v in self._r4c_b]
            self._rms_w_b = [w_m.copy() for w_m in self._rms_w]
            self._rms_b_b = [b_v.copy() for b_v in self._rms_b]
            return

        self._refit_head_set(
            self._tchr_hist,
            self._lat_w,
            self._lat_b,
            self._c4_w,
            self._c4_b,
            self._r4c_w,
            self._r4c_b,
            self._rms_w,
            self._rms_b,
        )
        if len(self._tchr_hist_b) >= self._step_index:
            self._refit_head_set(
                self._tchr_hist_b,
                self._lat_w_b,
                self._lat_b_b,
                self._c4_w_b,
                self._c4_b_b,
                self._r4c_w_b,
                self._r4c_b_b,
                self._rms_w_b,
                self._rms_b_b,
            )

    def _stage_pair_async(self, first_win: np.ndarray, second_win: np.ndarray, pin_attr: str) -> torch.Tensor:
        stacked_pair = np.stack([first_win, second_win], axis=0)
        if self._execution_device.type == "cuda" and stacked_pair.shape == (2, 41, 480):
            pin_buf = getattr(self, pin_attr, None)
            if pin_buf is None:
                try:
                    pin_buf = torch.empty((2, 41, 480), dtype=torch.float32, pin_memory=True)
                except Exception:
                    pin_buf = False
                setattr(self, pin_attr, pin_buf)
            if pin_buf is not False:
                pin_buf.copy_(torch.from_numpy(stacked_pair))
                return pin_buf.to(self._execution_device, non_blocking=True)
        return torch.as_tensor(stacked_pair, dtype=torch.float32, device=self._execution_device)

    def _stage_single_async(self, single_win: np.ndarray, pin_attr: str) -> torch.Tensor:
        if self._execution_device.type == "cuda" and single_win.shape == (41, 480):
            pin_buf = getattr(self, pin_attr, None)
            if pin_buf is None:
                try:
                    pin_buf = torch.empty((1, 41, 480), dtype=torch.float32, pin_memory=True)
                except Exception:
                    pin_buf = False
                setattr(self, pin_attr, pin_buf)
            if pin_buf is not False:
                pin_buf[0].copy_(torch.from_numpy(np.ascontiguousarray(single_win, dtype=np.float32)))
                return pin_buf.to(self._execution_device, non_blocking=True)
        return torch.as_tensor(single_win[None], dtype=torch.float32, device=self._execution_device)

    def launch_gpu_async(
        self,
        cleaned_window: np.ndarray,
        is_artifact_flagged: bool,
        pre_centered_window: np.ndarray | None = None,
        pop_inv_std0_id: np.ndarray | None = None,
        pseudo_chrono_u: np.ndarray | None = None,
    ) -> None:
        self._maybe_refit_heads()
        is_clean = not bool(is_artifact_flagged)
        self._ok_hist.append(is_clean)

        cur_u = (
            np.asarray(pseudo_chrono_u, dtype=np.float64)
            if pseudo_chrono_u is not None
            else self._active_pc_u
        )
        use_ee7 = self._eegpt_mode == "ee7_pcearms_pcrec_t58" and cur_u is not None

        x_cent = (
            pre_centered_window
            if pre_centered_window is not None
            else (cleaned_window - cleaned_window.mean(axis=-1, keepdims=True))
        )
        xc52 = SpectralSignalOperator.apply_zero_phase_fir(x_cent[None], self._fir52)[0]
        x_car = x_cent - x_cent.mean(axis=0, keepdims=True)

        if is_clean:
            cov52 = CovarianceMatrixOperator.ledoit_wolf_covariance(xc52[None])[0]
            v_cur = (x_car ** 2).mean(axis=-1)
            self._cov52_sum += cov52
            self._var_rms_sum += v_cur
            self._n_ok += 1
            if use_ee7 and cur_u is not None:
                self._ee_u_hist.append(cur_u)
                self._ee_cov52_hist.append(cov52)
                self._ee_var_hist.append(v_cur)
                if len(self._ee_u_hist) > self._max_pc_history:
                    self._ee_u_hist.pop(0)
                    self._ee_cov52_hist.pop(0)
                    self._ee_var_hist.pop(0)

        if use_ee7 and cur_u is not None:
            w_ee = self._pc_rbf_weights(cur_u, self._ee_u_hist, 1.50)
            w_sum = float(w_ee.sum())
            c_ker = (
                np.tensordot(w_ee, np.stack(self._ee_cov52_hist, axis=0), axes=(0, 0))
                if self._ee_cov52_hist
                else np.zeros_like(self._r0_52_broad)
            )
            v_ker = (w_ee @ np.stack(self._ee_var_hist, axis=0)) if self._ee_var_hist else np.zeros_like(self._r0_var)
            c_glob = (self._cov52_sum / float(self._n_ok)) if self._n_ok > 0 else self._r0_52_broad
            v_glob = (self._var_rms_sum / float(self._n_ok)) if self._n_ok > 0 else self._r0_var
            den_ee = 6.0 + 2.0 + w_sum
            self._w52_cur = CovarianceMatrixOperator.inverse_matrix_square_root(
                (6.0 * self._r0_52_broad + 2.0 * c_glob + c_ker) / den_ee
            )
            vt_cur = (6.0 * self._r0_var + 2.0 * v_glob + v_ker) / den_ee
            self._inv_std_cur = 1.0 / np.sqrt(np.maximum(vt_cur, 1e-6))
        elif is_clean:
            denom_ea = self._ea_prior + self._n_ok
            self._w52_cur = CovarianceMatrixOperator.inverse_matrix_square_root(
                (self._ea_prior * self._r0_52_broad + self._cov52_sum) / denom_ea
            )
            vt_cur = (self._ea_prior * self._r0_var + self._var_rms_sum) / denom_ea
            self._inv_std_cur = 1.0 / np.sqrt(np.maximum(vt_cur, 1e-6))

        xw52 = (self._w52_cur @ xc52).astype(np.float32)
        xw_rms = (x_car * self._inv_std_cur[:, None]).astype(np.float32)

        with torch.no_grad():
            if pop_inv_std0_id is not None:
                xw52_pop = (self._w0_52 @ xc52).astype(np.float32)
                xw56_pop = (x_car * np.asarray(pop_inv_std0_id, dtype=np.float64)[:, None]).astype(np.float32)
                xt52_2 = self._stage_pair_async(xw52, xw52_pop, "_pin52_2")
                xt56_2 = self._stage_pair_async(xw_rms, xw56_pop, "_pin56_2")
                x256_52_2 = BatchedEegptFoundationBackbone.fft_resample_common_average(xt52_2, 1024, lowpass_hz=52.0)
                x256_56_2 = BatchedEegptFoundationBackbone.fft_resample_common_average(xt56_2, 1024, lowpass_hz=56.0)
                out_tensor = self._batched_backbone.forward_dual_window(x256_52_2, x256_56_2).float()
                self._pending_eegpt = (is_clean, True, out_tensor, cur_u)
            else:
                xt52 = self._stage_single_async(xw52, "_pin52_1")
                xt_rms = self._stage_single_async(xw_rms, "_pin56_1")
                x256_52 = BatchedEegptFoundationBackbone.fft_resample_common_average(xt52, 1024, lowpass_hz=52.0)
                x256_56 = BatchedEegptFoundationBackbone.fft_resample_common_average(xt_rms, 1024, lowpass_hz=56.0)
                out_tensor = self._batched_backbone.forward_single_window(x256_52, x256_56).float()
                self._pending_eegpt = (is_clean, False, out_tensor, cur_u)

    def step_quad_views(
        self,
        cleaned_window: np.ndarray,
        is_artifact_flagged: bool,
        pre_centered_window: np.ndarray | None = None,
        pop_inv_std0_id: np.ndarray | None = None,
        pseudo_chrono_u: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._pending_eegpt is None:
            self.launch_gpu_async(
                cleaned_window,
                is_artifact_flagged=is_artifact_flagged,
                pre_centered_window=pre_centered_window,
                pop_inv_std0_id=pop_inv_std0_id,
                pseudo_chrono_u=pseudo_chrono_u,
            )
        assert self._pending_eegpt is not None
        is_clean, has_pop, out_tensor, cur_u = self._pending_eegpt
        self._pending_eegpt = None
        arr_f = out_tensor.detach().cpu().numpy().astype(np.float64)
        if has_pop:
            all_f = arr_f[:, 0, :]
            self.last_all_f_causal = all_f
            self.last_all_f_pop = arr_f[:, 1, :]
        else:
            all_f = arr_f
            self.last_all_f_causal = all_f
            self.last_all_f_pop = None

        n_lat_cur = len(self._lat_mus)
        n_c4_cur = len(self._clean4_mus)
        n_r4c_cur = len(self._rms4c_mus)
        lat_feats = all_f[:n_lat_cur]
        c4_feats = all_f[n_lat_cur : n_lat_cur + n_c4_cur]
        r4c_feats = all_f[n_lat_cur + n_c4_cur : n_lat_cur + n_c4_cur + n_r4c_cur]
        rms_feats = all_f[n_lat_cur + n_c4_cur + n_r4c_cur :]

        n_good = self._n_ok
        denom_rec = self._recenter_n0 + n_good
        use_pcrec = self._eegpt_mode == "ee7_pcearms_pcrec_t58" and cur_u is not None
        use_r4c_pcr = self._rms4c_pc_recenter and use_pcrec

        lat_ker_all: np.ndarray | None = None
        rms_ker_all: np.ndarray | None = None
        r4c_ker_all: np.ndarray | None = None
        den_pcr = 10.0

        if use_pcrec and cur_u is not None:
            if is_clean:
                self._ee_lat_raw.append(lat_feats.copy())
                if len(rms_feats) > 0:
                    self._ee_rms_raw.append(rms_feats.copy())
                if use_r4c_pcr and len(r4c_feats) > 0:
                    self._ee_r4c_raw.append(r4c_feats.copy())
                    if len(self._ee_r4c_raw) > self._max_pc_history:
                        self._ee_r4c_raw.pop(0)
                if len(self._ee_lat_raw) > self._max_pc_history:
                    self._ee_lat_raw.pop(0)
                if len(self._ee_rms_raw) > self._max_pc_history:
                    self._ee_rms_raw.pop(0)
            w_pcr = self._pc_rbf_weights(cur_u, self._ee_u_hist, 1.50)
            den_pcr = 8.0 + 2.0 + float(w_pcr.sum())
            if self._ee_lat_raw and len(self._ee_lat_raw) == len(w_pcr):
                lat_ker_all = np.tensordot(w_pcr, np.stack(self._ee_lat_raw, axis=0), axes=(0, 0))
            if self._ee_rms_raw and len(self._ee_rms_raw) == len(w_pcr):
                rms_ker_all = np.tensordot(w_pcr, np.stack(self._ee_rms_raw, axis=0), axes=(0, 0))
            if use_r4c_pcr and self._ee_r4c_raw and len(self._ee_r4c_raw) == len(w_pcr):
                r4c_ker_all = np.tensordot(w_pcr, np.stack(self._ee_r4c_raw, axis=0), axes=(0, 0))

        lp_lat = np.zeros(3, dtype=np.float64)
        lp_lat_b = np.zeros(3, dtype=np.float64)
        for s_i, f_cur in enumerate(lat_feats):
            mu_tr = self._lat_mus[s_i]
            if is_clean:
                self._lat_f_sums[s_i] += f_cur
            if use_pcrec:
                lg_s = (self._lat_f_sums[s_i] / float(n_good)) if n_good > 0 else mu_tr
                lk_s = lat_ker_all[s_i] if lat_ker_all is not None else np.zeros_like(mu_tr)
                mean_cur = (8.0 * mu_tr + 2.0 * lg_s + lk_s) / den_pcr
            else:
                mean_cur = (self._recenter_n0 * mu_tr + self._lat_f_sums[s_i]) / denom_rec
            f_rec = f_cur - mean_cur + mu_tr
            self._lat_f_hist[s_i].append(f_rec)
            pr_a = CovarianceMatrixOperator.stable_softmax((f_rec @ self._lat_w[s_i].T + self._lat_b[s_i])[None])[0]
            lp_lat += np.log(np.clip(pr_a, 1e-12, 1.0)) / max(1, n_lat_cur)
            pr_b = CovarianceMatrixOperator.stable_softmax((f_rec @ self._lat_w_b[s_i].T + self._lat_b_b[s_i])[None])[0]
            lp_lat_b += np.log(np.clip(pr_b, 1e-12, 1.0)) / max(1, n_lat_cur)
        u_lat = CovarianceMatrixOperator.stable_softmax(lp_lat[None])[0] if n_lat_cur > 0 else np.full(3, 1.0 / 3.0)
        self.last_u_lat_b = (
            CovarianceMatrixOperator.stable_softmax(lp_lat_b[None])[0] if n_lat_cur > 0 else np.full(3, 1.0 / 3.0)
        )

        u_c4: np.ndarray | None = None
        u_c4_b: np.ndarray | None = None
        if n_c4_cur > 0:
            lp_c4 = np.zeros(3, dtype=np.float64)
            lp_c4_b = np.zeros(3, dtype=np.float64)
            for s_i, f_cur in enumerate(c4_feats):
                mu_tr = self._clean4_mus[s_i]
                if is_clean:
                    self._c4_f_sums[s_i] += f_cur
                mean_cur = (self._recenter_n0 * mu_tr + self._c4_f_sums[s_i]) / denom_rec
                f_rec = f_cur - mean_cur + mu_tr
                self._c4_f_hist[s_i].append(f_rec)
                pr_a = CovarianceMatrixOperator.stable_softmax((f_rec @ self._c4_w[s_i].T + self._c4_b[s_i])[None])[0]
                lp_c4 += np.log(np.clip(pr_a, 1e-12, 1.0)) / max(1, n_c4_cur)
                pr_b = CovarianceMatrixOperator.stable_softmax((f_rec @ self._c4_w_b[s_i].T + self._c4_b_b[s_i])[None])[0]
                lp_c4_b += np.log(np.clip(pr_b, 1e-12, 1.0)) / max(1, n_c4_cur)
            u_c4 = CovarianceMatrixOperator.stable_softmax(lp_c4[None])[0]
            u_c4_b = CovarianceMatrixOperator.stable_softmax(lp_c4_b[None])[0]

        u_r4c: np.ndarray | None = None
        u_r4c_b: np.ndarray | None = None
        if n_r4c_cur > 0:
            lp_r4c = np.zeros(3, dtype=np.float64)
            lp_r4c_b = np.zeros(3, dtype=np.float64)
            for s_i, f_cur in enumerate(r4c_feats):
                mu_tr = self._rms4c_mus[s_i]
                if is_clean:
                    self._r4c_f_sums[s_i] += f_cur
                if use_r4c_pcr:
                    qg_s = (self._r4c_f_sums[s_i] / float(n_good)) if n_good > 0 else mu_tr
                    qk_s = r4c_ker_all[s_i] if r4c_ker_all is not None else np.zeros_like(mu_tr)
                    mean_cur = (8.0 * mu_tr + 2.0 * qg_s + qk_s) / den_pcr
                else:
                    mean_cur = (self._recenter_n0 * mu_tr + self._r4c_f_sums[s_i]) / denom_rec
                f_rec = f_cur - mean_cur + mu_tr
                self._r4c_f_hist[s_i].append(f_rec)
                pr_a = CovarianceMatrixOperator.stable_softmax((f_rec @ self._r4c_w[s_i].T + self._r4c_b[s_i])[None])[0]
                lp_r4c += np.log(np.clip(pr_a, 1e-12, 1.0)) / max(1, n_r4c_cur)
                pr_b = CovarianceMatrixOperator.stable_softmax((f_rec @ self._r4c_w_b[s_i].T + self._r4c_b_b[s_i])[None])[0]
                lp_r4c_b += np.log(np.clip(pr_b, 1e-12, 1.0)) / max(1, n_r4c_cur)
            u_r4c = CovarianceMatrixOperator.stable_softmax(lp_r4c[None])[0]
            u_r4c_b = CovarianceMatrixOperator.stable_softmax(lp_r4c_b[None])[0]

        u_rms_leg: np.ndarray | None = None
        u_rms_leg_b: np.ndarray | None = None
        if len(rms_feats) > 0:
            lp_rms = np.zeros(3, dtype=np.float64)
            lp_rms_b = np.zeros(3, dtype=np.float64)
            for s_i, f_cur in enumerate(rms_feats):
                mu_tr = self._rms_mus[s_i]
                if is_clean:
                    self._rms_f_sums[s_i] += f_cur
                if use_pcrec:
                    rg_s = (self._rms_f_sums[s_i] / float(n_good)) if n_good > 0 else mu_tr
                    rk_s = rms_ker_all[s_i] if rms_ker_all is not None else np.zeros_like(mu_tr)
                    mean_cur = (8.0 * mu_tr + 2.0 * rg_s + rk_s) / den_pcr
                else:
                    mean_cur = (self._recenter_n0 * mu_tr + self._rms_f_sums[s_i]) / denom_rec
                f_rec = f_cur - mean_cur + mu_tr
                self._rms_f_hist[s_i].append(f_rec)
                pr_a = CovarianceMatrixOperator.stable_softmax((f_rec @ self._rms_w[s_i].T + self._rms_b[s_i])[None])[0]
                lp_rms += np.log(np.clip(pr_a, 1e-12, 1.0)) / max(1, len(rms_feats))
                pr_b = CovarianceMatrixOperator.stable_softmax((f_rec @ self._rms_w_b[s_i].T + self._rms_b_b[s_i])[None])[0]
                lp_rms_b += np.log(np.clip(pr_b, 1e-12, 1.0)) / max(1, len(rms_feats))
            u_rms_leg = CovarianceMatrixOperator.stable_softmax(lp_rms[None])[0]
            u_rms_leg_b = CovarianceMatrixOperator.stable_softmax(lp_rms_b[None])[0]

        self.last_u_rms_prot_a = u_rms_leg if u_rms_leg is not None else np.full(3, 1.0 / 3.0, dtype=np.float64)
        self.last_u_rms_leg_a = self.last_u_rms_prot_a
        self.last_u_rms_leg_b = u_rms_leg_b if u_rms_leg_b is not None else np.full(3, 1.0 / 3.0, dtype=np.float64)

        if u_c4 is not None and u_rms_leg is not None:
            a4 = self._alpha_clean4
            lp_a = (1.0 - a4) * np.log(np.clip(u_rms_leg, 1e-12, 1.0)) + a4 * np.log(np.clip(u_c4, 1e-12, 1.0))
            u_rms = CovarianceMatrixOperator.stable_softmax(lp_a[None])[0]
            u_c4_use_b = u_c4_b if u_c4_b is not None else u_c4
            assert u_rms_leg_b is not None
            lp_b = (1.0 - a4) * np.log(np.clip(u_rms_leg_b, 1e-12, 1.0)) + a4 * np.log(np.clip(u_c4_use_b, 1e-12, 1.0))
            self.last_u_rms_b = CovarianceMatrixOperator.stable_softmax(lp_b[None])[0]
        elif u_c4 is not None:
            u_rms = u_c4
            self.last_u_rms_b = u_c4_b if u_c4_b is not None else u_c4
        elif u_rms_leg is not None:
            u_rms = u_rms_leg
            self.last_u_rms_b = self.last_u_rms_leg_b
        else:
            u_rms = np.full(3, 1.0 / 3.0, dtype=np.float64)
            self.last_u_rms_b = np.full(3, 1.0 / 3.0, dtype=np.float64)

        self.last_u_rms_c4_a = u_rms
        self.last_u_rms_c4_b = self.last_u_rms_b

        if u_r4c is not None:
            ar4 = self._alpha_rms4c
            lp_s7_r4 = (1.0 - ar4) * np.log(np.clip(self.last_u_rms_leg_a, 1e-12, 1.0)) + ar4 * np.log(
                np.clip(u_r4c, 1e-12, 1.0)
            )
            self.last_u_rms_leg_a = CovarianceMatrixOperator.stable_softmax(lp_s7_r4[None])[0]
            lp_s6_r4 = (1.0 - ar4) * np.log(np.clip(u_rms, 1e-12, 1.0)) + ar4 * np.log(np.clip(u_r4c, 1e-12, 1.0))
            u_rms = CovarianceMatrixOperator.stable_softmax(lp_s6_r4[None])[0]
            if u_r4c_b is not None:
                lp_b_r4 = (1.0 - ar4) * np.log(np.clip(self.last_u_rms_b, 1e-12, 1.0)) + ar4 * np.log(
                    np.clip(u_r4c_b, 1e-12, 1.0)
                )
                self.last_u_rms_b = CovarianceMatrixOperator.stable_softmax(lp_b_r4[None])[0]

        self.last_u_lat_a = u_lat
        self.last_u_rms_a = u_rms
        self._step_index += 1
        return u_lat, u_rms

    def observe_teacher(
        self,
        teacher_prob_a: np.ndarray,
        teacher_prob_b: np.ndarray | None = None,
    ) -> None:
        self._tchr_hist.append(np.asarray(teacher_prob_a, dtype=np.float64).copy())
        if teacher_prob_b is not None:
            self._tchr_hist_b.append(np.asarray(teacher_prob_b, dtype=np.float64).copy())

    def step_causal_window(self, cleaned_window: np.ndarray, is_artifact_flagged: bool) -> np.ndarray:
        u_lat, u_rms = self.step_quad_views(cleaned_window, is_artifact_flagged)
        fused_lp = 0.5 * CovarianceMatrixOperator.centered_log_probabilities(u_lat) + 0.5 * CovarianceMatrixOperator.centered_log_probabilities(u_rms)
        return CovarianceMatrixOperator.stable_softmax(fused_lp[None])[0]

    @classmethod
    def load_from_artifact_directory(
        cls,
        artifact_directory: str | Path,
        n_lat52: int = 8,
        n_clean4: int = 6,
        n_rms4c: int = 2,
        n_rms56: int = 3,
        execution_device: str | torch.device = "cpu",
        ea_prior: float = 8.0,
        recenter_n0: float = 10.0,
        refit_at: Sequence[int] = (66, 126),
        alpha_clean4: float = 0.35,
        alpha_rms4c: float = 0.65,
        eegpt_mode: str = "ee7_pcearms_pcrec_t58",
        rms4c_pc_recenter: bool = False,
    ) -> QuadViewEegptFoundationOrgan:
        resolved_dir = ParameterTypeValidator.require_path(artifact_directory, "artifact_directory", must_exist=True)
        with np.load(resolved_dir / "base_eegpt_fp16.npz", allow_pickle=False) as zb:
            base_fp16 = {k_str: zb[k_str].copy() for k_str in zb.files}
        with np.load(resolved_dir / "eegpt_align_priors.npz", allow_pickle=False) as zp:
            r0_52_broad = zp["R0_52_broad"].copy()
            r0_var = zp["r0_var"].copy()

        lat_bundles, lat_mus, lat_reps = [], [], []
        for idx in range(int(n_lat52)):
            b_item, mu_item, rep_item = BatchedEegptFoundationBackbone.reconstruct_seed_bundle_from_npz(
                resolved_dir / f"lat52_s{idx}.npz", base_fp16, "lat52"
            )
            lat_bundles.append(b_item)
            lat_mus.append(mu_item)
            lat_reps.append(rep_item)

        c4_bundles, c4_mus, c4_reps = [], [], []
        for idx in range(int(n_clean4)):
            b_item, mu_item, rep_item = BatchedEegptFoundationBackbone.reconstruct_seed_bundle_from_npz(
                resolved_dir / f"eegpt_clean4_s{idx}.npz", base_fp16, "clean4"
            )
            c4_bundles.append(b_item)
            c4_mus.append(mu_item)
            c4_reps.append(rep_item)

        r4c_bundles, r4c_mus, r4c_reps = [], [], []
        for idx in range(int(n_rms4c)):
            b_item, mu_item, rep_item = BatchedEegptFoundationBackbone.reconstruct_seed_bundle_from_npz(
                resolved_dir / f"eegpt_4corp_nodreyer_s{idx}.npz", base_fp16, "rms4c"
            )
            r4c_bundles.append(b_item)
            r4c_mus.append(mu_item)
            r4c_reps.append(rep_item)

        rms_bundles, rms_mus, rms_reps = [], [], []
        for idx in range(int(n_rms56)):
            b_item, mu_item, rep_item = BatchedEegptFoundationBackbone.reconstruct_seed_bundle_from_npz(
                resolved_dir / f"rms56_s{idx}.npz", base_fp16, "rms56"
            )
            rms_bundles.append(b_item)
            rms_mus.append(mu_item)
            rms_reps.append(rep_item)

        all_bundles = lat_bundles + c4_bundles + r4c_bundles + rms_bundles
        backbone = BatchedEegptFoundationBackbone(
            seed_state_bundles=all_bundles,
            count_52hz_models=len(lat_bundles),
            execution_device=execution_device,
        )
        return cls(
            batched_backbone=backbone,
            lat_mus=lat_mus,
            lat_reps=lat_reps,
            rms_mus=rms_mus,
            rms_reps=rms_reps,
            r0_52_broad=r0_52_broad,
            r0_var=r0_var,
            execution_device=execution_device,
            ea_prior=ea_prior,
            recenter_n0=recenter_n0,
            refit_at=refit_at,
            clean4_mus=c4_mus,
            clean4_reps=c4_reps,
            alpha_clean4=alpha_clean4,
            rms4c_mus=r4c_mus,
            rms4c_reps=r4c_reps,
            alpha_rms4c=alpha_rms4c,
            eegpt_mode=eegpt_mode,
            rms4c_pc_recenter=rms4c_pc_recenter,
            base_fp16_tensors=base_fp16,
        )

    @classmethod
    def fit_from_corpora(
        cls,
        base_fp16_tensors: dict[str, np.ndarray],
        proteus_windows: np.ndarray,
        proteus_labels: np.ndarray,
        n_lat52: int = 1,
        n_clean4: int = 0,
        n_rms4c: int = 0,
        n_rms56: int = 1,
        auxiliary_windows: np.ndarray | None = None,
        auxiliary_labels: np.ndarray | None = None,
        fine_tune_epochs: int = 4,
        batch_size: int = 32,
        learning_rate: float = 2e-4,
        svd_rank: int = 64,
        replay_per_class: int = 64,
        random_seed: int = 2026,
        execution_device: str | torch.device = "cpu",
    ) -> QuadViewEegptFoundationOrgan:
        """Fine-tune quad-view EEGPT seeds from `base_eegpt_fp16` and compress Rank-`svd_rank` deltas."""
        dev = torch.device(execution_device)
        x_prot = np.asarray(proteus_windows, dtype=np.float64)
        y_prot = np.asarray(proteus_labels, dtype=np.int64)
        x_cent = x_prot - x_prot.mean(axis=-1, keepdims=True)
        fir52 = SpectralSignalOperator.design_bandpass_fir((4.0, 52.0), tap_count=121, sampling_rate_hz=120.0)
        xc52 = SpectralSignalOperator.apply_zero_phase_fir(x_cent, fir52)
        r0_52_broad = CovarianceMatrixOperator.ledoit_wolf_covariance(xc52).mean(axis=0)
        x_car = x_cent - x_cent.mean(axis=1, keepdims=True)
        r0_var = (x_car ** 2).mean(axis=(0, 2))

        w0_52 = CovarianceMatrixOperator.inverse_matrix_square_root(r0_52_broad)
        inv_std0 = 1.0 / np.sqrt(np.maximum(r0_var, 1e-6))
        xw52_all = np.einsum("ij,njt->nit", w0_52, xc52).astype(np.float32)
        xw56_all = (x_car * inv_std0[None, :, None]).astype(np.float32)

        if auxiliary_windows is not None and auxiliary_labels is not None and len(auxiliary_windows) > 0:
            x_aux = np.asarray(auxiliary_windows, dtype=np.float64)
            x_aux_c = x_aux - x_aux.mean(axis=-1, keepdims=True)
            x_aux_car = (x_aux_c - x_aux_c.mean(axis=1, keepdims=True)) * inv_std0[None, :, None]
            xw56_joint = np.concatenate([xw56_all, x_aux_car.astype(np.float32)], axis=0)
            y_joint = np.concatenate([y_prot, np.asarray(auxiliary_labels, dtype=np.int64)], axis=0)
        else:
            xw56_joint = xw56_all
            y_joint = y_prot

        seed_payloads: dict[str, dict[str, np.ndarray]] = {}
        lat_bundles, lat_mus, lat_reps = [], [], []
        c4_bundles, c4_mus, c4_reps = [], [], []
        r4c_bundles, r4c_mus, r4c_reps = [], [], []
        rms_bundles, rms_mus, rms_reps = [], [], []

        def _train_single_seed(
            seed_filename: str,
            stream_tag: str,
            train_x: np.ndarray,
            train_y: np.ndarray,
            eval_x_prot: np.ndarray,
            is_4tok: bool,
            lp_hz: float,
            in_sc: float,
            seed_offset: int,
        ) -> tuple[dict[str, object], np.ndarray, tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]]:
            torch.manual_seed(int(random_seed) + seed_offset)
            rng_local = np.random.default_rng(int(random_seed) + seed_offset)
            seed_mod = BatchedEegptFoundationBackbone.build_trainable_seed_module(
                base_fp16_tensors=base_fp16_tensors,
                is_four_token=is_4tok,
                lowpass_hz=lp_hz,
                in_scale_init=in_sc,
                execution_device=dev,
            )
            optimizer = torch.optim.AdamW(seed_mod.parameters(), lr=float(learning_rate), weight_decay=1e-2)
            seed_mod.train()
            n_tr = len(train_x)
            for _ in range(max(1, int(fine_tune_epochs))):
                perm = rng_local.permutation(n_tr)
                for b_start in range(0, n_tr, int(batch_size)):
                    batch_idx = perm[b_start : b_start + int(batch_size)]
                    xb_t = torch.as_tensor(train_x[batch_idx], dtype=torch.float32, device=dev)
                    yb_t = torch.as_tensor(train_y[batch_idx], dtype=torch.long, device=dev)
                    optimizer.zero_grad()
                    feats_b = BatchedEegptFoundationBackbone.extract_trainable_seed_features(seed_mod, xb_t)
                    logits_b = seed_mod["fc"](seed_mod["drop"](feats_b))
                    loss_b = F.cross_entropy(logits_b, yb_t)
                    loss_b.backward()
                    optimizer.step()

            seed_mod.eval()
            feat_chunks: list[np.ndarray] = []
            with torch.no_grad():
                for b_start in range(0, len(eval_x_prot), int(batch_size)):
                    xb_t = torch.as_tensor(
                        eval_x_prot[b_start : b_start + int(batch_size)], dtype=torch.float32, device=dev
                    )
                    feat_chunks.append(
                        BatchedEegptFoundationBackbone.extract_trainable_seed_features(seed_mod, xb_t)
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
            feats_prot = np.concatenate(feat_chunks, axis=0)
            mu_tr = feats_prot.mean(axis=0)
            w_head, b_head = cls.fit_multinomial_logistic_lbfgs(feats_prot, y_prot, inverse_reg_c=1.0, pull_strength=1.0)

            rep_indices: list[np.ndarray] = []
            for cls_id in range(3):
                cls_rows = np.flatnonzero(y_prot == cls_id)
                if len(cls_rows) > 0:
                    take_n = min(int(replay_per_class), len(cls_rows))
                    rep_indices.append(rng_local.choice(cls_rows, size=take_n, replace=False))
            sel_idx = np.concatenate(rep_indices, axis=0) if rep_indices else np.arange(min(len(y_prot), 32))
            rep_tuple = (feats_prot[sel_idx], y_prot[sel_idx], (w_head, b_head))

            npz_dict = BatchedEegptFoundationBackbone.compress_trained_seed_to_npz_arrays(
                seed_module=seed_mod,
                base_fp16_tensors=base_fp16_tensors,
                train_mean=mu_tr,
                replay_tuple=rep_tuple,
                svd_rank=svd_rank,
            )
            seed_payloads[seed_filename] = npz_dict

            state_tensors: dict[str, torch.Tensor] = {}
            for k_str, p_t in seed_mod["eegpt"].state_dict().items():
                if not k_str.startswith("final_layer."):
                    state_tensors[f"eegpt.{k_str}"] = p_t.detach().cpu().float()
            for k_str, p_t in seed_mod.state_dict().items():
                if not k_str.startswith(("eegpt.", "aux_", "lowpass_hz_buf", "is_four_tok_buf")):
                    state_tensors[k_str] = p_t.detach().cpu().float()
            state_tensors["eegpt.chans_id"] = seed_mod["eegpt"].chans_id.detach().cpu().to(torch.long)
            bundle_obj: dict[str, object] = {
                "is_four_token": is_4tok,
                "lowpass_hz": lp_hz,
                "in_scale_init": in_sc,
                "state_tensors": state_tensors,
            }
            return bundle_obj, mu_tr, rep_tuple

        for s_i in range(int(n_lat52)):
            b_o, m_o, r_o = _train_single_seed(
                f"lat52_s{s_i}.npz", "lat52", xw52_all, y_prot, xw52_all, True, 52.0, 0.015, s_i
            )
            lat_bundles.append(b_o)
            lat_mus.append(m_o)
            lat_reps.append(r_o)

        for s_i in range(int(n_clean4)):
            b_o, m_o, r_o = _train_single_seed(
                f"eegpt_clean4_s{s_i}.npz", "clean4", xw56_joint, y_joint, xw56_all, True, 56.0, 0.018, 100 + s_i
            )
            c4_bundles.append(b_o)
            c4_mus.append(m_o)
            c4_reps.append(r_o)

        for s_i in range(int(n_rms4c)):
            b_o, m_o, r_o = _train_single_seed(
                f"eegpt_4corp_nodreyer_s{s_i}.npz", "rms4c", xw56_joint, y_joint, xw56_all, True, 56.0, 0.018, 200 + s_i
            )
            r4c_bundles.append(b_o)
            r4c_mus.append(m_o)
            r4c_reps.append(r_o)

        for s_i in range(int(n_rms56)):
            b_o, m_o, r_o = _train_single_seed(
                f"rms56_s{s_i}.npz", "rms56", xw56_all, y_prot, xw56_all, False, 56.0, 0.018, 300 + s_i
            )
            rms_bundles.append(b_o)
            rms_mus.append(m_o)
            rms_reps.append(r_o)

        backbone = BatchedEegptFoundationBackbone(
            seed_state_bundles=lat_bundles + c4_bundles + r4c_bundles + rms_bundles,
            count_52hz_models=len(lat_bundles),
            execution_device=dev,
        )
        return cls(
            batched_backbone=backbone,
            lat_mus=lat_mus,
            lat_reps=lat_reps,
            rms_mus=rms_mus,
            rms_reps=rms_reps,
            r0_52_broad=r0_52_broad,
            r0_var=r0_var,
            execution_device=dev,
            clean4_mus=c4_mus,
            clean4_reps=c4_reps,
            rms4c_mus=r4c_mus,
            rms4c_reps=r4c_reps,
            base_fp16_tensors=base_fp16_tensors,
            seed_npz_payloads=seed_payloads,
        )

    def save_organ_artifacts(self, target_directory: Path) -> list[Path]:
        resolved_dir = ParameterTypeValidator.require_path(target_directory, "target_directory")
        resolved_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []

        priors_path = resolved_dir / "eegpt_align_priors.npz"
        np.savez_compressed(priors_path, R0_52_broad=self._r0_52_broad, r0_var=self._r0_var)
        saved_paths.append(priors_path)

        if self._base_fp16_tensors is not None:
            base_path = resolved_dir / "base_eegpt_fp16.npz"
            np.savez_compressed(base_path, **self._base_fp16_tensors)
            saved_paths.append(base_path)

        if self._seed_npz_payloads is not None:
            for seed_filename, npz_arrays in sorted(self._seed_npz_payloads.items()):
                seed_out_path = resolved_dir / seed_filename
                np.savez_compressed(seed_out_path, **npz_arrays)
                saved_paths.append(seed_out_path)

        return saved_paths
