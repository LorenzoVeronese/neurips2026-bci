"""S3D 16-D multi-subject Fisher anchored-ridge multimodal test-time-training organ."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from neurips_bci.operator.matrix_operators.SubspaceMatrixOperator import SubspaceMatrixOperator
from neurips_bci.operator.signal_operators.SpectralSignalOperator import SpectralSignalOperator
from neurips_bci.organ.multimodal_organs.MultimodalOrgan import MultimodalOrgan
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class FisherAnchoredRidgeMultimodalOrgan(MultimodalOrgan):
    """Causal `B=1` `16x16` PC-kernel sharpened-margin anchored-ridge multimodal organ (`s3d`)."""

    def __init__(self, fisher_spec: dict[str, Any], max_history: int = 512) -> None:
        super().__init__(organ_identifier="fisher_anchored_ridge_multimodal_organ")
        self._fisher_spec = fisher_spec
        self._max_history = ParameterTypeValidator.require_integer(max_history, "max_history", 8)
        self._w0_16 = np.asarray(fisher_spec["W0_16"], dtype=np.float64)
        self._logit_scale = float(fisher_spec["logit_scale_16"])
        self._q_z_sr = np.asarray(fisher_spec["Q_z_sr"], dtype=np.float64)
        self._q_z_ee = np.asarray(fisher_spec["Q_z_ee"], dtype=np.float64)
        self._q_cd = np.asarray(fisher_spec["Q_cd"], dtype=np.float64)
        self._q_cb = np.asarray(fisher_spec["Q_cb"], dtype=np.float64)
        self._r_mu0 = np.asarray(fisher_spec["r_mu0"], dtype=np.float64)
        self._mu_phi = np.asarray(fisher_spec["mu_phi"], dtype=np.float64)
        self._sd_phi = np.asarray(fisher_spec["sd_phi"], dtype=np.float64)
        self._u_run = np.asarray(fisher_spec["U_run"], dtype=np.float64)
        self._lam0 = float(fisher_spec["lam0"])
        self._alpha_loc = float(fisher_spec["alpha_loc"])
        self._gamma_sharp = float(fisher_spec["gamma_sharp"])
        self._beta_margin = float(fisher_spec["beta_margin"])
        self._pc_tau = 1.3
        self._n0_cent = 6.0
        self._n0_glob = 2.0
        self._bal_prior = 10.0
        self._eye16 = np.eye(16, dtype=np.float64)
        self.reset_organ_state()

    @property
    def q_cd(self) -> np.ndarray:
        return self._q_cd

    @property
    def q_cb(self) -> np.ndarray:
        return self._q_cb

    @property
    def q_z_sr(self) -> np.ndarray:
        return self._q_z_sr

    @property
    def q_z_ee(self) -> np.ndarray:
        return self._q_z_ee

    def reset_organ_state(self) -> None:
        self._cs_r = np.zeros(16, dtype=np.float64)
        self._cs_n = 0.0
        self._n_teach_all = 0
        self._p_teach_cum = np.zeros(3, dtype=np.float64)
        self._n_bal = 0
        self._adapt_bal_sum = np.zeros(3, dtype=np.float64)
        self._n_hist = 0
        self._u_run_buf = np.empty((self._max_history, 6), dtype=np.float64)
        self._ok_buf = np.empty(self._max_history, dtype=np.float64)
        self._r_buf = np.empty((self._max_history, 16), dtype=np.float64)
        self._u_buf = np.empty((self._max_history, 16), dtype=np.float64)
        self._p_base_buf = np.empty((self._max_history, 3), dtype=np.float64)
        self._q_buf = np.empty((self._max_history, 3), dtype=np.float64)
        self._a_buf = np.empty(self._max_history, dtype=np.float64)
        self._staged_context: tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None] | None = None
        self.last_p_s3d = np.full(3, 1.0 / 3.0, dtype=np.float64)

    def stage_multimodal_context(
        self,
        p_h9_t: np.ndarray,
        z_sr_t: np.ndarray | None,
        z_ee_t: np.ndarray | None,
        u16_cd_t: np.ndarray | None,
        u16_cb_t: np.ndarray | None,
        phi_nm_pre: np.ndarray | None = None,
    ) -> None:
        self._staged_context = (p_h9_t, z_sr_t, z_ee_t, u16_cd_t, u16_cb_t, phi_nm_pre)

    def _pc_sinkhorn_step(self, p_base_t: np.ndarray, k_past_t: np.ndarray, n_past: int) -> np.ndarray:
        u_t = np.clip(np.asarray(p_base_t, dtype=np.float64), 1e-12, 1.0)
        u_t = u_t / float(np.sum(u_t))
        u_glob_t = (self._p_teach_cum + u_t) / float(self._n_teach_all + 1)
        if n_past > 0:
            w_vec = np.concatenate([k_past_t, np.array([0.25], dtype=np.float64)])
            u_mat = np.vstack([self._p_base_buf[:n_past], u_t[None, :]])
        else:
            w_vec = np.array([0.25], dtype=np.float64)
            u_mat = u_t[None, :]
        den = 10.0 + 2.0 + float(np.sum(w_vec))
        col_scale = np.ones(3, dtype=np.float64)
        for _ in range(4):
            q_pri = col_scale / max(float(np.sum(col_scale)), 1e-15)
            ug_c = u_glob_t * col_scale
            q_glb = ug_c / max(float(np.sum(ug_c)), 1e-15)
            q_mat = u_mat * col_scale[None, :]
            q_mat = q_mat / np.maximum(np.sum(q_mat, axis=-1, keepdims=True), 1e-15)
            marg = (10.0 * q_pri + 2.0 * q_glb + (w_vec @ q_mat)) / max(den, 1e-8)
            col_scale = col_scale / (3.0 * np.clip(marg, 1e-6, 1.0))
            col_scale = col_scale / float(np.mean(col_scale))
        q_out = u_t * col_scale
        return q_out / float(np.sum(q_out))

    def step_multimodal(
        self,
        p_h9_t: np.ndarray,
        z_sr_t: np.ndarray | None,
        z_ee_t: np.ndarray | None,
        u16_cd_t: np.ndarray | None,
        u16_cb_t: np.ndarray | None,
        raw_window: np.ndarray,
        is_artifact_flagged: bool = False,
        phi_nm_pre: np.ndarray | None = None,
    ) -> np.ndarray | None:
        if z_sr_t is None or z_ee_t is None or u16_cd_t is None or u16_cb_t is None:
            return None
        is_extreme = (
            bool(is_artifact_flagged)
            or (not np.isfinite(raw_window).all())
            or bool(np.max(np.abs(raw_window)) > 5000.0)
        )
        r_t = (
            np.asarray(z_sr_t, dtype=np.float64) @ self._q_z_sr
            + np.asarray(z_ee_t, dtype=np.float64) @ self._q_z_ee
            + np.asarray(u16_cd_t, dtype=np.float64)
            + np.asarray(u16_cb_t, dtype=np.float64)
        )
        if not np.isfinite(r_t).all():
            return None

        phi_vec = (
            phi_nm_pre
            if phi_nm_pre is not None
            else SpectralSignalOperator.extract_non_motor_246_features(raw_window)
        )
        u_run_t = ((phi_vec - self._mu_phi) / self._sd_phi) @ self._u_run
        if not np.isfinite(u_run_t).all():
            u_run_t = np.zeros(6, dtype=np.float64)
            is_extreme = True

        nh = int(self._n_hist)
        ok_f = 0.0 if is_extreme else 1.0
        if nh > 0:
            d2 = np.sum((self._u_run_buf[:nh] - u_run_t[None, :]) ** 2, axis=1)
            k_past_t = np.exp(-0.5 * d2 / (6.0 * (self._pc_tau ** 2))) * self._ok_buf[:nh]
        else:
            k_past_t = np.zeros(0, dtype=np.float64)

        cs_r_new = self._cs_r + ok_f * r_t
        cs_n_new = self._cs_n + ok_f
        u_glob_r = (cs_r_new + 1e-6 * self._r_mu0) / max(cs_n_new, 1e-6)
        loc_sum_r = ((k_past_t @ self._r_buf[:nh]) if nh > 0 else np.zeros(16, dtype=np.float64)) + ok_f * r_t
        loc_w = float(np.sum(k_past_t)) + ok_f
        cen_r = (self._n0_cent * self._r_mu0 + self._n0_glob * u_glob_r + loc_sum_r) / (
            self._n0_cent + self._n0_glob + loc_w
        )
        u_t = r_t - cen_r

        if nh == 0:
            w_h = self._w0_16
        else:
            w_causal_t = self._alpha_loc * k_past_t + (1.0 - self._alpha_loc) * self._ok_buf[:nh]
            wt = w_causal_t * self._a_buf[:nh]
            if float(np.sum(wt)) <= 1e-8:
                w_h = self._w0_16
            else:
                u_p = self._u_buf[:nh]
                uw = u_p * wt[:, None]
                gram = uw.T @ u_p + self._lam0 * self._eye16
                rhs = uw.T @ (self._q_buf[:nh] - (1.0 / 3.0)) + self._lam0 * self._w0_16
                w_h = np.linalg.solve(gram, rhs)
                w_h -= np.mean(w_h, axis=1, keepdims=True)

        s_t = (u_t @ w_h) * self._logit_scale
        s_t = s_t - float(np.mean(s_t))
        e_t = np.exp(np.clip(s_t - float(np.max(s_t)), -60.0, 0.0))
        u_adapt = e_t / float(np.sum(e_t))

        if not is_extreme:
            self._n_bal += 1
            self._adapt_bal_sum += u_adapt
        p_bar = np.clip(
            (self._bal_prior / 3.0 + self._adapt_bal_sum) / (self._bal_prior + float(self._n_bal)), 1e-6, 1.0
        )
        q_adapt = u_adapt / p_bar
        p_adapt = q_adapt / float(np.sum(q_adapt))

        p_teach_t = self._pc_sinkhorn_step(p_h9_t, k_past_t, nh)
        q_t = np.power(np.clip(p_teach_t, 1e-9, 1.0), self._gamma_sharp)
        q_t = q_t / float(np.sum(q_t))
        if self._beta_margin > 0.0:
            ps = np.sort(p_teach_t)
            a_t = float(np.power(np.clip(ps[2] - ps[1], 0.02, 1.0), self._beta_margin))
        else:
            a_t = 1.0

        if not is_extreme:
            self._cs_r = cs_r_new
            self._cs_n = cs_n_new
            self._n_teach_all += 1
            norm_h9 = np.clip(np.asarray(p_h9_t, dtype=np.float64), 1e-12, 1.0) / float(np.sum(p_h9_t))
            self._p_teach_cum += norm_h9
            if nh < self._max_history:
                self._u_run_buf[nh] = u_run_t
                self._ok_buf[nh] = ok_f
                self._r_buf[nh] = r_t
                self._u_buf[nh] = u_t
                self._p_base_buf[nh] = norm_h9
                self._q_buf[nh] = q_t
                self._a_buf[nh] = a_t
                self._n_hist = nh + 1
            else:
                self._u_run_buf[:-1] = self._u_run_buf[1:]
                self._ok_buf[:-1] = self._ok_buf[1:]
                self._r_buf[:-1] = self._r_buf[1:]
                self._u_buf[:-1] = self._u_buf[1:]
                self._p_base_buf[:-1] = self._p_base_buf[1:]
                self._q_buf[:-1] = self._q_buf[1:]
                self._a_buf[:-1] = self._a_buf[1:]
                self._u_run_buf[-1] = u_run_t
                self._ok_buf[-1] = ok_f
                self._r_buf[-1] = r_t
                self._u_buf[-1] = u_t
                self._p_base_buf[-1] = norm_h9
                self._q_buf[-1] = q_t
                self._a_buf[-1] = a_t

        self.last_p_s3d = p_adapt
        return p_adapt

    def step_causal_window(self, cleaned_window: np.ndarray, is_artifact_flagged: bool) -> np.ndarray:
        if self._staged_context is None:
            return self.last_p_s3d.copy()
        p_h9, z_sr, z_ee, u16_cd, u16_cb, phi_pre = self._staged_context
        self._staged_context = None
        stepped = self.step_multimodal(
            p_h9_t=p_h9,
            z_sr_t=z_sr,
            z_ee_t=z_ee,
            u16_cd_t=u16_cd,
            u16_cb_t=u16_cb,
            raw_window=cleaned_window,
            is_artifact_flagged=is_artifact_flagged,
            phi_nm_pre=phi_pre,
        )
        return self.last_p_s3d.copy() if stepped is None else stepped

    @classmethod
    def load_from_npz(cls, s3d_npz_path: str | Path, max_history: int = 512) -> FisherAnchoredRidgeMultimodalOrgan:
        resolved_npz = ParameterTypeValidator.require_path(s3d_npz_path, "s3d_npz_path", must_exist=True)
        with np.load(resolved_npz, allow_pickle=False) as archive:
            q_sr = archive["blk_z_sr__Q16_f16"].astype(np.float64)
            q_ee = archive["blk_z_ee__Q16_f16"].astype(np.float64)
            q_cd = archive["blk_cd__Q16_f16"].astype(np.float64)
            q_cb = archive["blk_cb__Q16_f16"].astype(np.float64)
            mu_sr = archive["blk_z_sr__mu0"].astype(np.float64)
            mu_ee = archive["blk_z_ee__mu0"].astype(np.float64)
            mu_cd = archive["blk_cd__mu0"].astype(np.float64)
            mu_cb = archive["blk_cb__mu0"].astype(np.float64)
            r_mu0 = mu_sr @ q_sr + mu_ee @ q_ee + mu_cd @ q_cd + mu_cb @ q_cb
            fisher_spec: dict[str, Any] = {
                "W0_16": archive["W0_16"].astype(np.float64),
                "logit_scale_16": float(archive["logit_scale_16"]),
                "Q_z_sr": q_sr,
                "Q_z_ee": q_ee,
                "Q_cd": q_cd,
                "Q_cb": q_cb,
                "mu_z_sr": mu_sr,
                "mu_z_ee": mu_ee,
                "mu_cd": mu_cd,
                "mu_cb": mu_cb,
                "r_mu0": r_mu0,
                "mu_phi": archive["rsub__mu_phi"].astype(np.float64),
                "sd_phi": archive["rsub__sd_phi"].astype(np.float64),
                "U_run": archive["rsub__U_run"].astype(np.float64),
                "lam0": float(archive["cfg_lam0_16"]),
                "alpha_loc": float(archive["cfg_alpha_loc"]),
                "gamma_sharp": float(archive["cfg_gamma_sharp"]),
                "beta_margin": float(archive["cfg_beta_margin"]),
            }
        return cls(fisher_spec=fisher_spec, max_history=max_history)

    @classmethod
    def fit_from_multimodal_features(
        cls,
        windows: np.ndarray,
        class_labels: np.ndarray,
        subject_ids: np.ndarray,
        session_ids: np.ndarray,
        z_sr_matrix: np.ndarray,
        z_ee_matrix: np.ndarray,
        cd_feature_matrix: np.ndarray,
        cb_feature_matrix: np.ndarray,
        lam0: float = 36.0,
        alpha_loc: float = 0.75,
        gamma_sharp: float = 1.5,
        beta_margin: float = 1.0,
        logit_scale: float = 6.0,
    ) -> FisherAnchoredRidgeMultimodalOrgan:
        """Fit joint 16-D multi-subject Fisher subspace (`z_sr`, `z_ee`, `cd`, `cb`) and anchored ridge prior `W0_16`."""
        y_all = np.asarray(class_labels, dtype=np.int64)
        subj_all = np.asarray(subject_ids)
        sess_all = np.asarray(session_ids)
        mu_phi, sd_phi, u_run = cls.fit_non_motor_run_subspace(windows, sess_all, subspace_dim=6, within_shrinkage=0.25)

        blocks = [
            np.asarray(z_sr_matrix, dtype=np.float64),
            np.asarray(z_ee_matrix, dtype=np.float64),
            np.asarray(cd_feature_matrix, dtype=np.float64),
            np.asarray(cb_feature_matrix, dtype=np.float64),
        ]
        block_mus = [blk_m.mean(axis=0) for blk_m in blocks]
        block_sds = [np.maximum(blk_m.std(axis=0), 1e-6) for blk_m in blocks]
        reduced_parts: list[np.ndarray] = []
        block_bases: list[np.ndarray] = []
        for blk_m, mu_b, sd_b in zip(blocks, block_mus, block_sds):
            std_b = (blk_m - mu_b[None, :]) / sd_b[None, :]
            _, _, vh_b = np.linalg.svd(std_b, full_matrices=False)
            k_b = min(64, vh_b.shape[0])
            basis_b = (vh_b[:k_b].T / sd_b[:, None]) / np.sqrt(float(k_b))
            block_bases.append(basis_b)
            reduced_parts.append((blk_m - mu_b[None, :]) @ basis_b)

        joint_red = np.concatenate(reduced_parts, axis=1)
        red_d = joint_red.shape[1]
        between_s = np.zeros((red_d, red_d), dtype=np.float64)
        within_s = np.zeros((red_d, red_d), dtype=np.float64)
        for s_key in np.unique(subj_all):
            s_rows = joint_red[subj_all == s_key]
            s_y = y_all[subj_all == s_key]
            s_mean = s_rows.mean(axis=0)
            s_cent = s_rows - s_mean[None, :]
            for cls_id in range(3):
                c_rows = s_cent[s_y == cls_id]
                if len(c_rows) == 0:
                    continue
                c_mean = c_rows.mean(axis=0)
                between_s += float(len(c_rows)) * np.outer(c_mean, c_mean)
                c_res = c_rows - c_mean[None, :]
                within_s += c_res.T @ c_res
        between_s /= max(float(len(joint_red)), 1.0)
        within_s /= max(float(len(joint_red)), 1.0)
        q_red_16 = SubspaceMatrixOperator.solve_generalized_discriminant(between_s, within_s, subspace_dim=16, within_shrinkage=0.25)

        col_cursor = 0
        q_blocks: list[np.ndarray] = []
        for basis_b in block_bases:
            width_b = basis_b.shape[1]
            q_blocks.append(basis_b @ q_red_16[col_cursor : col_cursor + width_b])
            col_cursor += width_b
        q_sr, q_ee, q_cd, q_cb = q_blocks
        mu_sr, mu_ee, mu_cd, mu_cb = block_mus
        r_mu0 = mu_sr @ q_sr + mu_ee @ q_ee + mu_cd @ q_cd + mu_cb @ q_cb

        r_all = blocks[0] @ q_sr + blocks[1] @ q_ee + blocks[2] @ q_cd + blocks[3] @ q_cb
        u_cent = r_all.copy()
        for sess_key in np.unique(sess_all):
            m_sess = sess_all == sess_key
            u_cent[m_sess] = r_all[m_sess] - r_all[m_sess].mean(axis=0, keepdims=True)
        one_hot_centered = np.eye(3, dtype=np.float64)[y_all] - (1.0 / 3.0)
        w0_16 = np.linalg.solve(u_cent.T @ u_cent + float(lam0) * np.eye(16, dtype=np.float64), u_cent.T @ one_hot_centered)
        w0_16 -= np.mean(w0_16, axis=1, keepdims=True)

        fisher_spec: dict[str, Any] = {
            "W0_16": w0_16,
            "logit_scale_16": float(logit_scale),
            "Q_z_sr": q_sr,
            "Q_z_ee": q_ee,
            "Q_cd": q_cd,
            "Q_cb": q_cb,
            "mu_z_sr": mu_sr,
            "mu_z_ee": mu_ee,
            "mu_cd": mu_cd,
            "mu_cb": mu_cb,
            "r_mu0": r_mu0,
            "mu_phi": mu_phi,
            "sd_phi": sd_phi,
            "U_run": u_run,
            "lam0": float(lam0),
            "alpha_loc": float(alpha_loc),
            "gamma_sharp": float(gamma_sharp),
            "beta_margin": float(beta_margin),
        }
        return cls(fisher_spec=fisher_spec)

    def save_organ_artifacts(self, target_directory: Path) -> list[Path]:
        resolved_dir = ParameterTypeValidator.require_path(target_directory, "target_directory")
        resolved_dir.mkdir(parents=True, exist_ok=True)
        spec = self._fisher_spec
        mu_sr = np.asarray(spec.get("mu_z_sr", np.zeros(self._q_z_sr.shape[0], dtype=np.float64)), dtype=np.float32)
        mu_ee = np.asarray(spec.get("mu_z_ee", np.zeros(self._q_z_ee.shape[0], dtype=np.float64)), dtype=np.float32)
        mu_cd = np.asarray(spec.get("mu_cd", np.zeros(self._q_cd.shape[0], dtype=np.float64)), dtype=np.float32)
        mu_cb = np.asarray(spec.get("mu_cb", np.zeros(self._q_cb.shape[0], dtype=np.float64)), dtype=np.float32)
        out_path = resolved_dir / "s3d_fisher16_ancridge_2655.npz"
        np.savez_compressed(
            out_path,
            W0_16=self._w0_16.astype(np.float32),
            logit_scale_16=np.array(self._logit_scale, dtype=np.float32),
            blk_z_sr__Q16_f16=self._q_z_sr.astype(np.float16),
            blk_z_ee__Q16_f16=self._q_z_ee.astype(np.float16),
            blk_cd__Q16_f16=self._q_cd.astype(np.float16),
            blk_cb__Q16_f16=self._q_cb.astype(np.float16),
            blk_z_sr__mu0=mu_sr,
            blk_z_ee__mu0=mu_ee,
            blk_cd__mu0=mu_cd,
            blk_cb__mu0=mu_cb,
            rsub__mu_phi=self._mu_phi.astype(np.float32),
            rsub__sd_phi=self._sd_phi.astype(np.float32),
            rsub__U_run=self._u_run.astype(np.float32),
            cfg_lam0_16=np.array(self._lam0, dtype=np.float32),
            cfg_alpha_loc=np.array(self._alpha_loc, dtype=np.float32),
            cfg_gamma_sharp=np.array(self._gamma_sharp, dtype=np.float32),
            cfg_beta_margin=np.array(self._beta_margin, dtype=np.float32),
        )
        return [out_path]
