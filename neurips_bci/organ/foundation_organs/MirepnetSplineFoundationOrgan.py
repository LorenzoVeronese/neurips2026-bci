"""MIRepNet 45-channel spherical-spline + PseudoChrono-EA dual-readout foundation organ."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt
import torch

from neurips_bci.backbone.foundation_backbones.CudaGraphFoundationBackbone import CudaGraphFoundationBackbone
from neurips_bci.backbone.foundation_backbones.MirepnetFoundationBackbone import MirepnetFoundationBackbone
from neurips_bci.operator.matrix_operators.SubspaceMatrixOperator import SubspaceMatrixOperator
from neurips_bci.operator.signal_operators.SpectralSignalOperator import SpectralSignalOperator
from neurips_bci.organ.foundation_organs.FoundationOrgan import FoundationOrgan
from neurips_bci.organ.multimodal_organs.MultimodalOrgan import MultimodalOrgan
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class MirepnetSplineFoundationOrgan(FoundationOrgan):
    """Causal 250 Hz `8-30 Hz` `M_45_41` spline + `pc_45shr` PseudoChrono-EA + dual `mean256`/`logrms_std512` organ."""

    DEFAULT_MIREPNET_45_CHANNELS: tuple[str, ...] = (
        "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
        "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
        "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
        "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
        "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    )

    _CACHED_M250_8_30: np.ndarray | None = None

    def __init__(
        self,
        mirepnet_backbone: MirepnetFoundationBackbone,
        readout_spec: dict[str, Any],
        execution_device: str | torch.device = "cpu",
        max_history: int = 512,
    ) -> None:
        super().__init__(
            organ_identifier="mirepnet_spline_foundation_organ",
            backbone=mirepnet_backbone,
            execution_device=execution_device,
        )
        self._mirepnet_backbone = mirepnet_backbone
        self._readout_spec = readout_spec
        self._max_history = ParameterTypeValidator.require_integer(max_history, "max_history", 8)

        self._m_45_41 = np.asarray(readout_spec["M_45_41"], dtype=np.float64)
        self._r0_45 = np.asarray(readout_spec["R0_45"], dtype=np.float64)
        self._mu_phi = np.asarray(readout_spec["mu_phi"], dtype=np.float64)
        self._sd_phi = np.asarray(readout_spec["sd_phi"], dtype=np.float64)
        self._u_run = np.asarray(readout_spec["U_run"], dtype=np.float64)
        self._lam_shrink = float(readout_spec["lam_shrink"])
        self._n0_prior = float(readout_spec["n0_prior"])
        self._pc_tau = float(readout_spec["pc_tau"])
        self._n0_cent = float(readout_spec["n0_cent"])
        self._bal_prior = float(readout_spec["bal_prior"])

        m_mu0, m_w, m_b = readout_spec["ea45_m"]
        l_mu0, l_w, l_b = readout_spec["ea45_l"]
        self._m_mu0 = np.asarray(m_mu0, dtype=np.float64)
        self._m_w = np.asarray(m_w, dtype=np.float64)
        self._m_b = np.asarray(m_b, dtype=np.float64)
        self._l_mu0 = np.asarray(l_mu0, dtype=np.float64)
        self._l_w = np.asarray(l_w, dtype=np.float64)
        self._l_b = np.asarray(l_b, dtype=np.float64)

        self._eye45 = np.eye(45, dtype=np.float64)
        self._pin_mr: object = None
        self.reset_organ_state()

    def reset_organ_state(self) -> None:
        self._n_seen = 0
        self._m_sum = np.zeros(256, dtype=np.float64)
        self._l_sum = np.zeros(512, dtype=np.float64)
        self._m_bal = np.zeros(3, dtype=np.float64)
        self._l_bal = np.zeros(3, dtype=np.float64)
        self._u_hist: list[np.ndarray] = []
        self._cov_buf: np.ndarray | None = None
        self._n_buf = 0
        self._pending_mr: tuple[torch.Tensor, bool] | None = None

    @classmethod
    def _get_m250_8_30_operator(cls) -> np.ndarray:
        if cls._CACHED_M250_8_30 is None:
            sos_8_30 = butter(4, (8.0, 30.0), btype="bandpass", fs=250.0, output="sos")
            centered_eye480 = np.eye(480, dtype=np.float64) - (1.0 / 480.0)
            cls._CACHED_M250_8_30 = sosfiltfilt(
                sos_8_30, resample_poly(centered_eye480, 25, 12, axis=-1), axis=-1
            )
        return cls._CACHED_M250_8_30

    def _sym_inv_sqrt_shrunk(self, cov_45: np.ndarray) -> np.ndarray:
        sym_r = 0.5 * (cov_45 + cov_45.T)
        tr_bar = float(np.trace(sym_r)) / 45.0
        r_reg = (1.0 - self._lam_shrink) * sym_r + self._lam_shrink * max(tr_bar, 1e-12) * self._eye45
        eig_vals, eig_vecs = np.linalg.eigh(r_reg)
        floor_v = max(1e-12, 1e-8 * float(np.max(eig_vals)))
        inv_s = 1.0 / np.sqrt(np.maximum(eig_vals, floor_v))
        return (eig_vecs * inv_s[None, :]) @ eig_vecs.T

    def _prepare_ea45_window(
        self,
        raw_window: np.ndarray,
        is_artifact_flagged: bool = False,
        phi_nm_pre: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool]:
        is_extreme = (
            bool(is_artifact_flagged)
            or (not np.isfinite(raw_window).all())
            or bool(np.max(np.abs(raw_window)) > 5000.0)
        )
        source_f64 = np.asarray(
            np.nan_to_num(np.clip(raw_window, -5000.0, 5000.0), nan=0.0, posinf=5000.0, neginf=-5000.0)
            if is_extreme
            else raw_window,
            dtype=np.float64,
        )
        if source_f64.shape != (41, 480):
            return np.zeros((45, 1000), dtype=np.float32), True

        x250_f32 = (source_f64 @ self._get_m250_8_30_operator()).astype(np.float32)
        x45 = self._m_45_41 @ x250_f32.astype(np.float64)
        cov45 = (x45 @ x45.T) / 1000.0
        phi_vec = (
            phi_nm_pre
            if phi_nm_pre is not None
            else SpectralSignalOperator.extract_non_motor_246_features(source_f64)
        )
        u_t = ((phi_vec - self._mu_phi) / self._sd_phi) @ self._u_run
        if not np.isfinite(u_t).all():
            u_t = np.zeros(self._u_run.shape[1], dtype=np.float64)
            is_extreme = True

        if not is_extreme:
            if self._cov_buf is None or self._cov_buf.shape[0] != self._max_history:
                self._cov_buf = np.empty((self._max_history, 45, 45), dtype=np.float64)
                self._n_buf = 0
            self._u_hist.append(u_t)
            if self._n_buf < self._max_history:
                self._cov_buf[self._n_buf] = cov45
                self._n_buf += 1
            else:
                self._cov_buf[:-1] = self._cov_buf[1:]
                self._cov_buf[-1] = cov45
            if len(self._u_hist) > self._max_history:
                self._u_hist.pop(0)

        if self._n_buf > 0 and self._cov_buf is not None:
            u_mat = np.stack(self._u_hist, axis=0)
            d2 = np.sum((u_mat - u_t[None, :]) ** 2, axis=1)
            wt = np.exp(-0.5 * d2 / (float(u_mat.shape[1]) * (self._pc_tau ** 2)))
            r_t = (self._n0_prior * self._r0_45 + np.tensordot(wt, self._cov_buf[: self._n_buf], axes=(0, 0))) / (
                self._n0_prior + float(np.sum(wt))
            )
        else:
            r_t = self._r0_45

        w_ea = self._sym_inv_sqrt_shrunk(r_t)
        x_ea = (w_ea @ x45).astype(np.float32)
        if not np.isfinite(x_ea).all():
            x_ea = np.nan_to_num(x_ea, nan=0.0, posinf=50.0, neginf=-50.0).astype(np.float32)
            is_extreme = True
        return x_ea, is_extreme

    def _encode_core(self, xt: torch.Tensor) -> tuple[torch.Tensor | None]:
        with torch.no_grad():
            tok, feat = self._mirepnet_backbone.forward_all(xt)
            t0 = tok[0].float()
            lrms = torch.log(torch.sqrt((t0 * t0).mean(dim=0) + 1e-12) + 1e-6)
            lstd = torch.log(t0.std(dim=0, unbiased=False) + 1e-6)
            return (torch.cat([feat[0].float(), lrms, lstd], dim=0),)

    def _encode_gpu(self, x_ea45: np.ndarray) -> torch.Tensor:
        if self._execution_device.type == "cuda" and x_ea45.shape == (45, 1000):
            if self._pin_mr is None:
                try:
                    self._pin_mr = torch.empty((1, 45, 1000), dtype=torch.float32, pin_memory=True)
                except Exception:
                    self._pin_mr = False
            if self._pin_mr is not False:
                self._pin_mr[0].copy_(torch.from_numpy(np.ascontiguousarray(x_ea45, dtype=np.float32)))  # type: ignore[index]
                xt = self._pin_mr.to(self._execution_device, non_blocking=True)  # type: ignore[union-attr]
            else:
                xt = torch.from_numpy(np.ascontiguousarray(x_ea45, dtype=np.float32)[None]).to(self._execution_device)
        else:
            xt = torch.from_numpy(np.ascontiguousarray(x_ea45, dtype=np.float32)[None]).to(self._execution_device)

        if tuple(xt.shape) == (1, 45, 1000):
            replayed = CudaGraphFoundationBackbone.execute_cached_single_input_graph(
                self._mirepnet_backbone, ("mr",), self._encode_core, xt
            )
            if replayed is not None and replayed[0] is not None:
                return replayed[0]
        eager_out = self._encode_core(xt)[0]
        assert eager_out is not None
        return eager_out

    def launch_gpu_async(
        self,
        raw_window: np.ndarray,
        is_artifact_flagged: bool = False,
        phi_nm_pre: np.ndarray | None = None,
    ) -> None:
        x_ea45, is_extreme = self._prepare_ea45_window(
            raw_window, is_artifact_flagged=is_artifact_flagged, phi_nm_pre=phi_nm_pre
        )
        self._pending_mr = (self._encode_gpu(x_ea45), bool(is_extreme))

    def _score_readout_branch(
        self,
        f_vec: np.ndarray,
        f_sum: np.ndarray,
        bal_sum: np.ndarray,
        mu0: np.ndarray,
        w_mat: np.ndarray,
        b_vec: np.ndarray,
        n_eff: int,
        should_update: bool,
    ) -> np.ndarray:
        if should_update:
            f_sum += f_vec
        cen = (f_sum + self._n0_cent * mu0) / (float(n_eff) + self._n0_cent)
        logits = (f_vec - cen) @ w_mat.T + b_vec
        exp_z = np.exp(np.clip(logits - np.max(logits), -60.0, 0.0))
        u_prob = exp_z / float(np.sum(exp_z))
        if should_update:
            bal_sum += u_prob
        p_bar = np.clip((self._bal_prior / 3.0 + bal_sum) / (self._bal_prior + float(n_eff)), 1e-6, 1.0)
        q_out = u_prob / p_bar
        return q_out / float(np.sum(q_out))

    def step_score(
        self,
        raw_window: np.ndarray,
        is_artifact_flagged: bool = False,
        phi_nm_pre: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._pending_mr is not None:
            vec_t, is_extreme = self._pending_mr
            self._pending_mr = None
        else:
            x_ea45, is_extreme = self._prepare_ea45_window(
                raw_window, is_artifact_flagged=is_artifact_flagged, phi_nm_pre=phi_nm_pre
            )
            vec_t = self._encode_gpu(x_ea45)

        v768 = vec_t.detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(v768).all():
            return np.full(3, 1.0 / 3.0, dtype=np.float64)

        fm = v768[:256]
        fl = v768[256:]
        if not is_extreme:
            self._n_seen += 1

        p_m = self._score_readout_branch(
            fm, self._m_sum, self._m_bal, self._m_mu0, self._m_w, self._m_b, self._n_seen, should_update=(not is_extreme)
        )
        p_l = self._score_readout_branch(
            fl, self._l_sum, self._l_bal, self._l_mu0, self._l_w, self._l_b, self._n_seen, should_update=(not is_extreme)
        )
        lp_m = np.log(np.clip(p_m, 1e-12, 1.0))
        lp_l = np.log(np.clip(p_l, 1e-12, 1.0))
        z_combo = 0.5 * (lp_m - float(np.mean(lp_m))) + 0.5 * (lp_l - float(np.mean(lp_l)))
        exp_c = np.exp(z_combo - float(np.max(z_combo)))
        return exp_c / float(np.sum(exp_c))

    def step_causal_window(self, cleaned_window: np.ndarray, is_artifact_flagged: bool) -> np.ndarray:
        return self.step_score(cleaned_window, is_artifact_flagged=is_artifact_flagged)

    @classmethod
    def load_from_npz(
        cls,
        mirepnet_fp16_npz_path: str | Path,
        readouts_npz_path: str | Path,
        execution_device: str | torch.device = "cpu",
        max_history: int = 512,
    ) -> MirepnetSplineFoundationOrgan:
        resolved_rd = ParameterTypeValidator.require_path(readouts_npz_path, "readouts_npz_path", must_exist=True)
        backbone = MirepnetFoundationBackbone.from_fp16_npz(mirepnet_fp16_npz_path, execution_device=execution_device)
        with np.load(resolved_rd, allow_pickle=False) as archive:
            spec: dict[str, Any] = {
                "M_45_41": archive["M_45_41"].astype(np.float64),
                "R0_45": archive["R0_45"].astype(np.float64),
                "mu_phi": archive["rsub__mu_phi"].astype(np.float64),
                "sd_phi": archive["rsub__sd_phi"].astype(np.float64),
                "U_run": archive["rsub__U_run"].astype(np.float64),
                "lam_shrink": float(archive["cfg_lam_shrink"]),
                "n0_prior": float(archive["cfg_n0_prior"]),
                "pc_tau": float(archive["cfg_pc_tau"]),
                "n0_cent": float(archive["cfg_n0_cent"]),
                "bal_prior": float(archive["cfg_bal_prior"]),
                "ea45_m": (
                    archive["ea45__mean256__mu0"].astype(np.float64),
                    archive["ea45__mean256__W_eff"].astype(np.float64),
                    archive["ea45__mean256__b_eff"].astype(np.float64),
                ),
                "ea45_l": (
                    archive["ea45__logrms_std512__mu0"].astype(np.float64),
                    archive["ea45__logrms_std512__W_eff"].astype(np.float64),
                    archive["ea45__logrms_std512__b_eff"].astype(np.float64),
                ),
            }
        return cls(
            mirepnet_backbone=backbone,
            readout_spec=spec,
            execution_device=execution_device,
            max_history=max_history,
        )

    @classmethod
    def fit_from_corpus(
        cls,
        mirepnet_backbone: MirepnetFoundationBackbone,
        proteus_windows: np.ndarray,
        proteus_labels: np.ndarray,
        session_ids: np.ndarray,
        source_41_channels: tuple[str, ...] = SubspaceMatrixOperator.PROTEUS_41_CHANNELS,
        target_45_channels: tuple[str, ...] = DEFAULT_MIREPNET_45_CHANNELS,
        lam_shrink: float = 0.08,
        n0_prior: float = 12.0,
        pc_tau: float = 1.3,
        n0_cent: float = 6.0,
        bal_prior: float = 10.0,
        svd_components: int = 128,
        inverse_reg_c: float = 0.05,
        batch_size: int = 32,
        execution_device: str | torch.device = "cpu",
    ) -> MirepnetSplineFoundationOrgan:
        """Fit `M_45_41` spherical spline, `R0_45`, `rsub__*` run subspace, and dual `mean256`+`logrms_std512` heads."""
        dev = torch.device(execution_device)
        mirepnet_backbone = mirepnet_backbone.to(dev).eval()
        mirepnet_backbone.freeze_fused_stem()

        x_all = np.asarray(proteus_windows, dtype=np.float64)
        y_all = np.asarray(proteus_labels, dtype=np.int64)
        sess_all = np.asarray(session_ids)

        m_45_41 = SubspaceMatrixOperator.build_spherical_spline_mapping(
            source_channel_labels=source_41_channels,
            target_channel_labels=target_45_channels,
        )
        mu_phi, sd_phi, u_run = MultimodalOrgan.fit_non_motor_run_subspace(
            x_all, sess_all, subspace_dim=6, within_shrinkage=0.25
        )
        m250_op = cls._get_m250_8_30_operator()

        cov45_list: list[np.ndarray] = []
        x45_list: list[np.ndarray] = []
        for win_i in x_all:
            x250 = (win_i @ m250_op).astype(np.float32)
            x45 = m_45_41 @ x250.astype(np.float64)
            x45_list.append(x45)
            cov45_list.append((x45 @ x45.T) / 1000.0)
        r0_45 = np.mean(cov45_list, axis=0)

        eye45 = np.eye(45, dtype=np.float64)

        def _inv_sqrt_45(r_mat: np.ndarray) -> np.ndarray:
            rs = 0.5 * (r_mat + r_mat.T)
            tr_b = float(np.trace(rs)) / 45.0
            r_reg = (1.0 - float(lam_shrink)) * rs + float(lam_shrink) * max(tr_b, 1e-12) * eye45
            ev, vc = np.linalg.eigh(r_reg)
            fl = max(1e-12, 1e-8 * float(np.max(ev)))
            return (vc * (1.0 / np.sqrt(np.maximum(ev, fl)))[None, :]) @ vc.T

        x_ea_all = np.empty((len(x_all), 45, 1000), dtype=np.float32)
        for sess_k in np.unique(sess_all):
            s_idx = np.flatnonzero(sess_all == sess_k)
            r_sess = (float(n0_prior) * r0_45 + np.sum([cov45_list[i_w] for i_w in s_idx], axis=0)) / (
                float(n0_prior) + float(len(s_idx))
            )
            w_sess = _inv_sqrt_45(r_sess)
            for i_w in s_idx:
                x_ea_all[i_w] = (w_sess @ x45_list[i_w]).astype(np.float32)

        m_feats_list, l_feats_list = [], []
        with torch.no_grad():
            for b_start in range(0, len(x_ea_all), int(batch_size)):
                xb_t = torch.as_tensor(x_ea_all[b_start : b_start + int(batch_size)], dtype=torch.float32, device=dev)
                tok_b, feat_b = mirepnet_backbone.forward_all(xb_t)
                tf = tok_b.float()
                lrms = torch.log(torch.sqrt((tf * tf).mean(dim=1) + 1e-12) + 1e-6)
                lstd = torch.log(tf.std(dim=1, unbiased=False) + 1e-6)
                m_feats_list.append(feat_b.float().cpu().numpy().astype(np.float64))
                l_feats_list.append(torch.cat([lrms, lstd], dim=-1).cpu().numpy().astype(np.float64))

        m_mat = np.concatenate(m_feats_list, axis=0)
        l_mat = np.concatenate(l_feats_list, axis=0)
        m_mu0, m_w_eff, m_b_eff = cls.fit_collapsed_svd_logistic_readout(
            m_mat, y_all, svd_components=min(int(svd_components), 256), inverse_reg_c=inverse_reg_c, w_eff_transposed=False
        )
        l_mu0, l_w_eff, l_b_eff = cls.fit_collapsed_svd_logistic_readout(
            l_mat, y_all, svd_components=int(svd_components), inverse_reg_c=inverse_reg_c, w_eff_transposed=False
        )

        spec: dict[str, Any] = {
            "M_45_41": m_45_41,
            "R0_45": r0_45,
            "mu_phi": mu_phi,
            "sd_phi": sd_phi,
            "U_run": u_run,
            "lam_shrink": float(lam_shrink),
            "n0_prior": float(n0_prior),
            "pc_tau": float(pc_tau),
            "n0_cent": float(n0_cent),
            "bal_prior": float(bal_prior),
            "ea45_m": (m_mu0, m_w_eff, m_b_eff),
            "ea45_l": (l_mu0, l_w_eff, l_b_eff),
        }
        return cls(mirepnet_backbone=mirepnet_backbone, readout_spec=spec, execution_device=dev)

    def save_organ_artifacts(self, target_directory: Path) -> list[Path]:
        resolved_dir = ParameterTypeValidator.require_path(target_directory, "target_directory")
        resolved_dir.mkdir(parents=True, exist_ok=True)
        fp16_path = self._mirepnet_backbone.export_fp16_npz(resolved_dir / "mirepnet_fp16.npz")
        rd_path = resolved_dir / "s3c_mirepnet_readouts_2655.npz"
        np.savez_compressed(
            rd_path,
            M_45_41=self._m_45_41.astype(np.float32),
            R0_45=self._r0_45.astype(np.float32),
            rsub__mu_phi=self._mu_phi.astype(np.float32),
            rsub__sd_phi=self._sd_phi.astype(np.float32),
            rsub__U_run=self._u_run.astype(np.float32),
            cfg_lam_shrink=np.array(self._lam_shrink, dtype=np.float32),
            cfg_n0_prior=np.array(self._n0_prior, dtype=np.float32),
            cfg_pc_tau=np.array(self._pc_tau, dtype=np.float32),
            cfg_n0_cent=np.array(self._n0_cent, dtype=np.float32),
            cfg_bal_prior=np.array(self._bal_prior, dtype=np.float32),
            ea45__mean256__mu0=self._m_mu0.astype(np.float32),
            ea45__mean256__W_eff=self._m_w.astype(np.float16),
            ea45__mean256__b_eff=self._m_b.astype(np.float32),
            ea45__logrms_std512__mu0=self._l_mu0.astype(np.float32),
            ea45__logrms_std512__W_eff=self._l_w.astype(np.float16),
            ea45__logrms_std512__b_eff=self._l_b.astype(np.float32),
        )
        return [fp16_path, rd_path]
