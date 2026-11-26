"""CBraMod 12-layer Criss-Cross Transformer causal readout foundation organ."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from neurips_bci.backbone.foundation_backbones.CbramodFoundationBackbone import CbramodFoundationBackbone
from neurips_bci.backbone.foundation_backbones.CudaGraphFoundationBackbone import CudaGraphFoundationBackbone
from neurips_bci.organ.foundation_organs.FoundationOrgan import FoundationOrgan
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class CbramodReadoutFoundationOrgan(FoundationOrgan):
    """Causal 200 Hz CBraMod organ with `CumCenter` / `pc_cent` + `BalStd(10.0)` readout and S3D `cb_u16_pre`."""

    def __init__(
        self,
        cbramod_backbone: CbramodFoundationBackbone,
        mu0: np.ndarray,
        w_eff: np.ndarray,
        b_eff: np.ndarray,
        pool_mode: str = "mean",
        n0_cent: float = 3.0,
        prior_bal: float = 10.0,
        centering_mode: str = "shipped",
        execution_device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            organ_identifier="cbramod_readout_foundation_organ",
            backbone=cbramod_backbone,
            execution_device=execution_device,
        )
        self._cbramod_backbone = cbramod_backbone
        self._mu0 = np.asarray(mu0, dtype=np.float64)
        self._w_eff = np.asarray(w_eff, dtype=np.float64)
        self._b_eff = np.asarray(b_eff, dtype=np.float64)
        self._mu0_w = self._mu0 @ self._w_eff
        self._w_eff_gpu = torch.as_tensor(
            np.ascontiguousarray(self._w_eff, dtype=np.float32), device=self._execution_device
        )
        self._pool_mode = str(pool_mode)
        self._n0_cent = float(n0_cent)
        self._prior_bal = float(prior_bal)
        self._centering_mode = str(centering_mode)

        self._s3d_q_gpu: torch.Tensor | None = None
        self._pin_fm: object = None
        self.reset_organ_state()

    def set_centering_mode(self, centering_mode: str) -> None:
        self._centering_mode = str(centering_mode)

    def attach_s3d_projection(self, q_cb_matrix: np.ndarray) -> None:
        self._s3d_q_gpu = torch.as_tensor(
            np.ascontiguousarray(q_cb_matrix, dtype=np.float32), device=self._execution_device
        )

    def reset_organ_state(self) -> None:
        self._n_seen = 0
        self._z_sum = np.zeros(3, dtype=np.float64)
        self._bal_sum = np.zeros(3, dtype=np.float64)
        self._pending_fm: tuple[torch.Tensor, torch.Tensor | None, bool] | None = None
        self._pc_u_hist: list[np.ndarray] = []
        self._pc_z_hist: list[np.ndarray] = []
        self._pc_z_sum = np.zeros(3, dtype=np.float64)
        self._pc_n_ok = 0
        self._pc_bal_sum = np.zeros(3, dtype=np.float64)
        self.last_s3d_u16: np.ndarray | None = None

    def _stage_to_device(self, x_in_f32: np.ndarray) -> torch.Tensor:
        if self._execution_device.type == "cuda" and x_in_f32.shape == (41, 800):
            if self._pin_fm is None:
                try:
                    self._pin_fm = torch.empty((1, 41, 800), dtype=torch.float32, pin_memory=True)
                except Exception:
                    self._pin_fm = False
            if self._pin_fm is not False:
                self._pin_fm[0].copy_(torch.from_numpy(np.ascontiguousarray(x_in_f32, dtype=np.float32)))  # type: ignore[index]
                return self._pin_fm.to(self._execution_device, non_blocking=True)  # type: ignore[union-attr]
        return torch.from_numpy(np.ascontiguousarray(x_in_f32, dtype=np.float32)[None]).to(self._execution_device)

    def _encode_core(self, xt: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        with torch.no_grad():
            tok = self._cbramod_backbone.encode_window_features(xt).float()
            u16_t: torch.Tensor | None = None
            if self._s3d_q_gpu is not None:
                cb_m = tok.mean(dim=2).reshape(-1)
                cb_l = tok[:, :, 1:, :].mean(dim=2).reshape(-1)
                cb_v = torch.cat([cb_m, 0.70 * cb_l], dim=0)
                u16_t = torch.matmul(cb_v, self._s3d_q_gpu)

            if self._pool_mode == "mean":
                vec_t = tok.mean(dim=2).reshape(-1)
            elif self._pool_mode == "late_rms":
                vec_t = torch.cat(
                    [tok[:, :, 1:, :].mean(dim=2), torch.sqrt((tok ** 2).mean(dim=2) + 1e-6)], dim=-1
                ).reshape(-1)
            else:
                vec_t = tok[:, :, 1:, :].mean(dim=2).reshape(-1)
            return torch.matmul(vec_t, self._w_eff_gpu), u16_t

    def _encode_vec_gpu(self, x_in_f32: np.ndarray) -> tuple[torch.Tensor, torch.Tensor | None]:
        xt = self._stage_to_device(x_in_f32)
        if tuple(xt.shape) == (1, 41, 800):
            cache_key = ("cb_fm", id(self._w_eff), id(self._s3d_q_gpu), self._pool_mode)
            replayed = CudaGraphFoundationBackbone.execute_cached_single_input_graph(
                self._cbramod_backbone, cache_key, self._encode_core, xt
            )
            if replayed is not None:
                assert replayed[0] is not None
                return replayed[0], replayed[1]
        z_t, u16_t = self._encode_core(xt)
        assert z_t is not None
        return z_t, u16_t

    def launch_gpu_async(self, prepared_200hz: np.ndarray, is_extreme: bool = False) -> None:
        z_t, u16_t = self._encode_vec_gpu(prepared_200hz)
        self._pending_fm = (z_t, u16_t, bool(is_extreme))

    def _step_pc_centered(
        self, z_raw: np.ndarray, pseudo_chrono_u: np.ndarray, is_artifact_flagged: bool, is_extreme: bool
    ) -> np.ndarray:
        is_ok = (not is_artifact_flagged) and (not is_extreme)
        u_cur = np.asarray(pseudo_chrono_u, dtype=np.float64)
        ok_f = 1.0 if is_ok else 0.0
        n_ok_cur = float(self._pc_n_ok) + ok_f
        z_sum_cur = self._pc_z_sum + ok_f * z_raw
        z_glob = (z_sum_cur / n_ok_cur) if n_ok_cur > 0.0 else self._mu0_w
        if self._pc_u_hist:
            u_mat = np.stack(self._pc_u_hist, axis=0)
            sq_d = np.sum((u_mat - u_cur[None, :]) ** 2, axis=1)
            w_p = np.exp(-0.5 * sq_d / (float(u_cur.shape[0]) * (1.3 * 1.3)))
            loc_sum = w_p @ np.stack(self._pc_z_hist, axis=0) + ok_f * z_raw
            loc_w = float(np.sum(w_p)) + ok_f
        else:
            loc_sum = ok_f * z_raw
            loc_w = ok_f
        cen = (self._n0_cent * self._mu0_w + 1.0 * z_glob + loc_sum) / (self._n0_cent + 1.0 + loc_w)
        logits = z_raw - cen + self._b_eff
        exp_z = np.exp(np.clip(logits - float(np.max(logits)), -60.0, 0.0))
        u_prob = exp_z / float(np.sum(exp_z))
        if is_ok:
            self._pc_n_ok += 1
            self._pc_z_sum = z_sum_cur
            self._pc_u_hist.append(u_cur.copy())
            self._pc_z_hist.append(z_raw.copy())
            self._pc_bal_sum = self._pc_bal_sum + u_prob
        p_bar = np.clip(
            (self._prior_bal / 3.0 + self._pc_bal_sum) / (self._prior_bal + float(self._pc_n_ok)), 1e-6, 1.0
        )
        q_out = u_prob / p_bar
        return q_out / float(np.sum(q_out))

    def step_prepared_200hz(
        self,
        prepared_200hz: np.ndarray,
        is_extreme: bool = False,
        pseudo_chrono_u: np.ndarray | None = None,
        is_artifact_flagged: bool = False,
    ) -> np.ndarray:
        if self._pending_fm is not None:
            z_t, u16_t, is_extreme = self._pending_fm
            self._pending_fm = None
        else:
            z_t, u16_t = self._encode_vec_gpu(prepared_200hz)

        z_raw = z_t.detach().cpu().numpy().astype(np.float64)
        if (not is_extreme) and (u16_t is not None):
            u16_np = u16_t.detach().cpu().numpy().astype(np.float64)
            self.last_s3d_u16 = u16_np if np.isfinite(u16_np).all() else None
        else:
            self.last_s3d_u16 = None

        if not np.isfinite(z_raw).all():
            self.last_s3d_u16 = None
            return np.full(3, 1.0 / 3.0, dtype=np.float64)

        if self._centering_mode == "pc_cent" and pseudo_chrono_u is not None:
            return self._step_pc_centered(z_raw, pseudo_chrono_u, bool(is_artifact_flagged), bool(is_extreme))

        if is_extreme:
            mu_w = (self._z_sum + self._n0_cent * self._mu0_w) / (float(self._n_seen) + self._n0_cent)
            logits = z_raw - mu_w + self._b_eff
            exp_z = np.exp(np.clip(logits - np.max(logits), -60.0, 0.0))
            u_prob = exp_z / np.sum(exp_z)
            p_bar = np.clip(
                (self._prior_bal / 3.0 + self._bal_sum) / (self._prior_bal + float(self._n_seen)), 1e-6, 1.0
            )
            q_out = u_prob / p_bar
            return q_out / np.sum(q_out)

        self._n_seen += 1
        self._z_sum += z_raw
        mu_w = (self._z_sum + self._n0_cent * self._mu0_w) / (float(self._n_seen) + self._n0_cent)
        logits = z_raw - mu_w + self._b_eff
        exp_z = np.exp(np.clip(logits - np.max(logits), -60.0, 0.0))
        u_prob = exp_z / np.sum(exp_z)
        self._bal_sum += u_prob
        p_bar = np.clip((self._prior_bal / 3.0 + self._bal_sum) / (self._prior_bal + float(self._n_seen)), 1e-6, 1.0)
        q_out = u_prob / p_bar
        return q_out / np.sum(q_out)

    def step_raw_window(
        self,
        raw_window: np.ndarray,
        pseudo_chrono_u: np.ndarray | None = None,
        is_artifact_flagged: bool = False,
    ) -> np.ndarray:
        prep_200, is_ext = self.prepare_200hz_car_window(raw_window)
        return self.step_prepared_200hz(
            prep_200,
            is_extreme=is_ext,
            pseudo_chrono_u=pseudo_chrono_u,
            is_artifact_flagged=is_artifact_flagged,
        )

    def step_causal_window(self, cleaned_window: np.ndarray, is_artifact_flagged: bool) -> np.ndarray:
        prep_200, is_ext = self.prepare_200hz_car_window(cleaned_window)
        return self.step_prepared_200hz(
            prep_200,
            is_extreme=(is_ext or bool(is_artifact_flagged)),
            is_artifact_flagged=is_artifact_flagged,
        )

    @classmethod
    def load_from_npz(
        cls,
        cbramod_npz_path: str | Path,
        centering_mode: str = "shipped",
        execution_device: str | torch.device = "cpu",
    ) -> CbramodReadoutFoundationOrgan:
        resolved_npz = ParameterTypeValidator.require_path(cbramod_npz_path, "cbramod_npz_path", must_exist=True)
        backbone = CbramodFoundationBackbone.from_npz_archive(resolved_npz, execution_device=execution_device)
        with np.load(resolved_npz, allow_pickle=False) as archive:
            mu0 = archive["head__mu0"].astype(np.float64)
            w_eff = archive["head__W_eff"].astype(np.float64)
            b_eff = archive["head__b_eff"].astype(np.float64)
            pool_mode = str(archive["head__pool_mode"].item())
            n0_cent = float(archive["head__n0_cent"].item())
            prior_bal = float(archive["head__prior_bal"].item())
        return cls(
            cbramod_backbone=backbone,
            mu0=mu0,
            w_eff=w_eff,
            b_eff=b_eff,
            pool_mode=pool_mode,
            n0_cent=n0_cent,
            prior_bal=prior_bal,
            centering_mode=centering_mode,
            execution_device=execution_device,
        )

    @classmethod
    def fit_from_corpus(
        cls,
        cbramod_backbone: CbramodFoundationBackbone,
        proteus_windows: np.ndarray,
        proteus_labels: np.ndarray,
        batch_size: int = 32,
        svd_components: int = 128,
        inverse_reg_c: float = 0.02,
        pool_mode: str = "mean",
        n0_cent: float = 3.0,
        prior_bal: float = 10.0,
        centering_mode: str = "shipped",
        execution_device: str | torch.device = "cpu",
    ) -> CbramodReadoutFoundationOrgan:
        dev = torch.device(execution_device)
        cbramod_backbone = cbramod_backbone.to(dev).eval()
        y_prot = np.asarray(proteus_labels, dtype=np.int64)
        x_200_all = np.stack([cls.prepare_200hz_car_window(win_mat)[0] for win_mat in proteus_windows], axis=0)
        feat_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for b_start in range(0, len(x_200_all), int(batch_size)):
                xb_t = torch.as_tensor(x_200_all[b_start : b_start + int(batch_size)], dtype=torch.float32, device=dev)
                tok_b = cbramod_backbone.encode_window_features(xb_t).float()
                if pool_mode == "mean":
                    vec_b = tok_b.mean(dim=2).reshape(len(xb_t), -1)
                elif pool_mode == "late_rms":
                    vec_b = torch.cat(
                        [tok_b[:, :, 1:, :].mean(dim=2), torch.sqrt((tok_b ** 2).mean(dim=2) + 1e-6)], dim=-1
                    ).reshape(len(xb_t), -1)
                else:
                    vec_b = tok_b[:, :, 1:, :].mean(dim=2).reshape(len(xb_t), -1)
                feat_chunks.append(vec_b.cpu().numpy().astype(np.float64))
        feat_mat = np.concatenate(feat_chunks, axis=0)
        mu0, w_eff, b_eff = cls.fit_collapsed_svd_logistic_readout(
            feat_mat, y_prot, svd_components=svd_components, inverse_reg_c=inverse_reg_c, w_eff_transposed=True
        )
        return cls(
            cbramod_backbone=cbramod_backbone,
            mu0=mu0,
            w_eff=w_eff,
            b_eff=b_eff,
            pool_mode=pool_mode,
            n0_cent=n0_cent,
            prior_bal=prior_bal,
            centering_mode=centering_mode,
            execution_device=dev,
        )

    def save_organ_artifacts(self, target_directory: Path) -> list[Path]:
        resolved_dir = ParameterTypeValidator.require_path(target_directory, "target_directory")
        resolved_dir.mkdir(parents=True, exist_ok=True)
        npz_payload = self._cbramod_backbone.export_encoder_npz_arrays()
        npz_payload["head__mu0"] = self._mu0.astype(np.float32)
        npz_payload["head__W_eff"] = self._w_eff.astype(np.float16)
        npz_payload["head__b_eff"] = self._b_eff.astype(np.float32)
        npz_payload["head__pool_mode"] = np.array(self._pool_mode)
        npz_payload["head__n0_cent"] = np.array(float(self._n0_cent), dtype=np.float32)
        npz_payload["head__prior_bal"] = np.array(float(self._prior_bal), dtype=np.float32)
        out_path = resolved_dir / "cbramod_41.npz"
        np.savez_compressed(out_path, **npz_payload)
        return [out_path]
