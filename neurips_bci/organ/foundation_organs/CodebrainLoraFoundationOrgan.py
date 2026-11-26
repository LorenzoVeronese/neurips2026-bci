"""CodeBrain SGConv+SWA foundation organ with frozen readout and 2-seed Rank-8 LoRA tails."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np
import torch
import torch.nn.functional as F

from neurips_bci.backbone.foundation_backbones.CodebrainFoundationBackbone import CodebrainFoundationBackbone
from neurips_bci.backbone.foundation_backbones.CudaGraphFoundationBackbone import CudaGraphFoundationBackbone
from neurips_bci.organ.foundation_organs.FoundationOrgan import FoundationOrgan
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class CodebrainLoraFoundationOrgan(FoundationOrgan):
    """Causal 200 Hz CodeBrain organ with frozen `late`/`late_rms` head, 2-seed Rank-8 LoRA `s2c`, and S3D `u16`."""

    def __init__(
        self,
        codebrain_backbone: CodebrainFoundationBackbone,
        frozen_mu0: np.ndarray,
        frozen_w_eff: np.ndarray,
        frozen_b_eff: np.ndarray,
        pool_mode: str = "late",
        n0_cent: float = 3.0,
        prior_bal: float = 10.0,
        lora_readouts: Sequence[dict[str, np.ndarray]] | None = None,
        lora_n0_cent: float = 3.0,
        lora_prior_bal: float = 10.0,
        execution_device: str | torch.device = "cpu",
        exported_lora_npz_arrays: dict[str, np.ndarray] | None = None,
    ) -> None:
        super().__init__(
            organ_identifier="codebrain_lora_foundation_organ",
            backbone=codebrain_backbone,
            execution_device=execution_device,
        )
        self._codebrain_backbone = codebrain_backbone
        self._frozen_mu0 = np.asarray(frozen_mu0, dtype=np.float64)
        self._frozen_w_eff = np.asarray(frozen_w_eff, dtype=np.float64)
        self._frozen_b_eff = np.asarray(frozen_b_eff, dtype=np.float64)
        self._frozen_mu0_w = self._frozen_mu0 @ self._frozen_w_eff
        self._frozen_w_gpu = torch.as_tensor(
            np.ascontiguousarray(self._frozen_w_eff, dtype=np.float32), device=self._execution_device
        )
        self._pool_mode = str(pool_mode)
        self._n0_cent = float(n0_cent)
        self._prior_bal = float(prior_bal)

        self._lora_specs: list[dict[str, object]] = []
        for rd_item in lora_readouts or ():
            mu_s = np.asarray(rd_item["mu0"], dtype=np.float64)
            w_s = np.asarray(rd_item["W_eff"], dtype=np.float64)
            b_s = np.asarray(rd_item["b_eff"], dtype=np.float64)
            self._lora_specs.append(
                {
                    "mu0": mu_s,
                    "W_eff": w_s,
                    "W_eff_gpu": torch.as_tensor(
                        np.ascontiguousarray(w_s, dtype=np.float32), device=self._execution_device
                    ),
                    "mu0_W": mu_s @ w_s,
                    "b_eff": b_s,
                }
            )
        self._lora_n0_cent = float(lora_n0_cent)
        self._lora_prior_bal = float(lora_prior_bal)
        self._exported_lora_npz_arrays = exported_lora_npz_arrays

        self._s3d_q_gpu: torch.Tensor | None = None
        self._pin_fm: object = None
        self.reset_organ_state()

    @property
    def has_lora_tails(self) -> bool:
        return len(self._lora_specs) > 0

    def attach_s3d_projection(self, q_cd_matrix: np.ndarray) -> None:
        self._s3d_q_gpu = torch.as_tensor(
            np.ascontiguousarray(q_cd_matrix, dtype=np.float32), device=self._execution_device
        )

    def reset_organ_state(self) -> None:
        self._n_seen = 0
        self._z_sum = np.zeros(3, dtype=np.float64)
        self._bal_sum = np.zeros(3, dtype=np.float64)
        self._lora_z_sums = [np.zeros(3, dtype=np.float64) for _ in self._lora_specs]
        self._lora_bal_sums = [np.zeros(3, dtype=np.float64) for _ in self._lora_specs]
        self._pending_fm: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool] | None = None
        self.last_s3d_u16: np.ndarray | None = None
        self.last_p_s2c: np.ndarray | None = None

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

    def _encode_core(
        self, xt: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        with torch.no_grad():
            frozen_tok, lora_tok_list = self._codebrain_backbone.encode_window_features(xt)
            tf = frozen_tok.float()
            s2c_z_t: torch.Tensor | None = None
            if self._lora_specs and len(lora_tok_list) == len(self._lora_specs):
                z_s2c_parts: list[torch.Tensor] = []
                for tail_tokens, tail_spec in zip(lora_tok_list, self._lora_specs):
                    tfs = tail_tokens.float()
                    vs = torch.cat(
                        [tfs[:, :, 1:, :].mean(dim=2), torch.sqrt((tfs ** 2).mean(dim=2) + 1e-6)],
                        dim=-1,
                    ).reshape(-1)
                    z_s2c_parts.append(torch.matmul(vs, tail_spec["W_eff_gpu"]))  # type: ignore[arg-type]
                s2c_z_t = torch.stack(z_s2c_parts, dim=0)

            u16_t: torch.Tensor | None = None
            if self._s3d_q_gpu is not None:
                cd_l = tf[:, :, 1:, :].mean(dim=2).reshape(-1)
                cd_e = tf[:, :, :1, :].mean(dim=2).reshape(-1)
                cd_v = torch.cat([cd_l, 0.70 * (cd_l - cd_e)], dim=0)
                u16_t = torch.matmul(cd_v, self._s3d_q_gpu)

            if self._pool_mode == "mean":
                vec_t = tf.mean(dim=2).reshape(-1)
            elif self._pool_mode == "late_rms":
                vec_t = torch.cat(
                    [tf[:, :, 1:, :].mean(dim=2), torch.sqrt((tf ** 2).mean(dim=2) + 1e-6)], dim=-1
                ).reshape(-1)
            else:
                vec_t = tf[:, :, 1:, :].mean(dim=2).reshape(-1)

            return torch.matmul(vec_t, self._frozen_w_gpu), u16_t, s2c_z_t

    def _encode_vec_gpu(
        self, x_in_f32: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        xt = self._stage_to_device(x_in_f32)
        if tuple(xt.shape) == (1, 41, 800):
            cache_key = ("cd_fm", id(self._frozen_w_eff), id(self._s3d_q_gpu), len(self._lora_specs), self._pool_mode)
            replayed = CudaGraphFoundationBackbone.execute_cached_single_input_graph(
                self._codebrain_backbone, cache_key, self._encode_core, xt
            )
            if replayed is not None:
                assert replayed[0] is not None
                return replayed[0], replayed[1], replayed[2]
        z_t, u16_t, s2c_t = self._encode_core(xt)
        assert z_t is not None
        return z_t, u16_t, s2c_t

    def launch_gpu_async(self, prepared_200hz: np.ndarray, is_extreme: bool = False) -> None:
        z_t, u16_t, s2c_z_t = self._encode_vec_gpu(prepared_200hz)
        self._pending_fm = (z_t, u16_t, s2c_z_t, bool(is_extreme))

    def step_prepared_200hz(self, prepared_200hz: np.ndarray, is_extreme: bool = False) -> np.ndarray:
        if self._pending_fm is not None:
            z_t, u16_t, s2c_z_t, is_extreme = self._pending_fm
            self._pending_fm = None
        else:
            z_t, u16_t, s2c_z_t = self._encode_vec_gpu(prepared_200hz)

        z_raw = z_t.detach().cpu().numpy().astype(np.float64)
        if (not is_extreme) and (u16_t is not None):
            u16_np = u16_t.detach().cpu().numpy().astype(np.float64)
            self.last_s3d_u16 = u16_np if np.isfinite(u16_np).all() else None
        else:
            self.last_s3d_u16 = None

        s2c_raw = s2c_z_t.detach().cpu().numpy().astype(np.float64) if s2c_z_t is not None else None
        if not np.isfinite(z_raw).all():
            self.last_s3d_u16 = None
            self.last_p_s2c = np.full(3, 1.0 / 3.0, dtype=np.float64) if self._lora_specs else None
            return np.full(3, 1.0 / 3.0, dtype=np.float64)

        if is_extreme:
            mu_w = (self._z_sum + self._n0_cent * self._frozen_mu0_w) / (float(self._n_seen) + self._n0_cent)
            logits = z_raw - mu_w + self._frozen_b_eff
            exp_z = np.exp(np.clip(logits - np.max(logits), -60.0, 0.0))
            u_frz = exp_z / np.sum(exp_z)
            p_bar = np.clip(
                (self._prior_bal / 3.0 + self._bal_sum) / (self._prior_bal + float(self._n_seen)), 1e-6, 1.0
            )
            q_frz = u_frz / p_bar
            if self._lora_specs and s2c_raw is not None and np.isfinite(s2c_raw).all():
                lp_acc = np.zeros(3, dtype=np.float64)
                for s_i, tail_spec in enumerate(self._lora_specs):
                    mw_s = (self._lora_z_sums[s_i] + self._lora_n0_cent * tail_spec["mu0_W"]) / (  # type: ignore[operator]
                        float(self._n_seen) + self._lora_n0_cent
                    )
                    zs = s2c_raw[s_i] - mw_s + tail_spec["b_eff"]  # type: ignore[operator]
                    es = np.exp(np.clip(zs - np.max(zs), -60.0, 0.0))
                    us = es / np.sum(es)
                    pbs = np.clip(
                        (self._lora_prior_bal / 3.0 + self._lora_bal_sums[s_i])
                        / (self._lora_prior_bal + float(self._n_seen)),
                        1e-6,
                        1.0,
                    )
                    qs = us / pbs
                    qs /= np.sum(qs)
                    lps = np.log(np.clip(qs, 1e-12, 1.0))
                    lp_acc += (lps - np.mean(lps)) / float(len(self._lora_specs))
                es2 = np.exp(np.clip(lp_acc - np.max(lp_acc), -60.0, 0.0))
                self.last_p_s2c = es2 / np.sum(es2)
            else:
                self.last_p_s2c = np.full(3, 1.0 / 3.0, dtype=np.float64) if self._lora_specs else None
            return q_frz / np.sum(q_frz)

        self._n_seen += 1
        self._z_sum += z_raw
        mu_w = (self._z_sum + self._n0_cent * self._frozen_mu0_w) / (float(self._n_seen) + self._n0_cent)
        logits = z_raw - mu_w + self._frozen_b_eff
        exp_z = np.exp(np.clip(logits - np.max(logits), -60.0, 0.0))
        u_frz = exp_z / np.sum(exp_z)
        self._bal_sum += u_frz
        p_bar = np.clip((self._prior_bal / 3.0 + self._bal_sum) / (self._prior_bal + float(self._n_seen)), 1e-6, 1.0)
        q_frz = u_frz / p_bar

        if self._lora_specs and s2c_raw is not None and np.isfinite(s2c_raw).all():
            lp_acc = np.zeros(3, dtype=np.float64)
            for s_i, tail_spec in enumerate(self._lora_specs):
                self._lora_z_sums[s_i] += s2c_raw[s_i]
                mw_s = (self._lora_z_sums[s_i] + self._lora_n0_cent * tail_spec["mu0_W"]) / (  # type: ignore[operator]
                    float(self._n_seen) + self._lora_n0_cent
                )
                zs = s2c_raw[s_i] - mw_s + tail_spec["b_eff"]  # type: ignore[operator]
                es = np.exp(np.clip(zs - np.max(zs), -60.0, 0.0))
                us = es / np.sum(es)
                self._lora_bal_sums[s_i] += us
                pbs = np.clip(
                    (self._lora_prior_bal / 3.0 + self._lora_bal_sums[s_i])
                    / (self._lora_prior_bal + float(self._n_seen)),
                    1e-6,
                    1.0,
                )
                qs = us / pbs
                qs /= np.sum(qs)
                lps = np.log(np.clip(qs, 1e-12, 1.0))
                lp_acc += (lps - np.mean(lps)) / float(len(self._lora_specs))
            es2 = np.exp(np.clip(lp_acc - np.max(lp_acc), -60.0, 0.0))
            self.last_p_s2c = es2 / np.sum(es2)
        else:
            self.last_p_s2c = np.full(3, 1.0 / 3.0, dtype=np.float64) if self._lora_specs else None

        return q_frz / np.sum(q_frz)

    def step_raw_window(self, raw_window: np.ndarray) -> np.ndarray:
        prep_200, is_ext = self.prepare_200hz_car_window(raw_window)
        return self.step_prepared_200hz(prep_200, is_extreme=is_ext)

    def step_causal_window(self, cleaned_window: np.ndarray, is_artifact_flagged: bool) -> np.ndarray:
        prep_200, is_ext = self.prepare_200hz_car_window(cleaned_window)
        frz_p = self.step_prepared_200hz(prep_200, is_extreme=(is_ext or bool(is_artifact_flagged)))
        return self.last_p_s2c if self.last_p_s2c is not None else frz_p

    @classmethod
    def load_from_npz(
        cls,
        codebrain_npz_path: str | Path,
        lora_npz_path: str | Path | None = None,
        execution_device: str | torch.device = "cpu",
    ) -> CodebrainLoraFoundationOrgan:
        backbone, frozen_meta, lora_readouts, lora_n0_cent, lora_prior_bal = (
            CodebrainFoundationBackbone.from_npz_archive(
                codebrain_npz_path=codebrain_npz_path,
                execution_device=execution_device,
                lora_npz_path=lora_npz_path,
            )
        )
        return cls(
            codebrain_backbone=backbone,
            frozen_mu0=frozen_meta["mu0"],  # type: ignore[arg-type]
            frozen_w_eff=frozen_meta["W_eff"],  # type: ignore[arg-type]
            frozen_b_eff=frozen_meta["b_eff"],  # type: ignore[arg-type]
            pool_mode=str(frozen_meta["pool_mode"]),
            n0_cent=float(frozen_meta["n0_cent"]),  # type: ignore[arg-type]
            prior_bal=float(frozen_meta["prior_bal"]),  # type: ignore[arg-type]
            lora_readouts=lora_readouts,
            lora_n0_cent=lora_n0_cent,
            lora_prior_bal=lora_prior_bal,
            execution_device=execution_device,
        )

    @classmethod
    def fit_from_corpora(
        cls,
        codebrain_backbone: CodebrainFoundationBackbone,
        proteus_windows: np.ndarray,
        proteus_labels: np.ndarray,
        lora_seed_count: int = 2,
        lora_rank: int = 8,
        lora_epochs: int = 4,
        batch_size: int = 32,
        learning_rate: float = 3e-4,
        svd_components: int = 128,
        inverse_reg_c: float = 0.02,
        pool_mode: str = "late",
        n0_cent: float = 3.0,
        prior_bal: float = 10.0,
        random_seed: int = 2655,
        execution_device: str | torch.device = "cpu",
    ) -> CodebrainLoraFoundationOrgan:
        """Fit frozen readout (`codebrain_41.npz`) and Rank-`lora_rank` LoRA tails (`s2c_codebrain_lora8_2655.npz`)."""
        dev = torch.device(execution_device)
        codebrain_backbone.finalize_frozen_stem_and_tail(dev)
        y_prot = np.asarray(proteus_labels, dtype=np.int64)
        prepped_list = [cls.prepare_200hz_car_window(win_mat)[0] for win_mat in proteus_windows]
        x_200_all = np.stack(prepped_list, axis=0)

        frz_feats: list[np.ndarray] = []
        with torch.no_grad():
            for b_start in range(0, len(x_200_all), int(batch_size)):
                xb_t = torch.as_tensor(x_200_all[b_start : b_start + int(batch_size)], dtype=torch.float32, device=dev)
                tok_b = codebrain_backbone.forward_frozen_only(xb_t).float()
                if pool_mode == "mean":
                    vec_b = tok_b.mean(dim=2).reshape(len(xb_t), -1)
                elif pool_mode == "late_rms":
                    vec_b = torch.cat(
                        [tok_b[:, :, 1:, :].mean(dim=2), torch.sqrt((tok_b ** 2).mean(dim=2) + 1e-6)], dim=-1
                    ).reshape(len(xb_t), -1)
                else:
                    vec_b = tok_b[:, :, 1:, :].mean(dim=2).reshape(len(xb_t), -1)
                frz_feats.append(vec_b.cpu().numpy().astype(np.float64))
        frz_mat = np.concatenate(frz_feats, axis=0)
        frz_mu0, frz_w_eff, frz_b_eff = cls.fit_collapsed_svd_logistic_readout(
            frz_mat, y_prot, svd_components=svd_components, inverse_reg_c=inverse_reg_c, w_eff_transposed=True
        )

        lora_npz_payload: dict[str, np.ndarray] = {
            "n_seeds": np.array(int(lora_seed_count), dtype=np.int64),
            "n0_cent": np.array(float(n0_cent), dtype=np.float32),
            "prior_bal": np.array(float(prior_bal), dtype=np.float32),
        }
        lora_readouts: list[dict[str, np.ndarray]] = []

        for s_idx in range(int(lora_seed_count)):
            torch.manual_seed(int(random_seed) + s_idx)
            rng_local = np.random.default_rng(int(random_seed) + s_idx)
            adapter_mod = codebrain_backbone.build_trainable_lora_tail_adapter(
                lora_rank=lora_rank, lora_alpha=float(lora_rank), execution_device=dev
            )
            optimizer = torch.optim.AdamW(adapter_mod.parameters(), lr=float(learning_rate), weight_decay=1e-2)
            adapter_mod.train()
            for _ in range(max(1, int(lora_epochs))):
                perm = rng_local.permutation(len(x_200_all))
                for b_start in range(0, len(x_200_all), int(batch_size)):
                    b_idx = perm[b_start : b_start + int(batch_size)]
                    xb_t = torch.as_tensor(x_200_all[b_idx], dtype=torch.float32, device=dev)
                    yb_t = torch.as_tensor(y_prot[b_idx], dtype=torch.long, device=dev)
                    optimizer.zero_grad()
                    _, feat_1016 = codebrain_backbone.forward_with_trainable_lora(xb_t, adapter_mod)
                    ro_mod = adapter_mod["readout"]
                    logits_b = ro_mod["fc"](ro_mod["drop"](feat_1016))  # type: ignore[index]
                    loss_b = F.cross_entropy(logits_b, yb_t)
                    loss_b.backward()
                    optimizer.step()

            adapter_mod.eval()
            lora_feats: list[np.ndarray] = []
            with torch.no_grad():
                for b_start in range(0, len(x_200_all), int(batch_size)):
                    xb_t = torch.as_tensor(
                        x_200_all[b_start : b_start + int(batch_size)], dtype=torch.float32, device=dev
                    )
                    tok_s, _ = codebrain_backbone.forward_with_trainable_lora(xb_t, adapter_mod)
                    tfs = tok_s.float()
                    vs = torch.cat(
                        [tfs[:, :, 1:, :].mean(dim=2), torch.sqrt((tfs ** 2).mean(dim=2) + 1e-6)], dim=-1
                    ).reshape(len(xb_t), -1)
                    lora_feats.append(vs.cpu().numpy().astype(np.float64))
            lora_mat = np.concatenate(lora_feats, axis=0)
            mu_s, w_eff_s, b_eff_s = cls.fit_collapsed_svd_logistic_readout(
                lora_mat, y_prot, svd_components=svd_components, inverse_reg_c=inverse_reg_c, w_eff_transposed=True
            )
            lora_npz_payload.update(
                CodebrainFoundationBackbone.export_lora_adapter_arrays(adapter_mod, s_idx, mu_s, w_eff_s, b_eff_s)
            )
            lora_readouts.append({"mu0": mu_s, "W_eff": w_eff_s, "b_eff": b_eff_s})

        # Fold trained LoRA adapters into merged tails on backbone via temp round-trip dict
        tmp_lora_path = Path("/tmp") / f"_proteus_cd_lora_{random_seed}_{id(codebrain_backbone)}.npz"
        try:
            np.savez_compressed(tmp_lora_path, **lora_npz_payload)
            codebrain_backbone.attach_merged_lora_tails_from_npz(tmp_lora_path, dev)
        finally:
            if tmp_lora_path.exists():
                tmp_lora_path.unlink()

        return cls(
            codebrain_backbone=codebrain_backbone,
            frozen_mu0=frz_mu0,
            frozen_w_eff=frz_w_eff,
            frozen_b_eff=frz_b_eff,
            pool_mode=pool_mode,
            n0_cent=n0_cent,
            prior_bal=prior_bal,
            lora_readouts=lora_readouts,
            lora_n0_cent=n0_cent,
            lora_prior_bal=prior_bal,
            execution_device=dev,
            exported_lora_npz_arrays=lora_npz_payload,
        )

    def save_organ_artifacts(self, target_directory: Path) -> list[Path]:
        resolved_dir = ParameterTypeValidator.require_path(target_directory, "target_directory")
        resolved_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []

        cb_payload: dict[str, np.ndarray] = {
            "cfg_json": np.array(self._codebrain_backbone.architecture_config_json),
            "head__mu0": self._frozen_mu0.astype(np.float32),
            "head__W_eff": self._frozen_w_eff.astype(np.float16),
            "head__b_eff": self._frozen_b_eff.astype(np.float32),
            "head__pool_mode": np.array(self._pool_mode),
            "head__n0_cent": np.array(float(self._n0_cent), dtype=np.float32),
            "head__prior_bal": np.array(float(self._prior_bal), dtype=np.float32),
        }
        for key_str, param_t in self._codebrain_backbone.state_dict().items():
            if key_str.startswith(("merged_lora_tails.", "trainable_lora_tails.")):
                continue
            if key_str.endswith("kernel_norm_initialized"):
                cb_payload[f"enc__{key_str}"] = np.array(1.0 if bool(param_t.item()) else 0.0, dtype=np.float32)
            else:
                cb_payload[f"enc__{key_str}"] = param_t.detach().cpu().to(torch.float16).numpy()
        for blk_i, blk_mod in enumerate(self._codebrain_backbone.residual_layer["residual_blocks"]):  # type: ignore[arg-type]
            if "_r32_sglora" in blk_mod.__dict__:
                cb_payload[f"sglora__{blk_i}"] = blk_mod.__dict__["_r32_sglora"].detach().cpu().to(torch.float16).numpy()

        cb_path = resolved_dir / "codebrain_41.npz"
        np.savez_compressed(cb_path, **cb_payload)
        saved_paths.append(cb_path)

        if self._exported_lora_npz_arrays is not None:
            lora_path = resolved_dir / "s2c_codebrain_lora8_2655.npz"
            np.savez_compressed(lora_path, **self._exported_lora_npz_arrays)
            saved_paths.append(lora_path)

        return saved_paths
