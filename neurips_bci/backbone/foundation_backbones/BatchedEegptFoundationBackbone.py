"""Pure-PyTorch 8-layer 512-dim 8-head batched EEGPT backbone and Rank-64 SVD delta codec."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from neurips_bci.backbone.foundation_backbones.FoundationBackbone import FoundationBackbone
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class BatchedEegptFoundationBackbone(FoundationBackbone):
    """Stacked grouped-BMM execution of 4-token (`lat52/clean4/rms4c`) and mean-pool (`rms56`) EEGPT seeds."""

    def __init__(
        self,
        seed_state_bundles: Sequence[dict[str, object]],
        count_52hz_models: int,
        execution_device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(backbone_architecture="batched_eegpt_8L_512d_8h", embedding_dimension=512)
        validated_n52 = ParameterTypeValidator.require_integer(count_52hz_models, "count_52hz_models", 0)
        total_models = len(seed_state_bundles)
        if validated_n52 > total_models:
            raise ValueError(f"count_52hz_models ({validated_n52}) exceeds total seeds ({total_models})")

        self._count_52hz = validated_n52
        self._count_56hz = total_models - validated_n52
        self._total_models = total_models

        four_tok_flags = [bool(bundle_item["is_four_token"]) for bundle_item in seed_state_bundles]
        four_tok_count = sum(1 for flag_item in four_tok_flags if flag_item)
        if not (all(four_tok_flags[:four_tok_count]) and not any(four_tok_flags[four_tok_count:])):
            raise ValueError("All four-token EEGPT seed bundles must precede mean-pool RMS56 seed bundles")
        self._four_token_count = four_tok_count

        resolved_device = torch.device(execution_device)
        weight_dtype = torch.float16 if resolved_device.type == "cuda" else torch.float32
        self._depth = 8

        if total_models > 0:
            with torch.no_grad():
                state_maps: list[dict[str, torch.Tensor]] = [
                    bundle_item["state_tensors"]  # type: ignore[misc]
                    for bundle_item in seed_state_bundles
                ]
                self.register_buffer(
                    "in_scales",
                    torch.stack([sd_map["in_scale"].float() for sd_map in state_maps]).view(total_models, 1, 1),
                )
                self.register_buffer(
                    "pe_w",
                    torch.cat(
                        [sd_map["eegpt.target_encoder.patch_embed.proj.weight"] for sd_map in state_maps], dim=0
                    ).to(dtype=weight_dtype),
                )
                self.register_buffer(
                    "pe_b",
                    torch.cat(
                        [sd_map["eegpt.target_encoder.patch_embed.proj.bias"] for sd_map in state_maps], dim=0
                    ).to(dtype=weight_dtype),
                )
                chan_embeds = [
                    F.embedding(
                        sd_map["eegpt.chans_id"].to(dtype=torch.long),
                        sd_map["eegpt.target_encoder.chan_embed.weight"],
                    )
                    for sd_map in state_maps
                ]
                self.register_buffer("chan_add", torch.stack(chan_embeds, dim=0).to(dtype=weight_dtype))
                self.register_buffer(
                    "summary_tok",
                    torch.stack([sd_map["eegpt.target_encoder.summary_token"] for sd_map in state_maps], dim=0).to(
                        dtype=weight_dtype
                    ),
                )

                for block_idx in range(self._depth):
                    blk_pfx = f"eegpt.target_encoder.blocks.{block_idx}"
                    self.register_buffer(
                        f"n1_w_{block_idx}",
                        torch.stack([sd_map[f"{blk_pfx}.norm1.weight"] for sd_map in state_maps])
                        .unsqueeze(1)
                        .to(dtype=weight_dtype),
                    )
                    self.register_buffer(
                        f"n1_b_{block_idx}",
                        torch.stack([sd_map[f"{blk_pfx}.norm1.bias"] for sd_map in state_maps])
                        .unsqueeze(1)
                        .to(dtype=weight_dtype),
                    )
                    self.register_buffer(
                        f"qkv_wT_{block_idx}",
                        torch.stack(
                            [sd_map[f"{blk_pfx}.attn.qkv.weight"].t().contiguous() for sd_map in state_maps]
                        ).to(dtype=weight_dtype),
                    )
                    self.register_buffer(
                        f"qkv_b_{block_idx}",
                        torch.stack([sd_map[f"{blk_pfx}.attn.qkv.bias"] for sd_map in state_maps])
                        .unsqueeze(1)
                        .to(dtype=weight_dtype),
                    )
                    self.register_buffer(
                        f"proj_wT_{block_idx}",
                        torch.stack(
                            [sd_map[f"{blk_pfx}.attn.proj.weight"].t().contiguous() for sd_map in state_maps]
                        ).to(dtype=weight_dtype),
                    )
                    self.register_buffer(
                        f"proj_b_{block_idx}",
                        torch.stack([sd_map[f"{blk_pfx}.attn.proj.bias"] for sd_map in state_maps])
                        .unsqueeze(1)
                        .to(dtype=weight_dtype),
                    )
                    self.register_buffer(
                        f"n2_w_{block_idx}",
                        torch.stack([sd_map[f"{blk_pfx}.norm2.weight"] for sd_map in state_maps])
                        .unsqueeze(1)
                        .to(dtype=weight_dtype),
                    )
                    self.register_buffer(
                        f"n2_b_{block_idx}",
                        torch.stack([sd_map[f"{blk_pfx}.norm2.bias"] for sd_map in state_maps])
                        .unsqueeze(1)
                        .to(dtype=weight_dtype),
                    )
                    self.register_buffer(
                        f"fc1_wT_{block_idx}",
                        torch.stack(
                            [sd_map[f"{blk_pfx}.mlp.fc1.weight"].t().contiguous() for sd_map in state_maps]
                        ).to(dtype=weight_dtype),
                    )
                    self.register_buffer(
                        f"fc1_b_{block_idx}",
                        torch.stack([sd_map[f"{blk_pfx}.mlp.fc1.bias"] for sd_map in state_maps])
                        .unsqueeze(1)
                        .to(dtype=weight_dtype),
                    )
                    self.register_buffer(
                        f"fc2_wT_{block_idx}",
                        torch.stack(
                            [sd_map[f"{blk_pfx}.mlp.fc2.weight"].t().contiguous() for sd_map in state_maps]
                        ).to(dtype=weight_dtype),
                    )
                    self.register_buffer(
                        f"fc2_b_{block_idx}",
                        torch.stack([sd_map[f"{blk_pfx}.mlp.fc2.bias"] for sd_map in state_maps])
                        .unsqueeze(1)
                        .to(dtype=weight_dtype),
                    )

                self.register_buffer(
                    "te_nw",
                    torch.stack([sd_map["eegpt.target_encoder.norm.weight"] for sd_map in state_maps])
                    .unsqueeze(1)
                    .to(dtype=weight_dtype),
                )
                self.register_buffer(
                    "te_nb",
                    torch.stack([sd_map["eegpt.target_encoder.norm.bias"] for sd_map in state_maps])
                    .unsqueeze(1)
                    .to(dtype=weight_dtype),
                )
                self.register_buffer(
                    "hd_nw",
                    torch.stack([sd_map["norm.weight"] for sd_map in state_maps]).to(dtype=weight_dtype),
                )
                self.register_buffer(
                    "hd_nb",
                    torch.stack([sd_map["norm.bias"] for sd_map in state_maps]).to(dtype=weight_dtype),
                )

                if four_tok_count > 0:
                    tok4_maps = state_maps[:four_tok_count]
                    self.register_buffer(
                        "tok_nw",
                        torch.stack([sd_map["tok_norm.weight"] for sd_map in tok4_maps])
                        .unsqueeze(1)
                        .to(dtype=weight_dtype),
                    )
                    self.register_buffer(
                        "tok_nb",
                        torch.stack([sd_map["tok_norm.bias"] for sd_map in tok4_maps])
                        .unsqueeze(1)
                        .to(dtype=weight_dtype),
                    )
                    self.register_buffer(
                        "tok_pwT",
                        torch.stack([sd_map["tok_proj.weight"].t().contiguous() for sd_map in tok4_maps]).to(
                            dtype=weight_dtype
                        ),
                    )
                    self.register_buffer(
                        "tok_pb",
                        torch.stack([sd_map["tok_proj.bias"] for sd_map in tok4_maps])
                        .unsqueeze(1)
                        .to(dtype=weight_dtype),
                    )
                    self.register_buffer(
                        "dw_w",
                        torch.cat([sd_map["temp_dw.weight"] for sd_map in tok4_maps], dim=0).to(dtype=weight_dtype),
                    )
                    self.register_buffer(
                        "dw_b",
                        torch.cat([sd_map["temp_dw.bias"] for sd_map in tok4_maps], dim=0).to(dtype=weight_dtype),
                    )
                    self.register_buffer(
                        "h4_nw",
                        torch.stack([sd_map["head_4tok_norm.weight"] for sd_map in tok4_maps]).to(dtype=weight_dtype),
                    )
                    self.register_buffer(
                        "h4_nb",
                        torch.stack([sd_map["head_4tok_norm.bias"] for sd_map in tok4_maps]).to(dtype=weight_dtype),
                    )
                    self.register_buffer(
                        "alpha_4tok",
                        torch.stack([sd_map["alpha_4tok"] for sd_map in tok4_maps])
                        .view(four_tok_count, 1)
                        .to(dtype=weight_dtype),
                    )

        self._single_cuda_graph: object = None
        self._dual_cuda_graph: object = None
        self.to(resolved_device).eval()

    @property
    def total_models(self) -> int:
        return self._total_models

    @property
    def count_52hz(self) -> int:
        return self._count_52hz

    @property
    def count_56hz(self) -> int:
        return self._count_56hz

    @property
    def four_token_count(self) -> int:
        return self._four_token_count

    @staticmethod
    def fft_resample_common_average(
        window_tensor: torch.Tensor,
        target_samples: int = 1024,
        lowpass_hz: float = 52.0,
        mirror_pad: int = 30,
    ) -> torch.Tensor:
        """Exact DC + CAR + reflect-padded FFT resampler matching `_slot6_fft_resample_car`."""
        centered = window_tensor - window_tensor.mean(dim=-1, keepdim=True)
        car_referenced = centered - centered.mean(dim=1, keepdim=True)
        in_samples = car_referenced.shape[-1]
        padded_len = in_samples + 2 * mirror_pad
        extended_len = (padded_len * target_samples) // in_samples
        trim_offset = (mirror_pad * target_samples) // in_samples
        freq_spectrum = torch.fft.rfft(F.pad(car_referenced.float(), (mirror_pad, mirror_pad), mode="reflect"), dim=-1)
        keep_bins = min(padded_len // 2 + 1, int(lowpass_hz * padded_len / 120.0) + 1)
        filtered_spectrum = torch.zeros(
            *freq_spectrum.shape[:-1],
            extended_len // 2 + 1,
            dtype=freq_spectrum.dtype,
            device=freq_spectrum.device,
        )
        filtered_spectrum[..., :keep_bins] = freq_spectrum[..., :keep_bins]
        resampled = torch.fft.irfft(filtered_spectrum, n=extended_len, dim=-1) * (extended_len / padded_len)
        return resampled[..., trim_offset : trim_offset + target_samples]

    @staticmethod
    def reconstruct_seed_bundle_from_npz(
        seed_npz_path: str | Path,
        base_fp16_tensors: dict[str, np.ndarray],
        stream_tag: str,
    ) -> tuple[dict[str, object], np.ndarray, tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]]:
        """Reconstruct one fine-tuned EEGPT seed state bundle, train mean, and replay tuple from `base_fp16` + delta `.npz`."""
        resolved_npz = ParameterTypeValidator.require_path(seed_npz_path, "seed_npz_path", must_exist=True)
        with np.load(resolved_npz, allow_pickle=False) as archive:
            manifest = json.loads(str(archive["meta__manifest"]))
            arch_label = str(manifest.get("arch", ""))
            lowpass_hz = float(manifest.get("lowpass_hz", 52.0 if stream_tag == "lat52" else 56.0))
            in_scale_init = float(manifest.get("in_scale_init", 0.015 if stream_tag == "lat52" else 0.018))
            is_four_token = (
                (stream_tag == "lat52")
                or ("4Tok" in arch_label)
                or any("tok_proj" in head_key for head_key in manifest["head"])
            )

            reconstructed_eegpt: dict[str, torch.Tensor] = {}
            for svd_key in manifest["svd"]:
                base_tensor = torch.from_numpy(base_fp16_tensors[svd_key].astype(np.float32))
                factor_a = torch.from_numpy(archive[f"svdA__{svd_key}"].astype(np.float32))
                factor_b = torch.from_numpy(archive[f"svdB__{svd_key}"].astype(np.float32))
                reconstructed_eegpt[svd_key] = base_tensor + (factor_a @ factor_b)
            for delta_key in manifest["delta"]:
                base_tensor = torch.from_numpy(base_fp16_tensors[delta_key].astype(np.float32))
                reconstructed_eegpt[delta_key] = base_tensor + torch.from_numpy(
                    archive[f"dlt__{delta_key}"].astype(np.float32)
                )
            for full_key in manifest["full"]:
                reconstructed_eegpt[full_key] = torch.from_numpy(archive[f"full__{full_key}"].astype(np.float32))
            if "chans_id" in reconstructed_eegpt:
                reconstructed_eegpt["chans_id"] = torch.round(reconstructed_eegpt["chans_id"]).to(torch.long)

            state_tensors: dict[str, torch.Tensor] = {
                f"eegpt.{mod_key}": mod_tensor for mod_key, mod_tensor in reconstructed_eegpt.items()
            }
            for head_key in manifest["head"]:
                state_tensors[head_key] = torch.from_numpy(archive[f"head__{head_key}"].astype(np.float32))

            train_mean = archive["meta__mu_train"].astype(np.float64)
            replay_features = archive["meta__F_rep"].astype(np.float64)
            replay_labels = archive["meta__y_rep"].astype(np.int64)
            head_weights = archive["meta__W_head"].astype(np.float64)
            head_bias = archive["meta__b_head"].astype(np.float64)

        seed_bundle: dict[str, object] = {
            "is_four_token": is_four_token,
            "lowpass_hz": lowpass_hz,
            "in_scale_init": in_scale_init,
            "state_tensors": state_tensors,
        }
        return seed_bundle, train_mean, (replay_features, replay_labels, (head_weights, head_bias))

    @staticmethod
    def build_trainable_seed_module(
        base_fp16_tensors: dict[str, np.ndarray],
        is_four_token: bool = True,
        lowpass_hz: float = 52.0,
        in_scale_init: float = 0.015,
        class_count: int = 3,
        execution_device: str | torch.device = "cpu",
    ) -> nn.ModuleDict:
        """Build a standalone trainable single-seed EEGPT `ModuleDict` initialized from `base_eegpt_fp16`."""
        blocks_list = nn.ModuleList()
        for _ in range(8):
            blocks_list.append(
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(512, eps=1e-6),
                        "attn": nn.ModuleDict({"qkv": nn.Linear(512, 1536, bias=True), "proj": nn.Linear(512, 512)}),
                        "norm2": nn.LayerNorm(512, eps=1e-6),
                        "mlp": nn.ModuleDict({"fc1": nn.Linear(512, 2048), "fc2": nn.Linear(2048, 512)}),
                    }
                )
            )
        target_enc = nn.ModuleDict(
            {
                "patch_embed": nn.ModuleDict({"proj": nn.Conv2d(1, 512, kernel_size=(1, 64), stride=(1, 64))}),
                "chan_embed": nn.Embedding(62, 512),
                "blocks": blocks_list,
                "norm": nn.LayerNorm(512, eps=1e-6),
            }
        )
        target_enc.register_parameter("summary_token", nn.Parameter(torch.zeros(1, 4, 512)))

        eegpt_mod = nn.ModuleDict({"target_encoder": target_enc})
        eegpt_mod.register_buffer("chans_id", torch.zeros(1, 41, dtype=torch.long))

        eegpt_sd: dict[str, torch.Tensor] = {}
        for key_str, arr_val in base_fp16_tensors.items():
            if key_str == "chans_id":
                eegpt_sd[key_str] = torch.round(torch.from_numpy(arr_val.astype(np.float32))).to(torch.long)
            else:
                eegpt_sd[key_str] = torch.from_numpy(arr_val.astype(np.float32))
        eegpt_mod.load_state_dict(eegpt_sd, strict=True)

        seed_mod = nn.ModuleDict(
            {
                "eegpt": eegpt_mod,
                "norm": nn.LayerNorm(512),
                "drop": nn.Dropout(0.30),
                "fc": nn.Linear(512, class_count),
                "aux_shin18": nn.Linear(512, 2),
                "aux_zyma19": nn.Linear(512, 2),
            }
        )
        seed_mod.register_parameter("in_scale", nn.Parameter(torch.tensor(float(in_scale_init), dtype=torch.float32)))
        seed_mod.register_buffer("lowpass_hz_buf", torch.tensor(float(lowpass_hz), dtype=torch.float32))
        seed_mod.register_buffer("is_four_tok_buf", torch.tensor(1 if is_four_token else 0, dtype=torch.long))

        if is_four_token:
            seed_mod["tok_norm"] = nn.LayerNorm(512)
            seed_mod["tok_proj"] = nn.Linear(512, 64)
            seed_mod["temp_dw"] = nn.Conv1d(256, 256, kernel_size=5, padding=2, groups=256)
            seed_mod["head_4tok_norm"] = nn.LayerNorm(512)
            seed_mod.register_parameter("alpha_4tok", nn.Parameter(torch.tensor(0.25, dtype=torch.float32)))

        return seed_mod.to(torch.device(execution_device))

    @classmethod
    def extract_trainable_seed_features(
        cls,
        seed_module: nn.ModuleDict,
        window_batch: torch.Tensor,
        channel_subset: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Forward pass through a trainable single-seed EEGPT `ModuleDict`."""
        lp_hz = float(seed_module.lowpass_hz_buf.item())
        resampled = cls.fft_resample_common_average(window_batch, target_samples=1024, lowpass_hz=lp_hz)
        scaled = resampled * seed_module.in_scale
        eegpt_mod: nn.ModuleDict = seed_module["eegpt"]  # type: ignore[assignment]
        target_enc: nn.ModuleDict = eegpt_mod["target_encoder"]  # type: ignore[assignment]
        chan_ids = eegpt_mod.chans_id if channel_subset is None else eegpt_mod.chans_id[:, list(channel_subset)]

        patched = target_enc["patch_embed"]["proj"](scaled.unsqueeze(1)).permute(0, 3, 2, 1).contiguous()  # type: ignore[index]
        batch_sz, patch_cnt, chan_cnt, emb_dim = patched.shape
        chan_vec = target_enc["chan_embed"](chan_ids.to(device=patched.device, dtype=torch.long)).unsqueeze(0)
        tokens = (patched + chan_vec).reshape(batch_sz * patch_cnt, chan_cnt, emb_dim)
        summary_expanded = target_enc.summary_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([tokens, summary_expanded], dim=1)

        for blk_mod in target_enc["blocks"]:  # type: ignore[attr-defined]
            normed_1 = blk_mod["norm1"](tokens)
            seq_n = normed_1.shape[1]
            qkv_proj = (
                blk_mod["attn"]["qkv"](normed_1)
                .reshape(batch_sz * patch_cnt, seq_n, 3, 8, 64)
                .permute(2, 0, 3, 1, 4)
            )
            attn_out = F.scaled_dot_product_attention(qkv_proj[0], qkv_proj[1], qkv_proj[2], dropout_p=0.0, is_causal=False)
            tokens = tokens + blk_mod["attn"]["proj"](attn_out.transpose(1, 2).reshape(batch_sz * patch_cnt, seq_n, emb_dim))
            tokens = tokens + blk_mod["mlp"]["fc2"](F.gelu(blk_mod["mlp"]["fc1"](blk_mod["norm2"](tokens))))

        summary_out = target_enc["norm"](tokens[:, -4:, :]).reshape(batch_sz, patch_cnt, 4, emb_dim)
        base_vec = seed_module["norm"](summary_out.mean(dim=(1, 2)))
        if int(seed_module.is_four_tok_buf.item()) == 0:
            return base_vec

        tok_hidden = F.gelu(seed_module["tok_proj"](seed_module["tok_norm"](summary_out)))
        tok_flat = tok_hidden.reshape(batch_sz, 16, 256).transpose(1, 2)
        tok_dyn = tok_flat + F.gelu(seed_module["temp_dw"](tok_flat))
        early_half = tok_dyn[:, :, :8].mean(dim=-1)
        late_half = tok_dyn[:, :, 8:].mean(dim=-1)
        spatiotemporal_vec = seed_module["head_4tok_norm"](torch.cat([early_half, late_half], dim=-1))
        return base_vec + seed_module.alpha_4tok * spatiotemporal_vec

    @staticmethod
    def compress_trained_seed_to_npz_arrays(
        seed_module: nn.ModuleDict,
        base_fp16_tensors: dict[str, np.ndarray],
        train_mean: np.ndarray,
        replay_tuple: tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]],
        svd_rank: int | None = 64,
    ) -> dict[str, np.ndarray]:
        """Encode a trained single-seed EEGPT `ModuleDict` into pure-NumPy Rank-`svd_rank` delta arrays."""
        encoded_arrays: dict[str, np.ndarray] = {}
        eegpt_state = seed_module["eegpt"].state_dict()
        svd_keys: list[str] = []
        delta_keys: list[str] = []
        full_keys: list[str] = []

        for key_str, param_tensor in eegpt_state.items():
            if key_str.startswith("final_layer."):
                continue
            fp32_tensor = param_tensor.detach().cpu().to(torch.float32)
            if key_str not in base_fp16_tensors:
                encoded_arrays[f"full__{key_str}"] = (
                    fp32_tensor.numpy() if fp32_tensor.ndim < 2 else fp32_tensor.to(torch.float16).numpy()
                )
                full_keys.append(key_str)
                continue
            base_fp32 = torch.from_numpy(base_fp16_tensors[key_str].astype(np.float32))
            delta_tensor = fp32_tensor - base_fp32
            if (
                svd_rank is not None
                and delta_tensor.ndim == 2
                and min(delta_tensor.shape) > 2 * svd_rank
                and delta_tensor.numel() >= 32768
            ):
                u_mat, s_vals, vh_mat = torch.linalg.svd(delta_tensor, full_matrices=False)
                eff_rank = min(int(svd_rank), len(s_vals))
                sqrt_s = torch.sqrt(s_vals[:eff_rank])
                encoded_arrays[f"svdA__{key_str}"] = (u_mat[:, :eff_rank] * sqrt_s[None, :]).to(torch.float16).numpy()
                encoded_arrays[f"svdB__{key_str}"] = (sqrt_s[:, None] * vh_mat[:eff_rank, :]).to(torch.float16).numpy()
                svd_keys.append(key_str)
            else:
                encoded_arrays[f"dlt__{key_str}"] = (
                    delta_tensor.to(torch.float16).numpy() if delta_tensor.ndim >= 2 else delta_tensor.numpy()
                )
                delta_keys.append(key_str)

        head_keys: list[str] = []
        skip_prefixes = ("eegpt.", "aux_", "lowpass_hz_buf", "is_four_tok_buf")
        for key_str, param_tensor in seed_module.state_dict().items():
            if key_str.startswith(skip_prefixes):
                continue
            encoded_arrays[f"head__{key_str}"] = param_tensor.detach().cpu().to(torch.float32).numpy()
            head_keys.append(key_str)

        rep_features, rep_labels, (head_w, head_b) = replay_tuple
        is_4tok = bool(int(seed_module.is_four_tok_buf.item()) == 1)
        lp_hz = float(seed_module.lowpass_hz_buf.item())
        encoded_arrays["meta__mu_train"] = np.asarray(train_mean, dtype=np.float32)
        encoded_arrays["meta__F_rep"] = np.asarray(rep_features, dtype=np.float16)
        encoded_arrays["meta__y_rep"] = np.asarray(rep_labels, dtype=np.int64)
        encoded_arrays["meta__W_head"] = np.asarray(head_w, dtype=np.float32)
        encoded_arrays["meta__b_head"] = np.asarray(head_b, dtype=np.float32)
        encoded_arrays["meta__manifest"] = np.array(
            json.dumps(
                {
                    "arch": "EEGPT4Tok" if is_4tok else "EEGPTRMS56",
                    "lowpass_hz": lp_hz,
                    "svd": svd_keys,
                    "delta": delta_keys,
                    "full": full_keys,
                    "head": head_keys,
                }
            )
        )
        return encoded_arrays

    def _forward_eager(self, x256_52: torch.Tensor, x256_56: torch.Tensor) -> torch.Tensor:
        if self._total_models == 0:
            return torch.zeros((0, 512), dtype=torch.float32, device=x256_52.device)
        weight_dt = self.pe_w.dtype
        input_parts: list[torch.Tensor] = []
        if self._count_52hz > 0:
            input_parts.append(x256_52.expand(self._count_52hz, -1, -1))
        if self._count_56hz > 0:
            input_parts.append(x256_56.expand(self._count_56hz, -1, -1))
        stacked_input = (torch.cat(input_parts, dim=0) * self.in_scales).to(dtype=weight_dt).unsqueeze(0)
        tokens = F.conv2d(stacked_input, self.pe_w, self.pe_b, stride=(1, 64), groups=self._total_models)
        tokens = tokens.view(self._total_models, 512, 41, 16).permute(0, 3, 2, 1).contiguous()
        tokens = tokens + self.chan_add
        summary_expanded = self.summary_tok.expand(self._total_models, 16, -1, -1)
        tokens = torch.cat([tokens, summary_expanded], dim=2).reshape(self._total_models, 720, 512)

        for block_idx in range(self._depth):
            normed_1 = F.layer_norm(tokens, (512,), eps=1e-6) * getattr(self, f"n1_w_{block_idx}") + getattr(
                self, f"n1_b_{block_idx}"
            )
            qkv = torch.baddbmm(getattr(self, f"qkv_b_{block_idx}"), normed_1, getattr(self, f"qkv_wT_{block_idx}"))
            qkv = (
                qkv.view(self._total_models, 16, 45, 3, 8, 64)
                .permute(3, 0, 1, 4, 2, 5)
                .reshape(3, self._total_models * 16, 8, 45, 64)
            )
            attn_out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], dropout_p=0.0, is_causal=False)
            attn_out = attn_out.view(self._total_models, 16, 8, 45, 64).transpose(2, 3).reshape(self._total_models, 720, 512)
            tokens = tokens + torch.baddbmm(
                getattr(self, f"proj_b_{block_idx}"), attn_out, getattr(self, f"proj_wT_{block_idx}")
            )

            normed_2 = F.layer_norm(tokens, (512,), eps=1e-6) * getattr(self, f"n2_w_{block_idx}") + getattr(
                self, f"n2_b_{block_idx}"
            )
            mlp_hidden = F.gelu(
                torch.baddbmm(getattr(self, f"fc1_b_{block_idx}"), normed_2, getattr(self, f"fc1_wT_{block_idx}"))
            )
            tokens = tokens + torch.baddbmm(
                getattr(self, f"fc2_b_{block_idx}"), mlp_hidden, getattr(self, f"fc2_wT_{block_idx}")
            )

        summary_slice = tokens.view(self._total_models, 16, 45, 512)[:, :, -4:, :].reshape(self._total_models, 64, 512)
        summary_normed = (F.layer_norm(summary_slice, (512,), eps=1e-6) * self.te_nw + self.te_nb).view(
            self._total_models, 16, 4, 512
        )
        mean_pooled = summary_normed.mean(dim=(1, 2))
        base_vec = F.layer_norm(mean_pooled, (512,), eps=1e-5) * self.hd_nw + self.hd_nb

        if self._four_token_count == 0:
            return base_vec.float()
        z_four = summary_normed[: self._four_token_count].reshape(self._four_token_count, 64, 512)
        z_tok_norm = F.layer_norm(z_four, (512,), eps=1e-5) * self.tok_nw + self.tok_nb
        z_tok_proj = F.gelu(torch.baddbmm(self.tok_pb, z_tok_norm, self.tok_pwT))
        z_flat = z_tok_proj.view(self._four_token_count, 16, 256).transpose(1, 2).contiguous()
        z_dw = F.conv1d(
            z_flat.view(1, self._four_token_count * 256, 16),
            self.dw_w,
            self.dw_b,
            padding=2,
            groups=self._four_token_count * 256,
        ).view(self._four_token_count, 256, 16)
        z_dyn = z_flat + F.gelu(z_dw)
        early_late = torch.cat([z_dyn[:, :, :8].mean(dim=-1), z_dyn[:, :, 8:].mean(dim=-1)], dim=-1)
        st_vec = F.layer_norm(early_late, (512,), eps=1e-5) * self.h4_nw + self.h4_nb
        four_tok_features = base_vec[: self._four_token_count] + self.alpha_4tok * st_vec
        if self._four_token_count == self._total_models:
            return four_tok_features.float()
        return torch.cat([four_tok_features, base_vec[self._four_token_count :]], dim=0).float()

    def _forward_dual_eager(self, x52_pair: torch.Tensor, x56_pair: torch.Tensor) -> torch.Tensor:
        if self._total_models == 0:
            return torch.zeros((0, 2, 512), dtype=torch.float32, device=x52_pair.device)
        weight_dt = self.pe_w.dtype
        input_parts: list[torch.Tensor] = []
        if self._count_52hz > 0:
            input_parts.append(x52_pair.unsqueeze(1).expand(2, self._count_52hz, -1, -1))
        if self._count_56hz > 0:
            input_parts.append(x56_pair.unsqueeze(1).expand(2, self._count_56hz, -1, -1))
        stacked_pair = (torch.cat(input_parts, dim=1) * self.in_scales.unsqueeze(0)).to(dtype=weight_dt)
        tokens = F.conv2d(stacked_pair, self.pe_w, self.pe_b, stride=(1, 64), groups=self._total_models)
        tokens = tokens.view(2, self._total_models, 512, 41, 16).permute(1, 0, 4, 3, 2).contiguous()
        tokens = tokens + self.chan_add.unsqueeze(1)
        summary_expanded = self.summary_tok.unsqueeze(1).expand(self._total_models, 2, 16, -1, -1)
        tokens = torch.cat([tokens, summary_expanded], dim=3).reshape(self._total_models, 1440, 512)

        for block_idx in range(self._depth):
            normed_1 = F.layer_norm(tokens, (512,), eps=1e-6) * getattr(self, f"n1_w_{block_idx}") + getattr(
                self, f"n1_b_{block_idx}"
            )
            qkv = torch.baddbmm(getattr(self, f"qkv_b_{block_idx}"), normed_1, getattr(self, f"qkv_wT_{block_idx}"))
            qkv = (
                qkv.view(self._total_models, 32, 45, 3, 8, 64)
                .permute(3, 0, 1, 4, 2, 5)
                .reshape(3, self._total_models * 32, 8, 45, 64)
            )
            attn_out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], dropout_p=0.0, is_causal=False)
            attn_out = (
                attn_out.view(self._total_models, 32, 8, 45, 64).transpose(2, 3).reshape(self._total_models, 1440, 512)
            )
            tokens = tokens + torch.baddbmm(
                getattr(self, f"proj_b_{block_idx}"), attn_out, getattr(self, f"proj_wT_{block_idx}")
            )

            normed_2 = F.layer_norm(tokens, (512,), eps=1e-6) * getattr(self, f"n2_w_{block_idx}") + getattr(
                self, f"n2_b_{block_idx}"
            )
            mlp_hidden = F.gelu(
                torch.baddbmm(getattr(self, f"fc1_b_{block_idx}"), normed_2, getattr(self, f"fc1_wT_{block_idx}"))
            )
            tokens = tokens + torch.baddbmm(
                getattr(self, f"fc2_b_{block_idx}"), mlp_hidden, getattr(self, f"fc2_wT_{block_idx}")
            )

        summary_slice = tokens.view(self._total_models, 2, 16, 45, 512)[:, :, :, -4:, :].reshape(
            self._total_models, 128, 512
        )
        summary_normed = (F.layer_norm(summary_slice, (512,), eps=1e-6) * self.te_nw + self.te_nb).view(
            self._total_models, 2, 16, 4, 512
        )
        mean_pooled = summary_normed.mean(dim=(2, 3))
        base_vec = F.layer_norm(mean_pooled, (512,), eps=1e-5) * self.hd_nw.unsqueeze(1) + self.hd_nb.unsqueeze(1)

        if self._four_token_count == 0:
            return base_vec.float()
        z_four = summary_normed[: self._four_token_count].reshape(self._four_token_count, 128, 512)
        z_tok_norm = F.layer_norm(z_four, (512,), eps=1e-5) * self.tok_nw + self.tok_nb
        z_tok_proj = F.gelu(torch.baddbmm(self.tok_pb, z_tok_norm, self.tok_pwT))
        z_b2 = z_tok_proj.view(self._four_token_count, 2, 16, 256).permute(1, 0, 3, 2).contiguous()
        z_dw = F.conv1d(
            z_b2.view(2, self._four_token_count * 256, 16),
            self.dw_w,
            self.dw_b,
            padding=2,
            groups=self._four_token_count * 256,
        ).view(2, self._four_token_count, 256, 16)
        z_dyn = (z_b2 + F.gelu(z_dw)).permute(1, 0, 2, 3)
        early_late = torch.cat([z_dyn[:, :, :, :8].mean(dim=-1), z_dyn[:, :, :, 8:].mean(dim=-1)], dim=-1)
        st_vec = F.layer_norm(early_late, (512,), eps=1e-5) * self.h4_nw.unsqueeze(1) + self.h4_nb.unsqueeze(1)
        four_tok_features = base_vec[: self._four_token_count] + self.alpha_4tok.unsqueeze(1) * st_vec
        if self._four_token_count == self._total_models:
            return four_tok_features.float()
        return torch.cat([four_tok_features, base_vec[self._four_token_count :]], dim=0).float()

    def forward_single_window(self, x256_52: torch.Tensor, x256_56: torch.Tensor) -> torch.Tensor:
        if (
            self._total_models > 0
            and x256_52.is_cuda
            and x256_52.shape == (1, 41, 1024)
            and x256_56.shape == (1, 41, 1024)
            and self._single_cuda_graph is not False
        ):
            if self._single_cuda_graph is None:
                try:
                    cuda_dev = x256_52.device
                    static_52 = torch.zeros((1, 41, 1024), dtype=torch.float32, device=cuda_dev)
                    static_56 = torch.zeros((1, 41, 1024), dtype=torch.float32, device=cuda_dev)
                    side_stream = torch.cuda.Stream(device=cuda_dev)
                    side_stream.wait_stream(torch.cuda.current_stream(cuda_dev))
                    with torch.cuda.stream(side_stream), torch.no_grad():
                        for _ in range(2):
                            _ = self._forward_eager(static_52, static_56)
                    torch.cuda.current_stream(cuda_dev).wait_stream(side_stream)
                    captured_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(captured_graph, stream=side_stream), torch.no_grad():
                        static_out = self._forward_eager(static_52, static_56)
                    self._single_cuda_graph = (captured_graph, static_52, static_56, static_out)
                except Exception:
                    self._single_cuda_graph = False
            if self._single_cuda_graph:
                captured_graph, static_52, static_56, static_out = self._single_cuda_graph  # type: ignore[misc]
                static_52.copy_(x256_52)
                static_56.copy_(x256_56)
                captured_graph.replay()
                return static_out
        return self._forward_eager(x256_52, x256_56)

    def forward_dual_window(self, x52_pair: torch.Tensor, x56_pair: torch.Tensor) -> torch.Tensor:
        if (
            self._total_models > 0
            and x52_pair.is_cuda
            and x52_pair.shape == (2, 41, 1024)
            and x56_pair.shape == (2, 41, 1024)
            and self._dual_cuda_graph is not False
        ):
            if self._dual_cuda_graph is None:
                try:
                    cuda_dev = x52_pair.device
                    static_52 = torch.zeros((2, 41, 1024), dtype=torch.float32, device=cuda_dev)
                    static_56 = torch.zeros((2, 41, 1024), dtype=torch.float32, device=cuda_dev)
                    side_stream = torch.cuda.Stream(device=cuda_dev)
                    side_stream.wait_stream(torch.cuda.current_stream(cuda_dev))
                    with torch.cuda.stream(side_stream), torch.no_grad():
                        for _ in range(2):
                            _ = self._forward_dual_eager(static_52, static_56)
                    torch.cuda.current_stream(cuda_dev).wait_stream(side_stream)
                    captured_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(captured_graph, stream=side_stream), torch.no_grad():
                        static_out = self._forward_dual_eager(static_52, static_56)
                    self._dual_cuda_graph = (captured_graph, static_52, static_56, static_out)
                except Exception:
                    self._dual_cuda_graph = False
            if self._dual_cuda_graph:
                captured_graph, static_52, static_56, static_out = self._dual_cuda_graph  # type: ignore[misc]
                static_52.copy_(x52_pair)
                static_56.copy_(x56_pair)
                captured_graph.replay()
                return static_out
        return self._forward_dual_eager(x52_pair, x56_pair)

    def encode_window_features(self, *tensor_inputs: torch.Tensor) -> torch.Tensor:
        if len(tensor_inputs) != 2:
            raise ValueError("BatchedEegptFoundationBackbone expects (x256_52, x256_56)")
        x52_in, x56_in = tensor_inputs
        if x52_in.shape[0] == 1 and x56_in.shape[0] == 1:
            return self.forward_single_window(x52_in, x56_in)
        if x52_in.shape[0] == 2 and x56_in.shape[0] == 2:
            return self.forward_dual_window(x52_in, x56_in)
        per_window = [
            self._forward_eager(x52_in[win_idx : win_idx + 1], x56_in[win_idx : win_idx + 1])
            for win_idx in range(x52_in.shape[0])
        ]
        return torch.stack(per_window, dim=0)

    def forward(self, x256_52: torch.Tensor, x256_56: torch.Tensor) -> torch.Tensor:
        return self.encode_window_features(x256_52, x256_56)
