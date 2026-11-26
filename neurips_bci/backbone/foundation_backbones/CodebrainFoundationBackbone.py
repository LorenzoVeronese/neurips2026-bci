"""Pure-PyTorch 8-block SGConv + sliding-window-attention CodeBrain backbone with merged Rank-8 LoRA tails."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from neurips_bci.backbone.foundation_backbones.FoundationBackbone import FoundationBackbone
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class CodebrainFoundationBackbone(FoundationBackbone):
    """Pure-PyTorch 8-block S4/SGConv + SWA CodeBrain encoder with shared stem (0..4) and merged Rank-8 LoRA tails."""

    def __init__(
        self,
        patch_size: int = 200,
        res_channels: int = 200,
        skip_channels: int = 200,
        out_channels: int = 200,
        num_res_layers: int = 8,
        s4_lmax: int = 570,
        s4_d_state: int = 64,
        s4_bidirectional: bool = True,
        s4_layernorm: bool = False,
        swa_window_size: int = 1,
        drop_prob: float = 0.1,
        conv_out_chans: int = 25,
        conv_groups: int = 5,
        proj_kernel_size: int = 49,
        proj_padding: int = 24,
        proj_refine_kernel: int = 3,
        pos_kernel: tuple[int, int] = (19, 7),
        spectral_dropout: float = 0.1,
        architecture_config_json: str | None = None,
    ) -> None:
        validated_patch = ParameterTypeValidator.require_integer(patch_size, "patch_size", minimum_bound=8)
        validated_res = ParameterTypeValidator.require_integer(res_channels, "res_channels", minimum_bound=8)
        validated_skip = ParameterTypeValidator.require_integer(skip_channels, "skip_channels", minimum_bound=8)
        validated_out = ParameterTypeValidator.require_integer(out_channels, "out_channels", minimum_bound=8)
        validated_layers = ParameterTypeValidator.require_integer(num_res_layers, "num_res_layers", minimum_bound=1)
        if swa_window_size % 2 != 1:
            raise ValueError(f"swa_window_size must be odd, got {swa_window_size}")

        super().__init__(backbone_architecture="codebrain_sgconv_swa_8L_200d", embedding_dimension=validated_out)
        self._patch_size = validated_patch
        self._res_channels = validated_res
        self._skip_channels = validated_skip
        self._out_channels = validated_out
        self._num_res_layers = validated_layers
        self._s4_lmax = int(s4_lmax)
        self._s4_d_state = int(s4_d_state)
        self._s4_bidirectional = bool(s4_bidirectional)
        self._s4_layernorm = bool(s4_layernorm)
        self._swa_window_size = int(swa_window_size)
        self._architecture_config_json = (
            architecture_config_json
            if architecture_config_json is not None
            else json.dumps(
                {
                    "patch_size": validated_patch,
                    "res_channels": validated_res,
                    "skip_channels": validated_skip,
                    "out_channels": validated_out,
                    "num_res_layers": validated_layers,
                    "s4_lmax": int(s4_lmax),
                    "s4_d_state": int(s4_d_state),
                    "s4_bidirectional": bool(s4_bidirectional),
                    "s4_layernorm": bool(s4_layernorm),
                    "swa_window_size": int(swa_window_size),
                    "drop_prob": float(drop_prob),
                    "conv_out_chans": int(conv_out_chans),
                    "conv_groups": int(conv_groups),
                    "proj_kernel_size": int(proj_kernel_size),
                    "proj_padding": int(proj_padding),
                    "proj_refine_kernel": int(proj_refine_kernel),
                    "pos_kernel": list(pos_kernel),
                    "spectral_dropout": float(spectral_dropout),
                }
            )
        )

        refine_pad = proj_refine_kernel // 2
        pos_pad = (pos_kernel[0] // 2, pos_kernel[1] // 2)
        temporal_out_len = (validated_patch + 2 * proj_padding - proj_kernel_size) // conv_out_chans + 1
        emb_dim = conv_out_chans * temporal_out_len
        self._emb_dim = emb_dim

        patch_emb = nn.ModuleDict(
            {
                "proj_in": nn.Sequential(
                    nn.Conv2d(1, conv_out_chans, kernel_size=(1, proj_kernel_size), stride=(1, conv_out_chans), padding=(0, proj_padding)),
                    nn.GroupNorm(conv_groups, conv_out_chans),
                    nn.GELU(),
                    nn.Conv2d(conv_out_chans, conv_out_chans, kernel_size=(1, proj_refine_kernel), stride=(1, 1), padding=(0, refine_pad)),
                    nn.GroupNorm(conv_groups, conv_out_chans),
                    nn.GELU(),
                    nn.Conv2d(conv_out_chans, conv_out_chans, kernel_size=(1, proj_refine_kernel), stride=(1, 1), padding=(0, refine_pad)),
                    nn.GroupNorm(conv_groups, conv_out_chans),
                    nn.GELU(),
                ),
                "spectral_proj": nn.Sequential(
                    nn.Linear(validated_patch // 2 + 1, emb_dim),
                    nn.Dropout(spectral_dropout),
                ),
                "positional_encoding": nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim, kernel_size=pos_kernel, stride=(1, 1), padding=pos_pad, groups=emb_dim)
                ),
            }
        )
        patch_emb.register_parameter("mask_encoding", nn.Parameter(torch.zeros(emb_dim), requires_grad=False))
        self.patch_embedding = patch_emb

        self.init_conv = nn.ModuleList(
            [nn.ModuleDict({"conv": weight_norm(nn.Conv1d(emb_dim, validated_res, kernel_size=1))}), nn.ReLU()]
        )

        res_blocks = nn.ModuleList(
            [
                self._create_residual_block(
                    res_channels=validated_res,
                    skip_channels=validated_skip,
                    s4_lmax=int(s4_lmax),
                    s4_bidirectional=bool(s4_bidirectional),
                    s4_layernorm=bool(s4_layernorm),
                    s4_dropout=float(drop_prob),
                )
                for _ in range(validated_layers)
            ]
        )
        self.residual_layer = nn.ModuleDict({"residual_blocks": res_blocks})

        self.final_conv = nn.ModuleList(
            [
                nn.ModuleDict({"conv": weight_norm(nn.Conv1d(validated_skip, validated_skip, kernel_size=1))}),
                nn.ReLU(),
                nn.ModuleDict({"conv": nn.Conv1d(validated_skip, validated_out, kernel_size=1)}),
            ]
        )
        self.norm = nn.LayerNorm(validated_out)
        self.merged_lora_tails = nn.ModuleList()
        self.trainable_lora_tails = nn.ModuleList()

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def architecture_config_json(self) -> str:
        return self._architecture_config_json

    @staticmethod
    def _create_residual_block(
        res_channels: int,
        skip_channels: int,
        s4_lmax: int,
        s4_bidirectional: bool,
        s4_layernorm: bool,
        s4_dropout: float,
    ) -> nn.ModuleDict:
        d_model_sg = 2 * res_channels
        sg_channels = 4
        n_conv_channels = sg_channels * 2 if s4_bidirectional else sg_channels
        num_scales = 1 + math.ceil(math.log2(s4_lmax / 2.0))

        rms_mod = nn.ModuleDict()
        rms_mod.register_parameter("scale", nn.Parameter(torch.ones(1, res_channels, 1)))

        sg_mod = nn.ModuleDict(
            {
                "kernel_list": nn.ParameterList(
                    [nn.Parameter(torch.randn(n_conv_channels, d_model_sg, 2)) for _ in range(num_scales)]
                ),
                "activation": nn.GELU(),
                "dropout": nn.Dropout1d(s4_dropout) if s4_dropout > 0.0 else nn.Identity(),
                "norm": nn.LayerNorm(d_model_sg * sg_channels) if s4_layernorm else nn.Identity(),
                "output_linear": nn.Linear(d_model_sg * sg_channels, d_model_sg),
            }
        )
        sg_mod.register_parameter("skip_weight", nn.Parameter(torch.randn(sg_channels, d_model_sg)))
        sg_mod.register_buffer("multiplier", torch.linspace(2.0, 2.0, d_model_sg).view(1, -1, 1))
        sg_mod.register_buffer("kernel_norm", torch.ones(n_conv_channels, d_model_sg, 1))
        sg_mod.register_buffer("kernel_norm_initialized", torch.tensor(0, dtype=torch.bool))

        return nn.ModuleDict(
            {
                "rms_norm": rms_mod,
                "conv_layer": nn.ModuleDict(
                    {"conv": weight_norm(nn.Conv1d(res_channels, d_model_sg, kernel_size=3, padding=1))}
                ),
                "gelu": nn.GELU(),
                "sgconv": sg_mod,
                "attention": nn.MultiheadAttention(
                    embed_dim=d_model_sg,
                    num_heads=4,
                    dropout=s4_dropout,
                    bias=True,
                    batch_first=True,
                ),
                "res_conv": weight_norm(nn.Conv1d(res_channels, res_channels, kernel_size=1)),
                "skip_conv": weight_norm(nn.Conv1d(res_channels, skip_channels, kernel_size=1)),
            }
        )

    @staticmethod
    def _build_local_window_mask(seq_len: int, window_size: int, target_device: torch.device) -> torch.Tensor:
        half_win = window_size // 2
        positions = torch.arange(seq_len, device=target_device)
        distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        return torch.where(
            distances <= half_win,
            torch.zeros(1, device=target_device),
            torch.full((1,), float("-inf"), device=target_device),
        )

    def _compute_sgconv_kernel(self, sg_mod: nn.ModuleDict, seq_len: int) -> torch.Tensor:
        num_scales = len(sg_mod["kernel_list"])  # type: ignore[arg-type]
        kernel_parts: list[torch.Tensor] = []
        mult_buf: torch.Tensor = sg_mod.multiplier  # type: ignore[assignment]
        for scale_i in range(num_scales):
            scaled_part = F.interpolate(
                sg_mod["kernel_list"][scale_i],  # type: ignore[index]
                scale_factor=2 ** max(0, scale_i - 1),
                mode="linear",
            ) * (mult_buf ** (num_scales - scale_i - 1))
            kernel_parts.append(scaled_part)
        full_kernel = torch.cat(kernel_parts, dim=-1)
        if not bool(sg_mod.kernel_norm_initialized.item()):  # type: ignore[operator]
            sg_mod.kernel_norm.copy_(full_kernel.norm(dim=-1, keepdim=True).detach())  # type: ignore[operator]
            sg_mod.kernel_norm_initialized.fill_(True)  # type: ignore[operator]
        if full_kernel.size(-1) > seq_len:
            full_kernel = full_kernel[..., :seq_len]
        elif full_kernel.size(-1) < seq_len:
            full_kernel = F.pad(full_kernel, (0, seq_len - full_kernel.size(-1)))
        full_kernel = full_kernel / sg_mod.kernel_norm  # type: ignore[operator]
        if self._s4_bidirectional:
            half_ch = full_kernel.size(0) // 2
            k_fwd = full_kernel[:half_ch]
            k_bwd = full_kernel[half_ch:]
            full_kernel = F.pad(k_fwd, (0, seq_len)) + F.pad(k_bwd.flip(-1), (seq_len, 0))
        return full_kernel

    def _optimize_residual_blocks(self, block_list: nn.ModuleList, target_device: torch.device) -> None:
        with torch.no_grad():
            swa_mask_164 = self._build_local_window_mask(164, self._swa_window_size, target_device).detach()
            for blk_mod in block_list:
                sg_mod: nn.ModuleDict = blk_mod["sgconv"]  # type: ignore[index]
                kernel_164 = self._compute_sgconv_kernel(sg_mod, 164).to(target_device)
                kf_164 = torch.fft.rfft(kernel_164.float(), n=328).detach()
                skip_w = sg_mod.skip_weight.unsqueeze(0).unsqueeze(-1).detach()  # type: ignore[attr-defined]
                blk_mod.__dict__["_fast_kf164"] = kf_164
                blk_mod.__dict__["_fast_skip_w"] = skip_w
                blk_mod.__dict__["_fast_swa_mask164"] = swa_mask_164

    def _forward_residual_block(
        self,
        blk_mod: nn.ModuleDict,
        running_h: torch.Tensor,
        original_noise: torch.Tensor,
        lora_spec: nn.ModuleDict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = running_h.size(-1)
        rms_scale: torch.Tensor = blk_mod["rms_norm"].scale  # type: ignore[index,attr-defined]
        rms_denom = running_h.norm(dim=1, keepdim=True) / (running_h.shape[1] ** 0.5) + 1e-8
        x_norm = rms_scale * (running_h / rms_denom)

        conv_in = running_h + original_noise
        conv_out = blk_mod["conv_layer"]["conv"](conv_in)  # type: ignore[index]
        if lora_spec is not None:
            conv_scale = float(lora_spec.lora_scale_buf.item())  # type: ignore[attr-defined]
            conv_out = conv_out + conv_scale * lora_spec["conv_b"](lora_spec["conv_a"](conv_in))

        hidden = F.gelu(conv_out)
        sg_mod: nn.ModuleDict = blk_mod["sgconv"]  # type: ignore[index]

        if seq_len == 164 and "_fast_kf164" in blk_mod.__dict__ and not self.training:
            kf_164 = blk_mod.__dict__["_fast_kf164"]
            skip_w = blk_mod.__dict__["_fast_skip_w"]
            swa_mask = blk_mod.__dict__["_fast_swa_mask164"]
            x_freq = torch.fft.rfft(hidden.float(), n=328)
            out_sg = torch.fft.irfft(x_freq.unsqueeze(1) * kf_164.unsqueeze(0), n=328)[..., :164]
            out_sg = (out_sg + hidden.unsqueeze(1) * skip_w).reshape(hidden.size(0), -1, 164)
            out_sg = sg_mod["output_linear"](sg_mod["norm"](sg_mod["activation"](out_sg).transpose(1, 2))).transpose(1, 2)
        else:
            kern = self._compute_sgconv_kernel(sg_mod, seq_len)
            k_freq = torch.fft.rfft(kern.float(), n=2 * seq_len)
            x_freq = torch.fft.rfft(hidden.float(), n=2 * seq_len)
            out_freq = torch.einsum("bhl,chl->bchl", x_freq, k_freq)
            out_sg = torch.fft.irfft(out_freq, n=2 * seq_len)[..., :seq_len]
            out_sg = (out_sg + torch.einsum("bhl,ch->bchl", hidden, sg_mod.skip_weight)).reshape(  # type: ignore[attr-defined]
                hidden.size(0), -1, seq_len
            )
            out_sg = sg_mod["dropout"](sg_mod["activation"](out_sg))
            out_sg = sg_mod["output_linear"](sg_mod["norm"](out_sg.transpose(1, 2))).transpose(1, 2)
            swa_mask = self._build_local_window_mask(seq_len, self._swa_window_size, running_h.device)

        sglora_mat = blk_mod.__dict__.get("_r32_sglora", None)
        if sglora_mat is not None:
            out_sg = out_sg + torch.matmul(sglora_mat, out_sg)
        if lora_spec is not None:
            sg_scale = float(lora_spec.lora_scale_buf.item())  # type: ignore[attr-defined]
            out_sg = out_sg + sg_scale * lora_spec["sgconv_b"](lora_spec["sgconv_a"](out_sg))

        attn_in = out_sg.transpose(1, 2)
        attn_out, _ = blk_mod["attention"](attn_in, attn_in, attn_in, attn_mask=swa_mask, need_weights=False)
        if lora_spec is not None:
            at_scale = float(lora_spec.lora_scale_buf.item())  # type: ignore[attr-defined]
            attn_out = attn_out + at_scale * lora_spec["attn_b"](lora_spec["attn_a"](attn_out))

        combined = out_sg + attn_out.transpose(1, 2)
        gated = torch.tanh(combined[:, : self._res_channels, :]) * torch.sigmoid(combined[:, self._res_channels :, :])
        res_out = blk_mod["res_conv"](gated)
        skip_out = blk_mod["skip_conv"](gated)
        if lora_spec is not None:
            rs_scale = float(lora_spec.lora_scale_buf.item())  # type: ignore[attr-defined]
            res_out = res_out + rs_scale * lora_spec["res_conv_b"](lora_spec["res_conv_a"](gated))
            skip_out = skip_out + rs_scale * lora_spec["skip_conv_b"](lora_spec["skip_conv_a"](gated))

        return (x_norm + res_out) * math.sqrt(0.5), skip_out

    def _forward_patch_and_stem(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        batch_sz, chan_cnt, sample_cnt = inputs.shape
        seq_len = sample_cnt // self._patch_size
        patched_x = inputs[:, :, : seq_len * self._patch_size].reshape(batch_sz, chan_cnt, seq_len, self._patch_size)

        merged_conv = patched_x.contiguous().view(batch_sz, 1, chan_cnt * seq_len, self._patch_size)
        patch_emb = (
            self.patch_embedding["proj_in"](merged_conv)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_sz, chan_cnt, seq_len, self._emb_dim)
        )
        freq_spec = torch.abs(
            torch.fft.rfft(
                patched_x.contiguous().view(batch_sz * chan_cnt * seq_len, self._patch_size).float(),
                dim=-1,
                norm="forward",
            )
        ).view(batch_sz, chan_cnt, seq_len, self._patch_size // 2 + 1)
        patch_emb = patch_emb + self.patch_embedding["spectral_proj"](freq_spec)
        pos_emb = self.patch_embedding["positional_encoding"](
            patch_emb.permute(0, 3, 1, 2).contiguous()
        ).permute(0, 2, 3, 1)
        patch_emb = patch_emb + pos_emb

        flat_seq = patch_emb.reshape(batch_sz, chan_cnt * seq_len, self._emb_dim).transpose(1, 2).contiguous()
        noise = F.relu(self.init_conv[0]["conv"](flat_seq))  # type: ignore[index]

        running_h = noise
        running_skip = torch.zeros_like(noise)
        stem_blocks = min(5, self._num_res_layers)
        for blk_idx in range(stem_blocks):
            running_h, skip_step = self._forward_residual_block(
                self.residual_layer["residual_blocks"][blk_idx],  # type: ignore[index]
                running_h,
                noise,
            )
            running_skip = skip_step + running_skip
        return running_h, running_skip, noise, chan_cnt, seq_len

    @staticmethod
    def _forward_final_conv_and_norm(
        final_conv_list: nn.ModuleList,
        norm_layer: nn.LayerNorm,
        scaled_skip: torch.Tensor,
        batch_sz: int,
        chan_cnt: int,
        seq_len: int,
        out_channels: int,
        fc_lora_spec: nn.ModuleDict | None = None,
    ) -> torch.Tensor:
        fc0_out = final_conv_list[0]["conv"](scaled_skip)  # type: ignore[index]
        if fc_lora_spec is not None:
            sc0 = float(fc_lora_spec.lora_scale_buf.item())  # type: ignore[attr-defined]
            fc0_out = fc0_out + sc0 * fc_lora_spec["fc0_b"](fc_lora_spec["fc0_a"](scaled_skip))
        act_out = F.relu(fc0_out)
        fc2_out = final_conv_list[2]["conv"](act_out)  # type: ignore[index]
        if fc_lora_spec is not None:
            sc2 = float(fc_lora_spec.lora_scale_buf.item())  # type: ignore[attr-defined]
            fc2_out = fc2_out + sc2 * fc_lora_spec["fc2_b"](fc_lora_spec["fc2_a"](act_out))
        reshaped = fc2_out.transpose(1, 2).contiguous().view(batch_sz, chan_cnt, seq_len, out_channels)
        return norm_layer(reshaped)

    def finalize_frozen_stem_and_tail(self, execution_device: torch.device) -> None:
        self.to(execution_device).eval()
        with torch.no_grad():
            _ = self.forward_frozen_only(torch.zeros((1, 41, 800), dtype=torch.float32, device=execution_device))
        for sub_mod in self.modules():
            if hasattr(sub_mod, "parametrizations") and "weight" in sub_mod.parametrizations:
                remove_parametrizations(sub_mod, "weight", leave_parametrized=True)
        for param_tensor in self.parameters():
            param_tensor.requires_grad_(False)
        self._optimize_residual_blocks(self.residual_layer["residual_blocks"], execution_device)  # type: ignore[arg-type]

    def attach_merged_lora_tails_from_npz(
        self, lora_npz_path: str | Path, execution_device: torch.device
    ) -> tuple[list[dict[str, np.ndarray]], float, float]:
        """Fold Rank-8 LoRA deltas into cloned tail blocks (5..7) + final_conv + norm, returning head readout metadata."""
        resolved_lora = ParameterTypeValidator.require_path(lora_npz_path, "lora_npz_path", must_exist=True)
        self.merged_lora_tails = nn.ModuleList()
        tail_readouts: list[dict[str, np.ndarray]] = []
        with np.load(resolved_lora, allow_pickle=False) as archive, torch.no_grad():
            seed_count = int(archive["n_seeds"])
            n0_cent = float(archive["n0_cent"])
            prior_bal = float(archive["prior_bal"])
            for seed_idx in range(seed_count):
                pfx = f"s{seed_idx}__"
                tail_blocks = nn.ModuleList(
                    [copy.deepcopy(self.residual_layer["residual_blocks"][bi]) for bi in (5, 6, 7)]  # type: ignore[index]
                ).to(execution_device).eval()
                tail_fc = copy.deepcopy(self.final_conv).to(execution_device).eval()
                tail_norm = copy.deepcopy(self.norm).to(execution_device).eval()

                for loc_i, blk_idx in enumerate((5, 6, 7)):
                    blk_s: nn.ModuleDict = tail_blocks[loc_i]  # type: ignore[assignment]
                    rn_sd = {
                        k_str.split("__rms_norm__")[1]: torch.from_numpy(archive[k_str]).float().to(execution_device)
                        for k_str in archive.files
                        if k_str.startswith(f"{pfx}b{blk_idx}__rms_norm__")
                    }
                    blk_s["rms_norm"].load_state_dict(rn_sd, strict=True)

                    conv_a = torch.from_numpy(archive[f"{pfx}b{blk_idx}__conv__a"]).float().to(execution_device)
                    conv_b = torch.from_numpy(archive[f"{pfx}b{blk_idx}__conv__b"]).float().to(execution_device)
                    conv_sc = float(archive[f"{pfx}b{blk_idx}__conv__scale"])
                    blk_s["conv_layer"]["conv"].weight.add_(conv_sc * torch.einsum("or,rik->oik", conv_b[:, :, 0], conv_a))  # type: ignore[index]

                    sg_a = torch.from_numpy(archive[f"{pfx}b{blk_idx}__sgconv__a"]).float().to(execution_device)[:, :, 0]
                    sg_b = torch.from_numpy(archive[f"{pfx}b{blk_idx}__sgconv__b"]).float().to(execution_device)[:, :, 0]
                    sg_sc = float(archive[f"{pfx}b{blk_idx}__sgconv__scale"])
                    blk_s.__dict__["_r32_sglora"] = (sg_sc * (sg_b @ sg_a)).detach()

                    att_a = torch.from_numpy(archive[f"{pfx}b{blk_idx}__attn__a"]).float().to(execution_device)
                    att_b = torch.from_numpy(archive[f"{pfx}b{blk_idx}__attn__b"]).float().to(execution_device)
                    att_sc = float(archive[f"{pfx}b{blk_idx}__attn__scale"])
                    att_mult = torch.eye(att_a.shape[1], device=execution_device) + att_sc * (att_b @ att_a)
                    blk_s["attention"].out_proj.weight.copy_(att_mult @ blk_s["attention"].out_proj.weight)
                    if blk_s["attention"].out_proj.bias is not None:
                        blk_s["attention"].out_proj.bias.copy_(att_mult @ blk_s["attention"].out_proj.bias)

                    for conv_tag in ("res_conv", "skip_conv"):
                        c_a = torch.from_numpy(archive[f"{pfx}b{blk_idx}__{conv_tag}__a"]).float().to(execution_device)[:, :, 0]
                        c_b = torch.from_numpy(archive[f"{pfx}b{blk_idx}__{conv_tag}__b"]).float().to(execution_device)[:, :, 0]
                        c_sc = float(archive[f"{pfx}b{blk_idx}__{conv_tag}__scale"])
                        blk_s[conv_tag].weight.add_((c_sc * (c_b @ c_a)).unsqueeze(-1))

                for fc_idx in (0, 2):
                    fc_a = torch.from_numpy(archive[f"{pfx}fc{fc_idx}__a"]).float().to(execution_device)[:, :, 0]
                    fc_b = torch.from_numpy(archive[f"{pfx}fc{fc_idx}__b"]).float().to(execution_device)[:, :, 0]
                    fc_sc = float(archive[f"{pfx}fc{fc_idx}__scale"])
                    tail_fc[fc_idx]["conv"].weight.add_((fc_sc * (fc_b @ fc_a)).unsqueeze(-1))  # type: ignore[index]

                norm_sd = {
                    k_str.split("__norm__")[1]: torch.from_numpy(archive[k_str]).float().to(execution_device)
                    for k_str in archive.files
                    if k_str.startswith(f"{pfx}norm__")
                }
                tail_norm.load_state_dict(norm_sd, strict=True)
                for p_tail in (*tail_blocks.parameters(), *tail_fc.parameters(), *tail_norm.parameters()):
                    p_tail.requires_grad_(False)
                self._optimize_residual_blocks(tail_blocks, execution_device)

                self.merged_lora_tails.append(
                    nn.ModuleDict({"blocks": tail_blocks, "final_conv": tail_fc, "norm": tail_norm})
                )
                tail_readouts.append(
                    {
                        "mu0": archive[f"{pfx}head__mu0"].astype(np.float64),
                        "W_eff": archive[f"{pfx}head__W_eff"].astype(np.float64),
                        "b_eff": archive[f"{pfx}head__b_eff"].astype(np.float64),
                    }
                )
        return tail_readouts, n0_cent, prior_bal

    def build_trainable_lora_tail_adapter(
        self, lora_rank: int = 8, lora_alpha: float = 8.0, execution_device: str | torch.device = "cpu"
    ) -> nn.ModuleDict:
        """Create a trainable Rank-`lora_rank` LoRA adapter over blocks 5..7 + final_conv + norm + auxiliary head."""
        dev = torch.device(execution_device)
        scale_val = float(lora_alpha) / float(lora_rank)
        block_loras = nn.ModuleList()
        for blk_idx in (5, 6, 7):
            src_blk: nn.ModuleDict = self.residual_layer["residual_blocks"][blk_idx]  # type: ignore[index]
            blk_lora = nn.ModuleDict(
                {
                    "rms_norm": copy.deepcopy(src_blk["rms_norm"]),
                    "conv_a": nn.Conv1d(self._res_channels, lora_rank, kernel_size=3, padding=1, bias=False),
                    "conv_b": nn.Conv1d(lora_rank, 2 * self._res_channels, kernel_size=1, bias=False),
                    "sgconv_a": nn.Conv1d(2 * self._res_channels, lora_rank, kernel_size=1, bias=False),
                    "sgconv_b": nn.Conv1d(lora_rank, 2 * self._res_channels, kernel_size=1, bias=False),
                    "attn_a": nn.Linear(2 * self._res_channels, lora_rank, bias=False),
                    "attn_b": nn.Linear(lora_rank, 2 * self._res_channels, bias=False),
                    "res_conv_a": nn.Conv1d(self._res_channels, lora_rank, kernel_size=1, bias=False),
                    "res_conv_b": nn.Conv1d(lora_rank, self._res_channels, kernel_size=1, bias=False),
                    "skip_conv_a": nn.Conv1d(self._res_channels, lora_rank, kernel_size=1, bias=False),
                    "skip_conv_b": nn.Conv1d(lora_rank, self._skip_channels, kernel_size=1, bias=False),
                }
            )
            blk_lora.register_buffer("lora_scale_buf", torch.tensor(scale_val, dtype=torch.float32))
            for key_a in ("conv_a", "sgconv_a", "attn_a", "res_conv_a", "skip_conv_a"):
                nn.init.normal_(blk_lora[key_a].weight, std=0.02)
            for key_b in ("conv_b", "sgconv_b", "attn_b", "res_conv_b", "skip_conv_b"):
                nn.init.zeros_(blk_lora[key_b].weight)
            for p_rn in blk_lora["rms_norm"].parameters():
                p_rn.requires_grad_(True)
            block_loras.append(blk_lora)

        fc_lora = nn.ModuleDict(
            {
                "fc0_a": nn.Conv1d(self._skip_channels, lora_rank, kernel_size=1, bias=False),
                "fc0_b": nn.Conv1d(lora_rank, self._skip_channels, kernel_size=1, bias=False),
                "fc2_a": nn.Conv1d(self._skip_channels, lora_rank, kernel_size=1, bias=False),
                "fc2_b": nn.Conv1d(lora_rank, self._out_channels, kernel_size=1, bias=False),
            }
        )
        fc_lora.register_buffer("lora_scale_buf", torch.tensor(scale_val, dtype=torch.float32))
        nn.init.normal_(fc_lora["fc0_a"].weight, std=0.02)
        nn.init.zeros_(fc_lora["fc0_b"].weight)
        nn.init.normal_(fc_lora["fc2_a"].weight, std=0.02)
        nn.init.zeros_(fc_lora["fc2_b"].weight)

        norm_copy = copy.deepcopy(self.norm)
        for p_nm in norm_copy.parameters():
            p_nm.requires_grad_(True)

        feat_dim = 41 * 12 + 41 * 12 + 32
        readout_mod = nn.ModuleDict(
            {
                "ln_late": nn.LayerNorm(self._out_channels),
                "ln_rms": nn.LayerNorm(self._out_channels),
                "proj_late": nn.Linear(self._out_channels, 12),
                "proj_rms": nn.Linear(self._out_channels, 12),
                "proj_glob": nn.Linear(2 * self._out_channels, 32),
                "ln_out": nn.LayerNorm(feat_dim),
                "drop": nn.Dropout(0.35),
                "fc": nn.Linear(feat_dim, 3),
                "aux_prot_mr": nn.Linear(feat_dim, 2),
                "aux_dreyer_lr": nn.Linear(feat_dim, 2),
                "aux_dreyer_mr": nn.Linear(feat_dim, 2),
                "aux_shin18": nn.Linear(feat_dim, 2),
                "aux_zyma19": nn.Linear(feat_dim, 2),
            }
        )
        return nn.ModuleDict(
            {"block_loras": block_loras, "fc_lora": fc_lora, "norm": norm_copy, "readout": readout_mod}
        ).to(dev)

    def forward_with_trainable_lora(
        self, inputs: torch.Tensor, lora_adapter: nn.ModuleDict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run frozen stem (0..4) + trainable LoRA tail (5..7), returning `(tokens [B,41,4,200], feat_1016 [B,1016])`."""
        with torch.no_grad():
            stem_h, stem_skip, noise, chan_cnt, seq_len = self._forward_patch_and_stem(inputs)
        batch_sz = inputs.shape[0]
        skip_scale = float(math.sqrt(1.0 / float(self._num_res_layers)))
        cur_h, cur_skip = stem_h, stem_skip
        for loc_i, blk_idx in enumerate((5, 6, 7)):
            base_blk: nn.ModuleDict = self.residual_layer["residual_blocks"][blk_idx]  # type: ignore[index]
            blk_lora: nn.ModuleDict = lora_adapter["block_loras"][loc_i]  # type: ignore[index]
            # Temporarily swap rms_norm view to trainable rms_norm
            rms_scale: torch.Tensor = blk_lora["rms_norm"].scale  # type: ignore[index,attr-defined]
            rms_denom = cur_h.norm(dim=1, keepdim=True) / (cur_h.shape[1] ** 0.5) + 1e-8
            x_norm = rms_scale * (cur_h / rms_denom)

            conv_in = cur_h + noise
            l_sc = float(blk_lora.lora_scale_buf.item())  # type: ignore[attr-defined]
            conv_out = base_blk["conv_layer"]["conv"](conv_in) + l_sc * blk_lora["conv_b"](blk_lora["conv_a"](conv_in))  # type: ignore[index]
            hidden = F.gelu(conv_out)
            sg_mod: nn.ModuleDict = base_blk["sgconv"]  # type: ignore[index]
            kern = self._compute_sgconv_kernel(sg_mod, hidden.size(-1))
            s_len = hidden.size(-1)
            k_freq = torch.fft.rfft(kern.float(), n=2 * s_len)
            x_freq = torch.fft.rfft(hidden.float(), n=2 * s_len)
            out_sg = torch.fft.irfft(torch.einsum("bhl,chl->bchl", x_freq, k_freq), n=2 * s_len)[..., :s_len]
            out_sg = (out_sg + torch.einsum("bhl,ch->bchl", hidden, sg_mod.skip_weight)).reshape(  # type: ignore[attr-defined]
                hidden.size(0), -1, s_len
            )
            out_sg = sg_mod["output_linear"](sg_mod["norm"](sg_mod["activation"](out_sg).transpose(1, 2))).transpose(1, 2)
            out_sg = out_sg + l_sc * blk_lora["sgconv_b"](blk_lora["sgconv_a"](out_sg))

            swa_mask = self._build_local_window_mask(s_len, self._swa_window_size, cur_h.device)
            att_in = out_sg.transpose(1, 2)
            att_out, _ = base_blk["attention"](att_in, att_in, att_in, attn_mask=swa_mask, need_weights=False)
            att_out = att_out + l_sc * blk_lora["attn_b"](blk_lora["attn_a"](att_out))
            comb = out_sg + att_out.transpose(1, 2)
            gated = torch.tanh(comb[:, : self._res_channels, :]) * torch.sigmoid(comb[:, self._res_channels :, :])
            res_o = base_blk["res_conv"](gated) + l_sc * blk_lora["res_conv_b"](blk_lora["res_conv_a"](gated))
            skip_o = base_blk["skip_conv"](gated) + l_sc * blk_lora["skip_conv_b"](blk_lora["skip_conv_a"](gated))
            cur_h = (x_norm + res_o) * math.sqrt(0.5)
            cur_skip = skip_o + cur_skip

        tokens = self._forward_final_conv_and_norm(
            self.final_conv,
            lora_adapter["norm"],  # type: ignore[arg-type]
            cur_skip * skip_scale,
            batch_sz,
            chan_cnt,
            seq_len,
            self._out_channels,
            fc_lora_spec=lora_adapter["fc_lora"],  # type: ignore[arg-type]
        )
        ro_mod: nn.ModuleDict = lora_adapter["readout"]  # type: ignore[assignment]
        h_late = tokens[:, :, 1:, :].mean(dim=2)
        h_rms = torch.sqrt((tokens.float() ** 2).mean(dim=2) + 1e-6)
        z_late = F.gelu(ro_mod["proj_late"](ro_mod["ln_late"](h_late))).reshape(batch_sz, -1)
        z_rms = F.gelu(ro_mod["proj_rms"](ro_mod["ln_rms"](h_rms))).reshape(batch_sz, -1)
        z_glob = F.gelu(ro_mod["proj_glob"](torch.cat([h_late.mean(dim=1), h_rms.mean(dim=1)], dim=-1)))
        feat_1016 = ro_mod["ln_out"](torch.cat([z_late, z_rms, z_glob], dim=-1))
        return tokens, feat_1016

    @staticmethod
    def export_lora_adapter_arrays(
        lora_adapter: nn.ModuleDict,
        seed_index: int,
        mu0: np.ndarray,
        w_eff: np.ndarray,
        b_eff: np.ndarray,
    ) -> dict[str, np.ndarray]:
        pfx = f"s{seed_index}__"
        exported: dict[str, np.ndarray] = {
            f"{pfx}head__mu0": mu0.astype(np.float32),
            f"{pfx}head__W_eff": w_eff.astype(np.float16),
            f"{pfx}head__b_eff": b_eff.astype(np.float32),
        }
        for loc_i, blk_idx in enumerate((5, 6, 7)):
            blk_l: nn.ModuleDict = lora_adapter["block_loras"][loc_i]  # type: ignore[index]
            l_sc = float(blk_l.lora_scale_buf.item())  # type: ignore[attr-defined]
            for rn_key, rn_val in blk_l["rms_norm"].state_dict().items():
                exported[f"{pfx}b{blk_idx}__rms_norm__{rn_key}"] = rn_val.detach().cpu().float().numpy()
            for mod_tag in ("conv", "sgconv", "attn", "res_conv", "skip_conv"):
                exported[f"{pfx}b{blk_idx}__{mod_tag}__a"] = blk_l[f"{mod_tag}_a"].weight.detach().cpu().half().numpy()
                exported[f"{pfx}b{blk_idx}__{mod_tag}__b"] = blk_l[f"{mod_tag}_b"].weight.detach().cpu().half().numpy()
                exported[f"{pfx}b{blk_idx}__{mod_tag}__scale"] = np.array(l_sc, dtype=np.float32)

        fc_l: nn.ModuleDict = lora_adapter["fc_lora"]  # type: ignore[assignment]
        fc_sc = float(fc_l.lora_scale_buf.item())  # type: ignore[attr-defined]
        for fc_idx in (0, 2):
            exported[f"{pfx}fc{fc_idx}__a"] = fc_l[f"fc{fc_idx}_a"].weight.detach().cpu().half().numpy()
            exported[f"{pfx}fc{fc_idx}__b"] = fc_l[f"fc{fc_idx}_b"].weight.detach().cpu().half().numpy()
            exported[f"{pfx}fc{fc_idx}__scale"] = np.array(fc_sc, dtype=np.float32)
        for nm_key, nm_val in lora_adapter["norm"].state_dict().items():
            exported[f"{pfx}norm__{nm_key}"] = nm_val.detach().cpu().float().numpy()
        return exported

    @classmethod
    def from_npz_archive(
        cls,
        codebrain_npz_path: str | Path,
        execution_device: str | torch.device = "cpu",
        lora_npz_path: str | Path | None = None,
    ) -> tuple[
        CodebrainFoundationBackbone,
        dict[str, object],
        list[dict[str, np.ndarray]] | None,
        float,
        float,
    ]:
        resolved_npz = ParameterTypeValidator.require_path(codebrain_npz_path, "codebrain_npz_path", must_exist=True)
        dev = torch.device(execution_device)
        with np.load(resolved_npz, allow_pickle=False) as archive:
            cfg_raw = str(archive["cfg_json"].item())
            cfg_dict = json.loads(cfg_raw)
            backbone = cls(
                patch_size=int(cfg_dict["patch_size"]),
                res_channels=int(cfg_dict["res_channels"]),
                skip_channels=int(cfg_dict["skip_channels"]),
                out_channels=int(cfg_dict["out_channels"]),
                num_res_layers=int(cfg_dict["num_res_layers"]),
                s4_lmax=int(cfg_dict["s4_lmax"]),
                s4_d_state=int(cfg_dict["s4_d_state"]),
                s4_bidirectional=bool(cfg_dict["s4_bidirectional"]),
                s4_layernorm=bool(cfg_dict["s4_layernorm"]),
                swa_window_size=int(cfg_dict["swa_window_size"]),
                drop_prob=float(cfg_dict.get("drop_prob", 0.1)),
                conv_out_chans=int(cfg_dict.get("conv_out_chans", 25)),
                conv_groups=int(cfg_dict.get("conv_groups", 5)),
                proj_kernel_size=int(cfg_dict.get("proj_kernel_size", 49)),
                proj_padding=int(cfg_dict.get("proj_padding", 24)),
                proj_refine_kernel=int(cfg_dict.get("proj_refine_kernel", 3)),
                pos_kernel=tuple(int(v_k) for v_k in cfg_dict.get("pos_kernel", [19, 7])),  # type: ignore[arg-type]
                spectral_dropout=float(cfg_dict.get("spectral_dropout", 0.1)),
                architecture_config_json=cfg_raw,
            )
            enc_sd: dict[str, torch.Tensor] = {}
            for key_str in archive.files:
                if not key_str.startswith("enc__"):
                    continue
                short_key = key_str[5:]
                raw_arr = archive[key_str]
                if short_key.endswith("kernel_norm_initialized"):
                    enc_sd[short_key] = torch.tensor(bool(float(raw_arr) != 0.0), dtype=torch.bool)
                else:
                    enc_sd[short_key] = torch.from_numpy(raw_arr.astype(np.float32))
            backbone.load_state_dict(enc_sd, strict=True)

            for blk_idx, blk_mod in enumerate(backbone.residual_layer["residual_blocks"]):  # type: ignore[arg-type]
                if f"sglora__{blk_idx}" in archive.files:
                    blk_mod.__dict__["_r32_sglora"] = torch.from_numpy(
                        archive[f"sglora__{blk_idx}"].astype(np.float32)
                    ).to(dev)

            frozen_head_meta: dict[str, object] = {
                "mu0": archive["head__mu0"].astype(np.float64),
                "W_eff": archive["head__W_eff"].astype(np.float64),
                "b_eff": archive["head__b_eff"].astype(np.float64),
                "pool_mode": str(archive["head__pool_mode"].item()),
                "n0_cent": float(archive["head__n0_cent"].item()),
                "prior_bal": float(archive["head__prior_bal"].item()),
            }

        backbone.finalize_frozen_stem_and_tail(dev)
        lora_readouts: list[dict[str, np.ndarray]] | None = None
        lora_n0_cent = 3.0
        lora_prior_bal = 10.0
        if lora_npz_path is not None and Path(lora_npz_path).exists():
            lora_readouts, lora_n0_cent, lora_prior_bal = backbone.attach_merged_lora_tails_from_npz(lora_npz_path, dev)

        return backbone, frozen_head_meta, lora_readouts, lora_n0_cent, lora_prior_bal

    def forward_frozen_only(self, inputs: torch.Tensor) -> torch.Tensor:
        stem_h, stem_skip, noise, chan_cnt, seq_len = self._forward_patch_and_stem(inputs)
        batch_sz = inputs.shape[0]
        skip_scale = float(math.sqrt(1.0 / float(self._num_res_layers)))
        frz_h, frz_skip = stem_h, stem_skip
        for blk_idx in range(min(5, self._num_res_layers), self._num_res_layers):
            frz_h, skip_step = self._forward_residual_block(
                self.residual_layer["residual_blocks"][blk_idx],  # type: ignore[index]
                frz_h,
                noise,
            )
            frz_skip = skip_step + frz_skip
        return self._forward_final_conv_and_norm(
            self.final_conv,
            self.norm,
            frz_skip * skip_scale,
            batch_sz,
            chan_cnt,
            seq_len,
            self._out_channels,
        )

    def encode_window_features(
        self, *tensor_inputs: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if len(tensor_inputs) != 1:
            raise ValueError("CodebrainFoundationBackbone expects a single (B, C, T) tensor")
        inputs = tensor_inputs[0]
        stem_h, stem_skip, noise, chan_cnt, seq_len = self._forward_patch_and_stem(inputs)
        batch_sz = inputs.shape[0]
        skip_scale = float(math.sqrt(1.0 / float(self._num_res_layers)))

        frz_h, frz_skip = stem_h, stem_skip
        for blk_idx in range(min(5, self._num_res_layers), self._num_res_layers):
            frz_h, skip_step = self._forward_residual_block(
                self.residual_layer["residual_blocks"][blk_idx],  # type: ignore[index]
                frz_h,
                noise,
            )
            frz_skip = skip_step + frz_skip
        frozen_tokens = self._forward_final_conv_and_norm(
            self.final_conv,
            self.norm,
            frz_skip * skip_scale,
            batch_sz,
            chan_cnt,
            seq_len,
            self._out_channels,
        )

        lora_token_list: list[torch.Tensor] = []
        for tail_mod in self.merged_lora_tails:
            tail_h, tail_skip = stem_h, stem_skip
            for blk_s in tail_mod["blocks"]:  # type: ignore[index]
                tail_h, skip_step = self._forward_residual_block(blk_s, tail_h, noise)
                tail_skip = skip_step + tail_skip
            tail_tokens = self._forward_final_conv_and_norm(
                tail_mod["final_conv"],  # type: ignore[index]
                tail_mod["norm"],  # type: ignore[index]
                tail_skip * skip_scale,
                batch_sz,
                chan_cnt,
                seq_len,
                self._out_channels,
            )
            lora_token_list.append(tail_tokens)

        return frozen_tokens, lora_token_list

    def forward(self, inputs: torch.Tensor, return_features: bool = True) -> dict[str, torch.Tensor | None] | torch.Tensor:
        frozen_tokens = self.forward_frozen_only(inputs)
        if return_features:
            return {"features": frozen_tokens, "cls_token": None}
        return frozen_tokens
