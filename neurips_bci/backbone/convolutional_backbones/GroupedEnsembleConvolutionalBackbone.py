"""Batched N-seed grouped Conv2d/BMM ensemble backbone with PseudoChrono-BN, Tent, AG3, and self-training."""

from __future__ import annotations

import contextlib
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from neurips_bci.backbone.convolutional_backbones.AdaptiveBatchNormConvolutionalBackbone import (
    AdaptiveBatchNormConvolutionalBackbone,
)
from neurips_bci.backbone.convolutional_backbones.ConvolutionalBackbone import ConvolutionalBackbone
from neurips_bci.backbone.convolutional_backbones.FilterBankConvolutionalBackbone import (
    FilterBankConvolutionalBackbone,
)
from neurips_bci.backbone.convolutional_backbones.ShallowTemporalConvolutionalBackbone import (
    ShallowTemporalConvolutionalBackbone,
)


class GroupedEnsembleConvolutionalBackbone(ConvolutionalBackbone):
    """Stacks N homogeneous CNN seeds into one grouped network for fast B=1 inference, Tent, AG3, and replay ST."""

    def __init__(self, seed_networks: list[nn.Module] | tuple[nn.Module, ...]) -> None:
        if not seed_networks:
            raise ValueError("seed_networks must contain at least one network")
        first_net = seed_networks[0]
        is_shallow = isinstance(first_net, ShallowTemporalConvolutionalBackbone) or (
            hasattr(first_net, "temporal") and hasattr(first_net, "pool")
        )
        kind_tag = "shallow" if is_shallow else "fbcnet"
        super().__init__(backbone_architecture=f"grouped_{kind_tag}_x{len(seed_networks)}")

        if not all(type(member_net) is type(first_net) for member_net in seed_networks):
            raise TypeError("All seed_networks in GroupedEnsembleConvolutionalBackbone must share the exact class")

        self.kind = kind_tag
        self.n = len(seed_networks)
        self.version = 0
        first_bn = first_net.bn
        self.eps = float(first_bn.eps)
        self.adapt = float(first_bn.adapt)
        self.p_drop = float(first_net.drop.p)

        state_dicts = [member_net.state_dict() for member_net in seed_networks]
        ref_keys = state_dicts[0].keys()
        for other_sd in state_dicts[1:]:
            if other_sd.keys() != ref_keys or any(
                other_sd[k_str].shape != state_dicts[0][k_str].shape for k_str in ref_keys
            ):
                raise ValueError("Seed network state_dict schemas or shapes differ")

        if self.kind == "shallow":
            self.temporal_w = nn.Parameter(torch.cat([sd["temporal.weight"] for sd in state_dicts]).clone())
            self.temporal_b = nn.Parameter(torch.cat([sd["temporal.bias"] for sd in state_dicts]).clone())
            self.pool = (first_net.pool.kernel_size, first_net.pool.stride)
            self.modpool = False
            self.mod_fir = None
            self.n_segments = 0
        else:
            if not all(torch.equal(member_net.fir, first_net.fir) for member_net in seed_networks):
                raise ValueError("FilterBankConvolutionalBackbone seeds must share identical FIR filter banks")
            self.modpool = bool(getattr(first_net, "modpool", False))
            self.register_buffer("fir", first_net.fir.clone())
            if self.modpool and getattr(first_net, "mod_fir", None) is not None:
                self.register_buffer("mod_fir", first_net.mod_fir.clone())
            else:
                self.mod_fir = None
            self.n_segments = int(first_net.n_segments)

        self.spatial_w = nn.Parameter(torch.cat([sd["spatial.weight"] for sd in state_dicts]).clone())
        self.width = int(self.spatial_w.shape[0]) // self.n
        self.bn_w = nn.Parameter(torch.cat([sd["bn.weight"] for sd in state_dicts]).clone())
        self.bn_b = nn.Parameter(torch.cat([sd["bn.bias"] for sd in state_dicts]).clone())
        self.register_buffer(
            "bn_mean",
            torch.cat([sd["bn.running_mean"] for sd in state_dicts]).clone(),
        )
        self.register_buffer(
            "bn_var",
            torch.cat([sd["bn.running_var"] for sd in state_dicts]).clone(),
        )
        self.fc_w = nn.Parameter(torch.stack([sd["fc.weight"] for sd in state_dicts]).clone())
        self.fc_b = nn.Parameter(torch.stack([sd["fc.bias"] for sd in state_dicts]).clone())
        self._tta_chunk_buf: torch.Tensor | None = None
        self.to(state_dicts[0]["fc.weight"].device)

    @staticmethod
    @contextlib.contextmanager
    def _cuda_tf32_matmul_context(enable_tf32: bool):
        if not enable_tf32 or not torch.cuda.is_available():
            yield
            return
        matmul_backend = torch.backends.cuda.matmul
        attr_key = "fp32_precision" if hasattr(matmul_backend, "fp32_precision") else "allow_tf32"
        previous_setting = getattr(matmul_backend, attr_key)
        setattr(matmul_backend, attr_key, "tf32" if attr_key == "fp32_precision" else True)
        try:
            yield
        finally:
            setattr(matmul_backend, attr_key, previous_setting)

    def _parameter_mapping_specs(self) -> list[tuple[str, str, str]]:
        specs = [
            ("bn.weight", "bn_w", "cat"),
            ("bn.bias", "bn_b", "cat"),
            ("bn.running_mean", "bn_mean", "cat"),
            ("bn.running_var", "bn_var", "cat"),
            ("spatial.weight", "spatial_w", "cat"),
            ("fc.weight", "fc_w", "stack"),
            ("fc.bias", "fc_b", "stack"),
        ]
        if self.kind == "shallow":
            specs.extend(
                [
                    ("temporal.weight", "temporal_w", "cat"),
                    ("temporal.bias", "temporal_b", "cat"),
                ]
            )
        return specs

    def _synchronize_tensors(self, seed_networks: list[nn.Module], copy_to_seeds: bool) -> None:
        with torch.no_grad():
            for seed_idx, member_net in enumerate(seed_networks):
                member_state = member_net.state_dict()
                for member_key, grouped_attr, combine_mode in self._parameter_mapping_specs():
                    grouped_tensor = getattr(self, grouped_attr)
                    slice_width = len(grouped_tensor) // self.n
                    grouped_slice = (
                        grouped_tensor[seed_idx]
                        if combine_mode == "stack"
                        else grouped_tensor[seed_idx * slice_width : (seed_idx + 1) * slice_width]
                    )
                    if copy_to_seeds:
                        member_state[member_key].copy_(grouped_slice)
                    else:
                        grouped_slice.copy_(member_state[member_key])

    def write_back(self, seed_networks: list[nn.Module]) -> None:
        self._synchronize_tensors(seed_networks, copy_to_seeds=True)

    def read_from(self, seed_networks: list[nn.Module]) -> None:
        if len(seed_networks) != self.n:
            raise ValueError(f"Expected {self.n} seed networks, got {len(seed_networks)}")
        self.adapt = float(seed_networks[0].bn.adapt)
        self.version += 1
        self._synchronize_tensors(seed_networks, copy_to_seeds=False)

    def configure_tent_gradients(self, enable_tent: bool) -> None:
        self.requires_grad_(False)
        self.bn_w.requires_grad_(bool(enable_tent))
        self.bn_b.requires_grad_(bool(enable_tent))

    def filter_bank(self, window_batch: torch.Tensor) -> torch.Tensor:
        batch_len, chan_len, time_len = window_batch.shape
        pad_len = self.fir.shape[-1] // 2
        conv_out = F.conv1d(window_batch.reshape(batch_len * chan_len, 1, time_len), self.fir, padding=pad_len)
        return conv_out.reshape(batch_len, chan_len, -1, time_len).transpose(1, 2)

    def pre_bn(self, window_batch: torch.Tensor, fused: bool = False) -> torch.Tensor:
        batch_len, chan_len, time_len = window_batch.shape
        net_count = self.n
        if self.kind == "shallow" and fused:
            kern_len = self.temporal_w.shape[-1]
            filt_count = self.width
            spat_reshaped = self.spatial_w.reshape(net_count, filt_count, filt_count, chan_len)
            temp_reshaped = self.temporal_w.reshape(net_count, filt_count, kern_len)
            fused_weight = torch.bmm(
                spat_reshaped.permute(0, 1, 3, 2).reshape(net_count, filt_count * chan_len, filt_count),
                temp_reshaped,
            ).reshape(net_count * filt_count, 1, chan_len, kern_len)
            fused_bias = (spat_reshaped * self.temporal_b.reshape(net_count, 1, filt_count, 1)).sum((2, 3))
            return F.conv2d(window_batch[:, None], fused_weight, fused_bias.reshape(-1))
        if self.kind == "shallow":
            temp_act = F.conv2d(window_batch[:, None], self.temporal_w, self.temporal_b)
            return F.conv2d(temp_act, self.spatial_w, groups=net_count)
        fb_act = self.filter_bank(window_batch)
        return F.conv2d(fb_act.repeat(1, net_count, 1, 1), self.spatial_w, groups=net_count * self.fir.shape[0])

    def _segment_log_var(self, post_swish_tensor: torch.Tensor) -> torch.Tensor:
        seg_count = self.n_segments
        seg_len = post_swish_tensor.shape[-1] // seg_count
        segmented = post_swish_tensor[..., : seg_len * seg_count].reshape(
            *post_swish_tensor.shape[:-1], seg_count, seg_len
        )
        log_var = torch.log(torch.clamp(segmented.var(-1), min=1e-6))
        if self.modpool and self.mod_fir is not None:
            log_var = torch.cat(
                [log_var, FilterBankConvolutionalBackbone.modulation_log_power(post_swish_tensor, self.mod_fir)],
                dim=-1,
            )
        return log_var

    def band_features_from_xr(
        self,
        band_major_xr: torch.Tensor,
        batch_len: int,
        time_len: int,
        dropout: bool = False,
        tf32: bool = False,
    ) -> torch.Tensor:
        band_count, chan_len, _ = band_major_xr.shape
        net_count = self.n
        filt_per_band = self.width // band_count
        bn_w_bm = self.bn_w.reshape(net_count, band_count, filt_per_band).transpose(0, 1).reshape(-1)
        bn_b_bm = self.bn_b.reshape(net_count, band_count, filt_per_band).transpose(0, 1).reshape(-1)
        bn_mean_bm = self.bn_mean.reshape(net_count, band_count, filt_per_band).transpose(0, 1).reshape(-1)
        bn_var_bm = self.bn_var.reshape(net_count, band_count, filt_per_band).transpose(0, 1).reshape(-1)
        bn_scale = bn_w_bm / torch.sqrt(bn_var_bm + self.eps)
        bn_shift = bn_b_bm - bn_mean_bm * bn_scale
        folded_w = (
            self.spatial_w.reshape(net_count, band_count, filt_per_band, chan_len)
            .transpose(0, 1)
            .reshape(band_count, net_count * filt_per_band, chan_len)
            * bn_scale.reshape(band_count, net_count * filt_per_band, 1)
        )
        with self._cuda_tf32_matmul_context(bool(tf32 and band_major_xr.is_cuda)):
            proj_act = torch.bmm(folded_w, band_major_xr)
        swish_act = F.silu(proj_act + bn_shift.reshape(band_count, net_count * filt_per_band, 1))
        seg_lv = self._segment_log_var(swish_act.reshape(band_count * net_count * filt_per_band, batch_len, time_len))
        ordered_feat = seg_lv.reshape(band_count, net_count, filt_per_band, batch_len, -1).permute(3, 1, 0, 2, 4)
        flat_feat = ordered_feat.reshape(batch_len, net_count, -1)
        return F.dropout(flat_feat, self.p_drop, training=True) if dropout else flat_feat

    def band_features(self, filtered_bands: torch.Tensor, dropout: bool = False, tf32: bool = False) -> torch.Tensor:
        batch_len, band_count, chan_len, time_len = filtered_bands.shape
        band_major_xr = filtered_bands.permute(1, 2, 0, 3).reshape(band_count, chan_len, batch_len * time_len)
        return self.band_features_from_xr(band_major_xr, batch_len, time_len, dropout=dropout, tf32=tf32)

    def features(
        self,
        window_batch: torch.Tensor | None,
        adapt: float | None = None,
        dropout: bool = False,
        mask: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
        fused: bool = False,
    ) -> torch.Tensor:
        eff_adapt = self.adapt if adapt is None else float(adapt)
        pre_bn_act = self.pre_bn(window_batch, fused=fused) if h is None else h
        batch_len = len(pre_bn_act)
        bn_act = AdaptiveBatchNormConvolutionalBackbone.functional_adaptive_batch_norm(
            pre_bn_act,
            self.bn_mean,
            self.bn_var,
            self.bn_w,
            self.bn_b,
            self.eps,
            eff_adapt,
        )
        if self.kind == "shallow":
            pooled_act = torch.log(torch.clamp(F.avg_pool2d(bn_act * bn_act, *self.pool), min=1e-6))
        else:
            pooled_act = self._segment_log_var(F.silu(bn_act)[:, :, 0])
        flat_feat = pooled_act.reshape(batch_len, self.n, -1)
        if mask is not None:
            flat_feat = flat_feat * mask
        elif dropout:
            flat_feat = F.dropout(flat_feat, self.p_drop, training=True)
        return flat_feat

    def head(self, feat_tensor: torch.Tensor) -> torch.Tensor:
        weighted_sum = (feat_tensor.transpose(0, 1)[:, :, None, :] * self.fc_w[:, None]).sum(-1)
        return weighted_sum + self.fc_b[:, None]

    def forward(
        self,
        window_batch: torch.Tensor | None,
        adapt: float | None = None,
        dropout: bool = False,
        mask: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
        fused: bool = False,
    ) -> torch.Tensor:
        return self.head(self.features(window_batch, adapt=adapt, dropout=dropout, mask=mask, h=h, fused=fused))

    def predict_seed_probabilities(
        self,
        xw_batch: torch.Tensor,
        pre_bn_act: torch.Tensor | None = None,
        skip_update: bool = False,
        bn_frozen: bool = False,
        pseudo_chrono_adapter: Any = None,
        local_pc_bn_cache: dict[str, Any] | None = None,
        pc_bn_tau: float = 1.3,
        pc_after_freeze: bool = True,
    ) -> torch.Tensor:
        if pseudo_chrono_adapter is not None and pc_bn_tau > 0.0 and (not bn_frozen or pc_after_freeze):
            h0 = self.pre_bn(xw_batch) if pre_bn_act is None else pre_bn_act
            effective_skip = bool(skip_update) or bool(bn_frozen)
            if hasattr(pseudo_chrono_adapter, "prepare_forward") and hasattr(pseudo_chrono_adapter, "restore_cum"):
                pseudo_chrono_adapter.prepare_forward(self, h0, skip=effective_skip)
                try:
                    probs = torch.softmax(self(xw_batch, h=h0, adapt=0.0), dim=-1)
                    return probs.mean(dim=1)
                finally:
                    pseudo_chrono_adapter.restore_cum()
            if local_pc_bn_cache is not None:
                self._inject_local_pseudochrono_bn(
                    h0,
                    effective_skip,
                    pseudo_chrono_adapter,
                    local_pc_bn_cache,
                    pc_bn_tau,
                )
                try:
                    probs = torch.softmax(self(xw_batch, h=h0, adapt=0.0), dim=-1)
                    return probs.mean(dim=1)
                finally:
                    with torch.no_grad():
                        self.bn_mean.copy_(local_pc_bn_cache["mu_cum"])
                        self.bn_var.copy_(local_pc_bn_cache["v_cum"])

        probs = torch.softmax(self(xw_batch, h=pre_bn_act, adapt=0.0 if skip_update else None), dim=-1)
        return probs.mean(dim=1)

    def _inject_local_pseudochrono_bn(
        self,
        h_in: torch.Tensor,
        skip_flag: bool,
        pseudo_chrono_adapter: Any,
        bn_cache: dict[str, Any],
        bn_tau: float,
        bn_n0_prior: float = 12.0,
        bn_n0_glob: float = 3.0,
        max_hist: int = 512,
    ) -> None:
        if not bn_cache:
            bn_cache.update(
                {
                    "mu0": self.bn_mean.detach().clone(),
                    "v0": self.bn_var.detach().clone(),
                    "U_list": [],
                    "M_buf": torch.empty(
                        (max_hist, self.bn_mean.numel()),
                        dtype=self.bn_mean.dtype,
                        device=self.bn_mean.device,
                    ),
                    "S_buf": torch.empty(
                        (max_hist, self.bn_var.numel()),
                        dtype=self.bn_var.dtype,
                        device=self.bn_var.device,
                    ),
                    "n_buf": 0,
                    "m_cum": torch.zeros_like(self.bn_mean),
                    "s_cum": torch.zeros_like(self.bn_var),
                    "n_ok": 0,
                    "mu_cum": self.bn_mean.detach().clone(),
                    "v_cum": self.bn_var.detach().clone(),
                }
            )
        cur_u = getattr(pseudo_chrono_adapter, "cur_u", getattr(pseudo_chrono_adapter, "current_embedding", None))
        with torch.no_grad():
            if not skip_flag and cur_u is not None:
                m_t = h_in.mean((0, 2, 3))
                s_t = h_in.var((0, 2, 3), unbiased=False)
                bn_cache["m_cum"].add_(m_t)
                bn_cache["s_cum"].add_(s_t)
                bn_cache["n_ok"] += 1
                bn_cache["U_list"].append(np.asarray(cur_u, dtype=np.float64).copy())
                nb = int(bn_cache["n_buf"])
                if nb < max_hist:
                    bn_cache["M_buf"][nb].copy_(m_t.view(-1))
                    bn_cache["S_buf"][nb].copy_(s_t.view(-1))
                    bn_cache["n_buf"] = nb + 1
                else:
                    bn_cache["M_buf"][:-1].copy_(bn_cache["M_buf"][1:].clone())
                    bn_cache["S_buf"][:-1].copy_(bn_cache["S_buf"][1:].clone())
                    bn_cache["M_buf"][-1].copy_(m_t.view(-1))
                    bn_cache["S_buf"][-1].copy_(s_t.view(-1))
                if len(bn_cache["U_list"]) > max_hist:
                    bn_cache["U_list"].pop(0)

            n_tot_cum = bn_n0_prior + bn_n0_glob + float(bn_cache["n_ok"])
            mu_cum = ((bn_n0_prior + bn_n0_glob) * bn_cache["mu0"] + bn_cache["m_cum"]) / n_tot_cum
            v_cum = ((bn_n0_prior + bn_n0_glob) * bn_cache["v0"] + bn_cache["s_cum"]) / n_tot_cum
            bn_cache["mu_cum"] = mu_cum
            bn_cache["v_cum"] = v_cum

            n_m = int(bn_cache["n_buf"])
            if n_m > 0 and cur_u is not None and len(bn_cache["U_list"]) == n_m:
                u_mat = np.stack(bn_cache["U_list"], axis=0)
                sq_dist = np.sum((u_mat - np.asarray(cur_u, dtype=np.float64)[None, :]) ** 2, axis=1)
                w_slice = np.exp(-0.5 * sq_dist / (float(u_mat.shape[1]) * (bn_tau * bn_tau))).astype(np.float32)
                wt = torch.as_tensor(w_slice, dtype=torch.float32, device=self.bn_mean.device)
                w_sum = float(np.sum(w_slice, dtype=np.float32))
                m_ker = torch.mv(bn_cache["M_buf"][:n_m].t(), wt).view_as(self.bn_mean)
                s_ker = torch.mv(bn_cache["S_buf"][:n_m].t(), wt).view_as(self.bn_var)
                m_glob_avg = bn_cache["m_cum"] / float(bn_cache["n_ok"])
                s_glob_avg = bn_cache["s_cum"] / float(bn_cache["n_ok"])
                den_pc = bn_n0_prior + bn_n0_glob + w_sum
                mu_pc = (bn_n0_prior * bn_cache["mu0"] + bn_n0_glob * m_glob_avg + m_ker) / den_pc
                v_pc = (bn_n0_prior * bn_cache["v0"] + bn_n0_glob * s_glob_avg + s_ker) / den_pc
            else:
                mu_pc = mu_cum
                v_pc = v_cum
            self.bn_mean.copy_(mu_pc)
            self.bn_var.copy_(v_pc)

    def step_tent_entropy_minimization(
        self,
        pre_bn_buffer: torch.Tensor,
        tent_optimizer: torch.optim.Adam | None,
        tent_lr: float,
        diversity_weight: float = 1.0,
        min_buffer_windows: int = 2,
    ) -> torch.optim.Adam | None:
        buf_count = len(pre_bn_buffer)
        if min_buffer_windows > 1 and buf_count < min_buffer_windows:
            return tent_optimizer
        if tent_optimizer is None:
            tent_optimizer = torch.optim.Adam([self.bn_w, self.bn_b], lr=float(tent_lr))

        with torch.inference_mode(False):
            tent_optimizer.zero_grad(set_to_none=True)
            hb16 = self._tta_chunk_buf
            if (
                hb16 is None
                or hb16.shape[1:] != pre_bn_buffer.shape[1:]
                or hb16.device != pre_bn_buffer.device
                or hb16.dtype != pre_bn_buffer.dtype
            ):
                hb16 = torch.zeros(
                    (16, *pre_bn_buffer.shape[1:]),
                    dtype=pre_bn_buffer.dtype,
                    device=pre_bn_buffer.device,
                )
                self._tta_chunk_buf = hb16

            div_grad = None
            if diversity_weight > 0.0:
                with torch.no_grad():
                    pm_sum = None
                    for start_idx in range(0, buf_count, 16):
                        chunk_len = min(16, buf_count - start_idx)
                        if chunk_len == 16:
                            p_ng = torch.softmax(
                                self(None, adapt=0.0, h=pre_bn_buffer[start_idx : start_idx + 16]),
                                dim=-1,
                            ).sum(dim=1)
                        else:
                            hb16.zero_()
                            hb16[:chunk_len].copy_(pre_bn_buffer[start_idx : start_idx + chunk_len])
                            p_ng = torch.softmax(self(None, adapt=0.0, h=hb16)[:, :chunk_len], dim=-1).sum(dim=1)
                        pm_sum = p_ng if pm_sum is None else (pm_sum + p_ng)
                    p_mean = pm_sum / float(buf_count)
                    div_grad = (float(diversity_weight) / float(buf_count)) * (
                        torch.log(p_mean + 1e-8) + p_mean / (p_mean + 1e-8)
                    ).unsqueeze(1)

            with torch.enable_grad():
                for start_idx in range(0, buf_count, 16):
                    chunk_len = min(16, buf_count - start_idx)
                    if chunk_len == 16:
                        h_chunk = pre_bn_buffer[start_idx : start_idx + 16].clone()
                    else:
                        hb16.zero_()
                        hb16[:chunk_len].copy_(pre_bn_buffer[start_idx : start_idx + chunk_len])
                        h_chunk = hb16.clone()
                    p_chunk = torch.softmax(self(None, adapt=0.0, h=h_chunk)[:, :chunk_len], dim=-1)
                    loss_chunk = -(p_chunk * torch.log(p_chunk + 1e-8)).sum(dim=-1).sum(dim=-1) / float(buf_count)
                    if div_grad is not None:
                        loss_chunk = loss_chunk + (p_chunk * div_grad).sum(dim=-1).sum(dim=-1)
                    loss_chunk.sum().backward()
            tent_optimizer.step()
        return tent_optimizer

    @staticmethod
    def compute_step_budget(epochs: int, total_rows: int, batch_size: int, step_scale: float) -> int | None:
        if step_scale >= 1.0 or epochs <= 0 or total_rows <= 0:
            return None
        full_steps = int(epochs) * int(np.ceil(total_rows / float(batch_size)))
        return max(1, int(np.floor(step_scale * full_steps + 0.5)))

    @staticmethod
    def _ag3_im_entropy_loss(seed_logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(seed_logits, dim=-1)
        cond_ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean(dim=1)
        mean_probs = probs.mean(dim=1)
        marg_ent = -(mean_probs * torch.log(mean_probs.clamp_min(1e-12))).sum(dim=-1)
        return (cond_ent - marg_ent).mean()

    @classmethod
    def fit_ag3_spatial_adapter(
        cls,
        shallow_group: GroupedEnsembleConvolutionalBackbone,
        fbcnet_group: GroupedEnsembleConvolutionalBackbone,
        fbmod_group: GroupedEnsembleConvolutionalBackbone | None,
        whitened_targets: np.ndarray | torch.Tensor,
        pseudo_labels: np.ndarray,
        sample_weights: np.ndarray,
        initial_adapter: np.ndarray,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        im_weight: float,
        label_smoothing: float,
        batch_size: int,
        random_seed: int,
        target_device: torch.device,
        weight_shallow: float = 0.55,
        weight_fbcnet: float = 0.40,
        weight_fbmod: float = 0.0,
        row_l2_normalize: bool = True,
        step_scale: float = 1.0,
    ) -> np.ndarray:
        chan_len = whitened_targets.shape[1]
        if epochs <= 0 or shallow_group is None or fbcnet_group is None:
            return initial_adapter.copy()
        shallow_group.requires_grad_(False)
        fbcnet_group.requires_grad_(False)
        if fbmod_group is not None:
            fbmod_group.requires_grad_(False)

        cuda_devices = [target_device] if target_device.type == "cuda" else []
        cpu_gen = torch.Generator().manual_seed(int(random_seed))
        with torch.inference_mode(False), torch.enable_grad(), torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(random_seed))
            xt_all = (
                whitened_targets.to(dtype=torch.float32, device=target_device)
                if torch.is_tensor(whitened_targets)
                else torch.as_tensor(whitened_targets, dtype=torch.float32, device=target_device)
            )
            yt_all = torch.as_tensor(pseudo_labels, dtype=torch.long, device=target_device)
            wt_all = torch.as_tensor(sample_weights, dtype=torch.float32, device=target_device)
            eye_mat = torch.eye(chan_len, dtype=torch.float32, device=target_device)

            total_rows, _, time_len = xt_all.shape
            with torch.no_grad():
                xf2_full = torch.cat(
                    [fbcnet_group.filter_bank(xt_all[idx0 : idx0 + 32]) for idx0 in range(0, total_rows, 32)],
                    dim=0,
                )
                nb2 = xf2_full.shape[1]
                xf2_nb = xf2_full.permute(1, 2, 0, 3).contiguous()
                del xf2_full
                xf3_full = (
                    torch.cat(
                        [fbmod_group.filter_bank(xt_all[idx0 : idx0 + 32]) for idx0 in range(0, total_rows, 32)],
                        dim=0,
                    )
                    if (fbmod_group is not None and weight_fbmod > 0.0)
                    else None
                )

            delta_init = (initial_adapter - np.eye(chan_len, dtype=np.float64)).astype(np.float32)
            delta_param = nn.Parameter(torch.as_tensor(delta_init, device=target_device))
            adapter_opt = torch.optim.Adam([delta_param], lr=float(learning_rate))
            weight_denom = weight_shallow + weight_fbcnet + (weight_fbmod if xf3_full is not None else 0.0)

            perms_cpu = [torch.randperm(total_rows, generator=cpu_gen) for _ in range(epochs)]
            all_perms_dev = torch.cat(perms_cpu, dim=0).to(target_device)
            step_cap = cls.compute_step_budget(epochs, total_rows, batch_size, float(step_scale))
            steps_taken = 0

            for ep_idx in range(epochs):
                if step_cap is not None and steps_taken >= step_cap:
                    break
                ep_offset = ep_idx * total_rows
                for s0 in range(0, total_rows, batch_size):
                    if step_cap is not None and steps_taken >= step_cap:
                        break
                    steps_taken += 1
                    s1 = min(total_rows, s0 + batch_size)
                    cur_bs = s1 - s0
                    batch_idx = all_perms_dev[ep_offset + s0 : ep_offset + s1]
                    adapter_mat = eye_mat + delta_param
                    if row_l2_normalize:
                        adapter_mat = adapter_mat / adapter_mat.norm(dim=1, keepdim=True).clamp_min(1e-8)

                    wb = wt_all.index_select(0, batch_idx)
                    xt_b = xt_all.index_select(0, batch_idx)
                    yt_b = yt_all.index_select(0, batch_idx)

                    xb_sh = torch.matmul(adapter_mat, xt_b)
                    logits_sh = shallow_group(xb_sh, adapt=0.0, dropout=True, fused=True)
                    ce_sh = F.cross_entropy(
                        logits_sh.reshape(-1, logits_sh.shape[-1]),
                        yt_b.repeat(shallow_group.n),
                        reduction="none",
                        label_smoothing=float(label_smoothing),
                    )
                    loss_sh = (ce_sh.view(shallow_group.n, -1) * wb).mean()

                    xr2_raw = xf2_nb.index_select(2, batch_idx)
                    xr2_adapted = torch.matmul(adapter_mat, xr2_raw.reshape(nb2, chan_len, cur_bs * time_len))
                    feat_fb2 = fbcnet_group.band_features_from_xr(xr2_adapted, cur_bs, time_len, dropout=True, tf32=True)
                    logits_fb2 = fbcnet_group.head(feat_fb2)
                    ce_fb2 = F.cross_entropy(
                        logits_fb2.reshape(-1, logits_fb2.shape[-1]),
                        yt_b.repeat(fbcnet_group.n),
                        reduction="none",
                        label_smoothing=float(label_smoothing),
                    )
                    loss_fb2 = (ce_fb2.view(fbcnet_group.n, -1) * wb).mean()

                    if im_weight > 0.0 and cur_bs >= 4:
                        loss_sh = loss_sh + im_weight * cls._ag3_im_entropy_loss(logits_sh)
                        loss_fb2 = loss_fb2 + im_weight * cls._ag3_im_entropy_loss(logits_fb2)
                    total_ce = weight_shallow * loss_sh + weight_fbcnet * loss_fb2

                    if fbmod_group is not None and xf3_full is not None:
                        xf3_b = torch.matmul(adapter_mat, xf3_full.index_select(0, batch_idx))
                        logits_fb3 = fbmod_group.head(fbmod_group.band_features(xf3_b, dropout=True, tf32=True))
                        ce_fb3 = F.cross_entropy(
                            logits_fb3.reshape(-1, logits_fb3.shape[-1]),
                            yt_b.repeat(fbmod_group.n),
                            reduction="none",
                            label_smoothing=float(label_smoothing),
                        )
                        loss_fb3 = (ce_fb3.view(fbmod_group.n, -1) * wb).mean()
                        if im_weight > 0.0 and cur_bs >= 4:
                            loss_fb3 = loss_fb3 + im_weight * cls._ag3_im_entropy_loss(logits_fb3)
                        total_ce = total_ce + weight_fbmod * loss_fb3

                    total_loss = (total_ce / weight_denom) + float(weight_decay) * (delta_param * delta_param).sum()
                    adapter_opt.zero_grad(set_to_none=True)
                    total_loss.backward()
                    adapter_opt.step()

            with torch.no_grad():
                final_mat = eye_mat + delta_param
                if row_l2_normalize:
                    final_mat = final_mat / final_mat.norm(dim=1, keepdim=True).clamp_min(1e-8)
                final_np = final_mat.detach().cpu().numpy().astype(np.float64)
        return final_np

    @staticmethod
    def _im_term_all_session(logits_tensor: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits_tensor, dim=-1)
        cond_ent = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean(dim=-1)
        mean_probs = probs.mean(dim=-2)
        marg_ent = -(mean_probs * mean_probs.clamp_min(1e-8).log()).sum(dim=-1)
        return (cond_ent - marg_ent).sum()

    @staticmethod
    def _im_term_masked_session(logits_tensor: torch.Tensor, sess_mask: torch.Tensor) -> torch.Tensor:
        sess_count = int(sess_mask.sum().item())
        if sess_count < 2:
            return logits_tensor.sum() * 0.0
        sess_weights = sess_mask.to(dtype=logits_tensor.dtype) / float(sess_count)
        probs = F.softmax(logits_tensor, dim=-1)
        row_ent = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        cond_ent = (row_ent * sess_weights).sum(dim=-1)
        mean_probs = (probs * sess_weights.unsqueeze(-1)).sum(dim=-2)
        marg_ent = -(mean_probs * mean_probs.clamp_min(1e-8).log()).sum(dim=-1)
        return (cond_ent - marg_ent).sum()

    def run_grouped_replay_self_training(
        self,
        combined_windows: np.ndarray | torch.Tensor,
        combined_labels: np.ndarray,
        combined_weights: np.ndarray,
        session_mask: np.ndarray | None,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        label_smoothing: float,
        train_params_mode: str,
        im_weight: float,
        random_seed: int,
        step_scale: float = 1.0,
        allow_tf32: bool = True,
        enable_tent_after: bool = False,
    ) -> None:
        if epochs <= 0:
            return
        target_device = self.fc_w.device
        cuda_devices = [target_device] if target_device.type == "cuda" else []
        self.version += 1
        try:
            self.requires_grad_(False)
            trainable_params: list[nn.Parameter] = []
            for param_key, param_tensor in self.named_parameters():
                should_train = (
                    param_key.startswith("fc")
                    if train_params_mode == "head"
                    else (not param_key.startswith("temporal") if train_params_mode == "spatial_head" else True)
                )
                if should_train:
                    param_tensor.requires_grad_(True)
                    trainable_params.append(param_tensor)

            cpu_gen = torch.Generator().manual_seed(int(random_seed))
            with torch.inference_mode(False), torch.enable_grad(), torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(int(random_seed))
                xt_all = (
                    combined_windows.to(dtype=torch.float32, device=target_device)
                    if torch.is_tensor(combined_windows)
                    else torch.as_tensor(combined_windows, dtype=torch.float32, device=target_device)
                )
                yt_all = torch.as_tensor(combined_labels, dtype=torch.long, device=target_device)
                wt_all = torch.as_tensor(combined_weights, dtype=torch.float32, device=target_device)

                total_rows = len(xt_all)
                time_len = xt_all.shape[-1]
                chan_len = xt_all.shape[1]
                sess_count = int(session_mask.sum()) if session_mask is not None else total_rows
                all_rows_are_session = (session_mask is None) or (sess_count == total_rows)

                xf_nb = None
                nb_count = 0
                if self.kind == "fbcnet":
                    with torch.no_grad():
                        xf_full = torch.cat(
                            [self.filter_bank(xt_all[idx0 : idx0 + 32]) for idx0 in range(0, total_rows, 32)],
                            dim=0,
                        )
                        nb_count = xf_full.shape[1]
                        xf_nb = xf_full.permute(1, 2, 0, 3).contiguous()
                        del xf_full

                st_opt = torch.optim.Adam([{"params": trainable_params, "lr": float(learning_rate)}])
                perms_cpu = [torch.randperm(total_rows, generator=cpu_gen) for _ in range(epochs)]
                all_perms_dev = torch.cat(perms_cpu, dim=0).to(target_device)
                step_cap = self.compute_step_budget(epochs, total_rows, batch_size, float(step_scale))
                steps_taken = 0

                for ep_idx, perm_cpu in enumerate(perms_cpu):
                    if step_cap is not None and steps_taken >= step_cap:
                        break
                    ep_offset = ep_idx * total_rows
                    for s0 in range(0, total_rows, batch_size):
                        if step_cap is not None and steps_taken >= step_cap:
                            break
                        steps_taken += 1
                        s1 = min(total_rows, s0 + batch_size)
                        cur_bs = s1 - s0
                        batch_idx = all_perms_dev[ep_offset + s0 : ep_offset + s1]
                        batch_sess_count = cur_bs if all_rows_are_session else int((perm_cpu[s0:s1] < sess_count).sum().item())

                        if xf_nb is None:
                            logits = self(xt_all.index_select(0, batch_idx), adapt=0.0, dropout=True, fused=True)
                        else:
                            xr_sub = xf_nb.index_select(2, batch_idx).view(nb_count, chan_len, cur_bs * time_len)
                            logits = self.head(
                                self.band_features_from_xr(
                                    xr_sub, cur_bs, time_len, dropout=True, tf32=bool(allow_tf32)
                                )
                            )
                        yb = yt_all.index_select(0, batch_idx).repeat(self.n)
                        wb = wt_all.index_select(0, batch_idx)
                        ce_rows = F.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]),
                            yb,
                            reduction="none",
                            label_smoothing=float(label_smoothing),
                        )
                        batch_loss = (ce_rows.view(self.n, -1) * wb).mean(dim=1).sum()
                        if im_weight > 0.0 and batch_sess_count >= 2:
                            if batch_sess_count == cur_bs:
                                batch_loss = batch_loss + float(im_weight) * self._im_term_all_session(logits)
                            else:
                                batch_loss = batch_loss + float(im_weight) * self._im_term_masked_session(
                                    logits, batch_idx < sess_count
                                )
                        st_opt.zero_grad(set_to_none=True)
                        batch_loss.backward()
                        st_opt.step()
        finally:
            self.configure_tent_gradients(enable_tent=enable_tent_after)
