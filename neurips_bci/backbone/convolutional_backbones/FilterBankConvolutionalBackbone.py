"""Filter-bank spatial-projection convolutional backbone (9-band FBCNet and 10-band FBModNet)."""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin
import torch
import torch.nn.functional as F
from torch import nn

from neurips_bci.backbone.convolutional_backbones.AdaptiveBatchNormConvolutionalBackbone import (
    AdaptiveBatchNormConvolutionalBackbone,
)
from neurips_bci.backbone.convolutional_backbones.ConvolutionalBackbone import ConvolutionalBackbone
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class FilterBankConvolutionalBackbone(ConvolutionalBackbone):
    """Fixed FIR filter-bank spatial Conv2d network supporting sub-4Hz band and modulation pooling."""

    DEFAULT_NINE_BANDS_HZ: tuple[tuple[float, float], ...] = tuple(
        (float(4 * idx), float(4 * idx + 4)) for idx in range(1, 10)
    )

    def __init__(
        self,
        n_chans: int = 41,
        n_times: int = 480,
        n_classes: int = 3,
        bands: list[list[float]] | tuple[tuple[float, float], ...] | None = None,
        m: int = 32,
        taps: int = 121,
        n_segments: int = 4,
        dropout: float = 0.5,
        sampling_rate_hz: float = 120.0,
        sub4: bool = False,
        modpool: bool = False,
    ) -> None:
        arch_tag = "fbmodnet_10band" if (sub4 and modpool) else "fbcnet_9band"
        super().__init__(backbone_architecture=arch_tag)
        validated_chans = ParameterTypeValidator.require_integer(n_chans, "n_chans", minimum_bound=1)
        validated_times = ParameterTypeValidator.require_integer(n_times, "n_times", minimum_bound=16)
        validated_classes = ParameterTypeValidator.require_integer(n_classes, "n_classes", minimum_bound=2)
        validated_m = ParameterTypeValidator.require_integer(m, "m", minimum_bound=1)
        validated_taps = ParameterTypeValidator.require_integer(taps, "taps", minimum_bound=3)
        validated_segments = ParameterTypeValidator.require_integer(n_segments, "n_segments", minimum_bound=1)
        validated_dropout = ParameterTypeValidator.require_float(dropout, "dropout", 0.0, 0.95)
        validated_sfreq = ParameterTypeValidator.require_float(sampling_rate_hz, "sampling_rate_hz", 1.0)

        resolved_bands = (
            self.DEFAULT_NINE_BANDS_HZ
            if bands is None
            else tuple((float(band_pair[0]), float(band_pair[1])) for band_pair in bands)
        )
        self.band_input = False
        self.sub4 = bool(sub4)
        self.modpool = bool(modpool)
        self._n_chans = validated_chans
        self._n_times = validated_times
        self._n_classes = validated_classes

        fir_kernels = [firwin(validated_taps, 4.0, pass_zero=True, fs=validated_sfreq)] if self.sub4 else []
        fir_kernels.extend(
            firwin(validated_taps, list(band_pair), pass_zero=False, fs=validated_sfreq)
            for band_pair in resolved_bands
        )
        stacked_fir = np.stack(fir_kernels, axis=0)
        self.register_buffer("fir", torch.as_tensor(stacked_fir, dtype=torch.float32)[:, None])

        if self.modpool:
            self.register_buffer("mod_fir", torch.as_tensor(self.build_modulation_fir_bank(), dtype=torch.float32))
        else:
            self.mod_fir = None

        band_count = len(fir_kernels)
        self.spatial = nn.Conv2d(
            band_count,
            band_count * validated_m,
            (validated_chans, 1),
            groups=band_count,
            bias=False,
        )
        self.bn = AdaptiveBatchNormConvolutionalBackbone(band_count * validated_m)
        self.n_segments = validated_segments
        self.drop = nn.Dropout(validated_dropout)

        with torch.no_grad():
            dummy_features = self.features(torch.zeros(2, 1, validated_chans, validated_times))
            feature_dim = int(dummy_features.shape[1])
        self.bn.reset_running_stats()
        self.fc = nn.Linear(feature_dim, validated_classes)

    @staticmethod
    def build_modulation_fir_bank() -> np.ndarray:
        slow_taps = firwin(29, [0.5, 3.0], pass_zero=False, fs=30.0)
        fast_taps = firwin(29, [3.0, 7.5], pass_zero=False, fs=30.0)
        return np.stack([slow_taps, fast_taps], axis=0).astype(np.float32)[:, None, :]

    @staticmethod
    def modulation_log_power(post_swish_tensor: torch.Tensor, mod_fir_tensor: torch.Tensor) -> torch.Tensor:
        leading_dims = post_swish_tensor.shape[:-1]
        flattened_signal = post_swish_tensor.reshape(-1, 1, post_swish_tensor.shape[-1])
        envelope = torch.sqrt(F.avg_pool1d(flattened_signal * flattened_signal, kernel_size=8, stride=4) + 1e-6)
        envelope = envelope - envelope.mean(dim=-1, keepdim=True)
        mod_filtered = F.conv1d(envelope, mod_fir_tensor, padding=mod_fir_tensor.shape[-1] // 2)
        log_mod_power = torch.log(torch.clamp((mod_filtered * mod_filtered).mean(dim=-1), min=1e-6))
        return log_mod_power.reshape(*leading_dims, 2)

    def features(self, input_4d: torch.Tensor) -> torch.Tensor:
        batch_len, _, chan_len, time_len = input_4d.shape
        pad_len = self.fir.shape[-1] // 2
        filtered_1d = F.conv1d(input_4d.reshape(batch_len * chan_len, 1, time_len), self.fir, padding=pad_len)
        band_tensor = filtered_1d.reshape(batch_len, chan_len, -1, time_len).transpose(1, 2)
        spatial_act = self.bn(self.spatial(band_tensor))
        swish_act = spatial_act * torch.sigmoid(spatial_act)
        seg_len = time_len // self.n_segments
        segmented = swish_act[..., : seg_len * self.n_segments].reshape(
            batch_len, swish_act.shape[1], self.n_segments, seg_len
        )
        log_variance = torch.log(torch.clamp(segmented.var(-1), min=1e-6))
        if self.modpool and self.mod_fir is not None:
            log_variance = torch.cat(
                [log_variance, self.modulation_log_power(swish_act[:, :, 0], self.mod_fir)],
                dim=-1,
            )
        return self.drop(log_variance.flatten(1))

    def forward(self, input_3d: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(input_3d[:, None]))
