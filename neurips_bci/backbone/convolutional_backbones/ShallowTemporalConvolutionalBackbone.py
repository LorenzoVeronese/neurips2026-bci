"""Shallow temporal-spatial log-bandpower convolutional backbone (k=25 and k=49)."""

from __future__ import annotations

import torch
from torch import nn

from neurips_bci.backbone.convolutional_backbones.AdaptiveBatchNormConvolutionalBackbone import (
    AdaptiveBatchNormConvolutionalBackbone,
)
from neurips_bci.backbone.convolutional_backbones.ConvolutionalBackbone import ConvolutionalBackbone
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class ShallowTemporalConvolutionalBackbone(ConvolutionalBackbone):
    """Shallow temporal and spatial Conv2d network with log-variance pooling and strict r45 state_dict keys."""

    def __init__(
        self,
        n_chans: int = 41,
        n_times: int = 480,
        n_classes: int = 3,
        n_filters: int = 40,
        kernel: int = 25,
        pool: int = 35,
        stride: int = 7,
        dropout: float = 0.5,
    ) -> None:
        super().__init__(backbone_architecture=f"shallow_temporal_k{int(kernel)}")
        validated_chans = ParameterTypeValidator.require_integer(n_chans, "n_chans", minimum_bound=1)
        validated_times = ParameterTypeValidator.require_integer(n_times, "n_times", minimum_bound=16)
        validated_classes = ParameterTypeValidator.require_integer(n_classes, "n_classes", minimum_bound=2)
        validated_filters = ParameterTypeValidator.require_integer(n_filters, "n_filters", minimum_bound=1)
        validated_kernel = ParameterTypeValidator.require_integer(kernel, "kernel", minimum_bound=1)
        validated_pool = ParameterTypeValidator.require_integer(pool, "pool", minimum_bound=1)
        validated_stride = ParameterTypeValidator.require_integer(stride, "stride", minimum_bound=1)
        validated_dropout = ParameterTypeValidator.require_float(dropout, "dropout", 0.0, 0.95)

        self._n_chans = validated_chans
        self._n_times = validated_times
        self._n_classes = validated_classes
        self._kernel = validated_kernel

        self.temporal = nn.Conv2d(1, validated_filters, (1, validated_kernel))
        self.spatial = nn.Conv2d(validated_filters, validated_filters, (validated_chans, 1), bias=False)
        self.bn = AdaptiveBatchNormConvolutionalBackbone(validated_filters)
        self.pool = nn.AvgPool2d((1, validated_pool), (1, validated_stride))
        self.drop = nn.Dropout(validated_dropout)

        with torch.no_grad():
            dummy_features = self.features(torch.zeros(2, 1, validated_chans, validated_times))
            feature_dim = int(dummy_features.shape[1])
        self.bn.reset_running_stats()
        self.fc = nn.Linear(feature_dim, validated_classes)

    @property
    def n_chans(self) -> int:
        return self._n_chans

    @property
    def n_times(self) -> int:
        return self._n_times

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def kernel(self) -> int:
        return self._kernel

    def pooled(self, input_4d: torch.Tensor) -> torch.Tensor:
        conv_act = self.bn(self.spatial(self.temporal(input_4d)))
        return torch.log(torch.clamp(self.pool(conv_act * conv_act), min=1e-6))

    def features(self, input_4d: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pooled(input_4d)).flatten(1)

    def forward(self, input_3d: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(input_3d[:, None]))
