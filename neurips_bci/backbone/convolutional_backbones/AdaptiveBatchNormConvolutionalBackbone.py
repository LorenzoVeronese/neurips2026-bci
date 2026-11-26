"""Stream-adaptive 2D Batch Normalization layer supporting per-session grouped training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from neurips_bci.backbone.convolutional_backbones.ConvolutionalBackbone import ConvolutionalBackbone
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class AdaptiveBatchNormConvolutionalBackbone(ConvolutionalBackbone):
    """BatchNorm2d-compatible layer whose running statistics adapt causally in eval mode."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        adapt: float = 0.0,
        groups: int = 1,
    ) -> None:
        super().__init__(backbone_architecture="adaptive_batch_norm_2d")
        validated_features = ParameterTypeValidator.require_integer(num_features, "num_features", minimum_bound=1)
        self.num_features = validated_features
        self.eps = ParameterTypeValidator.require_float(eps, "eps", minimum_bound=0.0, strict_lower=True)
        self.momentum = None if momentum is None else float(momentum)
        self.adapt = ParameterTypeValidator.require_float(adapt, "adapt", minimum_bound=0.0, maximum_bound=1.0)
        self.groups = ParameterTypeValidator.require_integer(groups, "groups", minimum_bound=1)

        self.weight = nn.Parameter(torch.ones(validated_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(validated_features, dtype=torch.float32))
        self.register_buffer("running_mean", torch.zeros(validated_features, dtype=torch.float32))
        self.register_buffer("running_var", torch.ones(validated_features, dtype=torch.float32))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def reset_running_stats(self) -> None:
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.num_batches_tracked.zero_()

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    @staticmethod
    def functional_adaptive_batch_norm(
        activation_tensor: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        affine_weight: torch.Tensor,
        affine_bias: torch.Tensor,
        eps_constant: float,
        adapt_momentum: float,
    ) -> torch.Tensor:
        if adapt_momentum:
            with torch.no_grad():
                running_mean.mul_(1.0 - adapt_momentum).add_(adapt_momentum * activation_tensor.mean((0, 2, 3)))
                running_var.mul_(1.0 - adapt_momentum).add_(
                    adapt_momentum * activation_tensor.var((0, 2, 3), unbiased=False)
                )
        return F.batch_norm(
            activation_tensor,
            running_mean,
            running_var,
            affine_weight,
            affine_bias,
            False,
            0.0,
            eps_constant,
        )

    def _forward_grouped_sessions(self, activation_tensor: torch.Tensor) -> torch.Tensor:
        grouped_tensor = activation_tensor.reshape(self.groups, -1, *activation_tensor.shape[1:])
        slice_mean = grouped_tensor.mean((1, 3, 4), keepdim=True)
        slice_var = grouped_tensor.var((1, 3, 4), unbiased=False, keepdim=True)
        with torch.no_grad():
            self.num_batches_tracked.add_(1)
            eff_momentum = (
                1.0 / float(self.num_batches_tracked.item())
                if self.momentum is None
                else float(self.momentum)
            )
            if eff_momentum > 0.0:
                self.running_mean.mul_(1.0 - eff_momentum).add_(eff_momentum * slice_mean.mean(0).flatten())
                self.running_var.mul_(1.0 - eff_momentum).add_(eff_momentum * slice_var.mean(0).flatten())
        normalized_tensor = ((grouped_tensor - slice_mean) / torch.sqrt(slice_var + self.eps)).reshape(
            activation_tensor.shape
        )
        return normalized_tensor * self.weight[:, None, None] + self.bias[:, None, None]

    def forward(self, activation_tensor: torch.Tensor) -> torch.Tensor:
        if self.training and self.groups > 1:
            return self._forward_grouped_sessions(activation_tensor)
        if self.training:
            with torch.no_grad():
                self.num_batches_tracked.add_(1)
            exp_avg_factor = (
                1.0 / float(self.num_batches_tracked.item())
                if self.momentum is None
                else float(self.momentum)
            )
            return F.batch_norm(
                activation_tensor,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                True,
                exp_avg_factor,
                self.eps,
            )
        if not self.adapt:
            return F.batch_norm(
                activation_tensor,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                0.0,
                self.eps,
            )
        return self.functional_adaptive_batch_norm(
            activation_tensor,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.eps,
            self.adapt,
        )
