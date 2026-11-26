"""Abstract base class for convolutional neural network backbones in Layer 6."""

from __future__ import annotations

from abc import ABC
import torch
from torch import nn

from neurips_bci.backbone.Backbone import Backbone


class ConvolutionalBackbone(Backbone, ABC):
    """Shared contract and BatchNorm state utilities for EEG convolutional backbones."""

    def __init__(self, backbone_architecture: str) -> None:
        super().__init__(backbone_architecture=backbone_architecture)

    def parameter_count(self) -> int:
        return int(sum(param_tensor.numel() for param_tensor in self.parameters()))

    def set_adaptive_bn_momentum(self, adaptation_momentum: float) -> None:
        momentum_float = float(adaptation_momentum)
        for submodule in self.modules():
            if hasattr(submodule, "adapt") and hasattr(submodule, "running_mean"):
                submodule.adapt = momentum_float

    def extract_bn_state_dictionary(self) -> dict[str, torch.Tensor]:
        bn_keys: set[str] = set()
        for mod_prefix, submodule in self.named_modules():
            if hasattr(submodule, "running_mean") and hasattr(submodule, "running_var"):
                for suffix_key in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                    full_key = f"{mod_prefix}.{suffix_key}" if mod_prefix else suffix_key
                    bn_keys.add(full_key)
        current_state = self.state_dict()
        return {
            state_key: state_tensor.detach().clone()
            for state_key, state_tensor in current_state.items()
            if state_key in bn_keys
        }

    def recompute_running_bn_statistics(self, whitened_windows: torch.Tensor, mini_batch_size: int = 64) -> None:
        bn_modules = [
            submodule
            for submodule in self.modules()
            if hasattr(submodule, "running_mean")
            and hasattr(submodule, "running_var")
            and hasattr(submodule, "reset_running_stats")
        ]
        saved_momenta = [submodule.momentum for submodule in bn_modules]
        saved_groups = [getattr(submodule, "groups", 1) for submodule in bn_modules]
        for submodule in bn_modules:
            submodule.reset_running_stats()
            submodule.momentum = None
            if hasattr(submodule, "groups"):
                submodule.groups = 1
        was_training = self.training
        self.train()
        try:
            with torch.no_grad():
                for start_idx in range(0, len(whitened_windows), mini_batch_size):
                    self(whitened_windows[start_idx : start_idx + mini_batch_size])
        finally:
            for submodule, mom_val, grp_val in zip(bn_modules, saved_momenta, saved_groups):
                submodule.momentum = mom_val
                if hasattr(submodule, "groups"):
                    submodule.groups = grp_val
            self.train(was_training)
