"""Abstract base class for pure-PyTorch EEG foundation model backbones."""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch

from neurips_bci.backbone.Backbone import Backbone
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class FoundationBackbone(Backbone, ABC):
    """Base contract for standalone and batched EEG foundation encoders (EEGPT, CodeBrain, CBraMod, MIRepNet)."""

    def __init__(self, backbone_architecture: str, embedding_dimension: int) -> None:
        super().__init__(backbone_architecture=backbone_architecture)
        self._embedding_dimension = ParameterTypeValidator.require_integer(
            embedding_dimension, "embedding_dimension", minimum_bound=1
        )

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    def parameter_count(self) -> int:
        trainable_or_frozen_params = sum(int(param_tensor.numel()) for param_tensor in self.parameters())
        if trainable_or_frozen_params > 0:
            return trainable_or_frozen_params
        return sum(int(buf_tensor.numel()) for buf_tensor in self.buffers() if buf_tensor.is_floating_point())

    @abstractmethod
    def encode_window_features(
        self, *tensor_inputs: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Encode resampled single-window or batched EEG tensors into foundation representations."""
