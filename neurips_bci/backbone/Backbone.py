"""Abstract root class for PyTorch neural network Backbones."""

from __future__ import annotations

from abc import ABC, abstractmethod
from torch import nn


class Backbone(nn.Module, ABC):
    """Base contract for convolutional, grouped-BMM, and standalone foundation PyTorch backbones."""

    def __init__(self, backbone_architecture: str) -> None:
        super().__init__()
        if not isinstance(backbone_architecture, str) or not backbone_architecture.strip():
            raise TypeError("backbone_architecture must be a non-empty str")
        self._backbone_architecture = backbone_architecture

    @property
    def backbone_architecture(self) -> str:
        return self._backbone_architecture

    @abstractmethod
    def parameter_count(self) -> int:
        """Return the number of trainable or active tensor elements in the backbone."""
