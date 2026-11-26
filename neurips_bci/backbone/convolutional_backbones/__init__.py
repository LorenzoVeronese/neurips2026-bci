"""Convolutional neural network backbones for H10 EEG decoding."""

from neurips_bci.backbone.convolutional_backbones.AdaptiveBatchNormConvolutionalBackbone import (
    AdaptiveBatchNormConvolutionalBackbone,
)
from neurips_bci.backbone.convolutional_backbones.ConvolutionalBackbone import ConvolutionalBackbone
from neurips_bci.backbone.convolutional_backbones.FilterBankConvolutionalBackbone import (
    FilterBankConvolutionalBackbone,
)
from neurips_bci.backbone.convolutional_backbones.GroupedEnsembleConvolutionalBackbone import (
    GroupedEnsembleConvolutionalBackbone,
)
from neurips_bci.backbone.convolutional_backbones.ShallowTemporalConvolutionalBackbone import (
    ShallowTemporalConvolutionalBackbone,
)

__all__ = [
    "AdaptiveBatchNormConvolutionalBackbone",
    "ConvolutionalBackbone",
    "FilterBankConvolutionalBackbone",
    "GroupedEnsembleConvolutionalBackbone",
    "ShallowTemporalConvolutionalBackbone",
]
