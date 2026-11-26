"""Multi-organ causal ensemble BCI decoders."""

from neurips_bci.decoder.ensemble_decoders.EnsembleDecoder import EnsembleDecoder
from neurips_bci.decoder.ensemble_decoders.H10MultimodalEnsembleDecoder import (
    H10MultimodalEnsembleDecoder,
)

__all__ = [
    "EnsembleDecoder",
    "H10MultimodalEnsembleDecoder",
]
