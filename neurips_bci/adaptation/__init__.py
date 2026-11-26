"""Exports for the Adapter family."""

from neurips_bci.adaptation.Adapter import Adapter
from neurips_bci.adaptation.gate_adapters import (
    AsymmetricConfidenceGateAdapter,
    ExcessAgreementGateAdapter,
    GateAdapter,
    SessionPaceGateAdapter,
)
from neurips_bci.adaptation.probability_adapters import (
    KalmanPriorProbabilityAdapter,
    ProbabilityAdapter,
    PseudoChronoSinkhornProbabilityAdapter,
    RunningPriorProbabilityAdapter,
)
from neurips_bci.adaptation.signal_adapters import (
    CausalArtifactSignalAdapter,
    PseudoChronoSignalAdapter,
    SignalAdapter,
)

__all__ = [
    "Adapter",
    "SignalAdapter",
    "CausalArtifactSignalAdapter",
    "PseudoChronoSignalAdapter",
    "ProbabilityAdapter",
    "RunningPriorProbabilityAdapter",
    "KalmanPriorProbabilityAdapter",
    "PseudoChronoSinkhornProbabilityAdapter",
    "GateAdapter",
    "AsymmetricConfidenceGateAdapter",
    "ExcessAgreementGateAdapter",
    "SessionPaceGateAdapter",
]
