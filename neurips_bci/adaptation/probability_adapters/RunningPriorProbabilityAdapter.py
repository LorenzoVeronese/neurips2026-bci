"""Session-cumulative running prior probability balancer (BalStd)."""

from __future__ import annotations

import numpy as np

from neurips_bci.adaptation.probability_adapters.ProbabilityAdapter import ProbabilityAdapter
from neurips_bci.configuration.section_configurations.FusionPaceSectionConfiguration import (
    FusionPaceSectionConfiguration,
)
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class RunningPriorProbabilityAdapter(ProbabilityAdapter):
    """Divides incoming class probabilities by a Dirichlet-smoothed cumulative session prior."""

    def __init__(self, prior_strength: float = 10.0, class_count: int = 3) -> None:
        super().__init__(adapter_role="running_prior_probability_balancer", class_count=class_count)
        self._prior_strength = ParameterTypeValidator.require_float(prior_strength, "prior_strength", minimum_bound=0.0)
        self._cumulative_probabilities: np.ndarray = np.zeros(class_count, dtype=np.float64)
        self._effective_window_count: float = 0.0
        self._step_count: int = 0
        self.reset_session_state()

    @property
    def prior_strength(self) -> float:
        return self._prior_strength

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def cumulative_probabilities(self) -> np.ndarray:
        return self._cumulative_probabilities.copy()

    @classmethod
    def from_fusion_configuration(
        cls,
        fusion_configuration: FusionPaceSectionConfiguration,
        class_count: int = 3,
    ) -> RunningPriorProbabilityAdapter:
        if not isinstance(fusion_configuration, FusionPaceSectionConfiguration):
            raise TypeError(
                f"fusion_configuration must be FusionPaceSectionConfiguration, got {type(fusion_configuration).__name__}"
            )
        return cls(prior_strength=fusion_configuration.balance_std_prior, class_count=class_count)

    def reset_session_state(self) -> None:
        self._cumulative_probabilities = np.zeros(self._class_count, dtype=np.float64)
        self._effective_window_count = 0.0
        self._step_count = 0

    def adapt_probabilities(self, raw_probabilities: np.ndarray) -> np.ndarray:
        unit_probs = self._validate_and_normalize_simplex(raw_probabilities)
        self._step_count += 1
        self._cumulative_probabilities = self._cumulative_probabilities + unit_probs
        self._effective_window_count += 1.0
        running_prior = (
            self._prior_strength / float(self._class_count) + self._cumulative_probabilities
        ) / max(self._prior_strength + self._effective_window_count, 1e-12)
        reweighted = unit_probs / np.clip(running_prior, 1e-6, 1.0)
        return reweighted / float(np.sum(reweighted))

    def step(self, raw_probabilities: np.ndarray) -> np.ndarray:
        return self.adapt_probabilities(raw_probabilities)
