"""Random-walk scalar Kalman prior probability balancer (BalKal)."""

from __future__ import annotations

import numpy as np

from neurips_bci.adaptation.probability_adapters.ProbabilityAdapter import ProbabilityAdapter
from neurips_bci.configuration.section_configurations.FusionPaceSectionConfiguration import (
    FusionPaceSectionConfiguration,
)
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class KalmanPriorProbabilityAdapter(ProbabilityAdapter):
    """Tracks non-stationary marginal class drift via a scalar random-walk Kalman filter."""

    def __init__(
        self,
        initial_pseudo_count: float = 5.0,
        drift_variance: float = 0.002,
        class_count: int = 3,
    ) -> None:
        super().__init__(adapter_role="kalman_prior_probability_balancer", class_count=class_count)
        self._initial_pseudo_count = ParameterTypeValidator.require_float(
            initial_pseudo_count, "initial_pseudo_count", minimum_bound=0.0, strict_lower=True
        )
        self._drift_variance = ParameterTypeValidator.require_float(drift_variance, "drift_variance", minimum_bound=0.0)
        self._prior_estimate: np.ndarray = np.full(class_count, 1.0 / float(class_count), dtype=np.float64)
        self._posterior_variance: float = 1.0 / self._initial_pseudo_count
        self._step_count: int = 0
        self.reset_session_state()

    @property
    def initial_pseudo_count(self) -> float:
        return self._initial_pseudo_count

    @property
    def drift_variance(self) -> float:
        return self._drift_variance

    @property
    def prior_estimate(self) -> np.ndarray:
        return self._prior_estimate.copy()

    @classmethod
    def from_fusion_configuration(
        cls,
        fusion_configuration: FusionPaceSectionConfiguration,
        class_count: int = 3,
    ) -> KalmanPriorProbabilityAdapter:
        if not isinstance(fusion_configuration, FusionPaceSectionConfiguration):
            raise TypeError(
                f"fusion_configuration must be FusionPaceSectionConfiguration, got {type(fusion_configuration).__name__}"
            )
        return cls(
            initial_pseudo_count=fusion_configuration.kalman_n0,
            drift_variance=fusion_configuration.kalman_q_drift,
            class_count=class_count,
        )

    def reset_session_state(self) -> None:
        self._prior_estimate = np.full(self._class_count, 1.0 / float(self._class_count), dtype=np.float64)
        self._posterior_variance = 1.0 / max(self._initial_pseudo_count, 1e-6)
        self._step_count = 0

    def adapt_probabilities(self, raw_probabilities: np.ndarray) -> np.ndarray:
        unit_probs = self._validate_and_normalize_simplex(raw_probabilities)
        self._step_count += 1
        predicted_variance = self._posterior_variance + self._drift_variance
        kalman_gain = predicted_variance / (predicted_variance + 1.0)
        self._prior_estimate = self._prior_estimate + kalman_gain * (unit_probs - self._prior_estimate)
        self._posterior_variance = (1.0 - kalman_gain) * predicted_variance
        reweighted = unit_probs / np.clip(self._prior_estimate, 1e-6, 1.0)
        return reweighted / float(np.sum(reweighted))

    def step(self, raw_probabilities: np.ndarray) -> np.ndarray:
        return self.adapt_probabilities(raw_probabilities)
