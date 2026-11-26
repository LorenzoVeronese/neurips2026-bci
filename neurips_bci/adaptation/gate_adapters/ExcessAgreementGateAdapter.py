"""Causal B=1 excess-agreement reliability gate (G1) for foundation model organs."""

from __future__ import annotations

import numpy as np

from neurips_bci.adaptation.gate_adapters.GateAdapter import GateAdapter
from neurips_bci.configuration.section_configurations.FusionPaceSectionConfiguration import (
    FusionPaceSectionConfiguration,
)
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class ExcessAgreementGateAdapter(GateAdapter):
    """Weights auxiliary foundation models by causal excess inner-product agreement above 1/3 chance."""

    def __init__(
        self,
        enabled: bool = True,
        min_gate: float = 0.25,
        max_gate: float = 1.00,
        agreement_low: float = 0.005,
        agreement_high: float = 0.055,
        prior_pseudo_weight: float = 12.0,
        prior_excess_agreement: float = 0.035,
        warmup_windows: int = 6,
        anchor_confidence_floor: float = 0.45,
        anchor_confidence_span: float = 0.35,
        chance_agreement: float = 1.0 / 3.0,
    ) -> None:
        super().__init__(adapter_role="excess_agreement_reliability_gate")
        self._enabled = ParameterTypeValidator.require_boolean(enabled, "enabled")
        self._min_gate = ParameterTypeValidator.require_float(min_gate, "min_gate", minimum_bound=0.0, maximum_bound=1.0)
        self._max_gate = ParameterTypeValidator.require_float(
            max_gate, "max_gate", minimum_bound=self._min_gate, maximum_bound=1.0
        )
        self._agreement_low = ParameterTypeValidator.require_float(agreement_low, "agreement_low")
        self._agreement_high = ParameterTypeValidator.require_float(
            agreement_high, "agreement_high", minimum_bound=self._agreement_low, strict_lower=True
        )
        self._prior_pseudo_weight = ParameterTypeValidator.require_float(
            prior_pseudo_weight, "prior_pseudo_weight", minimum_bound=0.0
        )
        self._prior_excess_agreement = ParameterTypeValidator.require_float(
            prior_excess_agreement, "prior_excess_agreement"
        )
        self._warmup_windows = ParameterTypeValidator.require_integer(warmup_windows, "warmup_windows", minimum_bound=0)
        self._anchor_confidence_floor = ParameterTypeValidator.require_float(
            anchor_confidence_floor, "anchor_confidence_floor", minimum_bound=0.0, maximum_bound=1.0
        )
        self._anchor_confidence_span = ParameterTypeValidator.require_float(
            anchor_confidence_span, "anchor_confidence_span", minimum_bound=0.0, strict_lower=True
        )
        self._chance_agreement = ParameterTypeValidator.require_float(
            chance_agreement, "chance_agreement", minimum_bound=0.0, maximum_bound=1.0
        )

        self._weighted_count: float = 0.0
        self._weighted_excess_sum: float = 0.0
        self._observed_window_count: int = 0
        self.reset_session_state()

    @classmethod
    def from_fusion_configuration(
        cls,
        fusion_configuration: FusionPaceSectionConfiguration,
    ) -> ExcessAgreementGateAdapter:
        if not isinstance(fusion_configuration, FusionPaceSectionConfiguration):
            raise TypeError(
                f"fusion_configuration must be FusionPaceSectionConfiguration, got {type(fusion_configuration).__name__}"
            )
        return cls(
            enabled=fusion_configuration.g1_gate_enabled,
            min_gate=fusion_configuration.g1_gate_min,
            max_gate=fusion_configuration.g1_gate_max,
        )

    def reset_session_state(self) -> None:
        self._weighted_count = 0.0
        self._weighted_excess_sum = 0.0
        self._observed_window_count = 0

    def current_reliability_weight(self) -> float:
        """Compute causal reliability weight g_t in [min_gate, max_gate] from 0..t-1 observations."""
        if not self._enabled:
            return 1.0
        if self._observed_window_count >= self._warmup_windows and self._weighted_count > 0.0:
            posterior_excess = (
                self._prior_pseudo_weight * self._prior_excess_agreement + self._weighted_excess_sum
            ) / (self._prior_pseudo_weight + self._weighted_count)
        else:
            posterior_excess = self._prior_excess_agreement
        ramp_fraction = float(
            np.clip(
                (posterior_excess - self._agreement_low) / max(self._agreement_high - self._agreement_low, 1e-6),
                0.0,
                1.0,
            )
        )
        return float(self._min_gate + (self._max_gate - self._min_gate) * ramp_fraction)

    def observe_agreement(
        self,
        anchor_probs: np.ndarray,
        candidate_probs: np.ndarray,
        is_clean_window: bool = True,
    ) -> None:
        """Update session agreement posterior after scoring window t."""
        ParameterTypeValidator.require_numpy_array(anchor_probs, "anchor_probs", expected_ndim=1)
        ParameterTypeValidator.require_numpy_array(
            candidate_probs, "candidate_probs", expected_shape=anchor_probs.shape
        )
        ParameterTypeValidator.require_boolean(is_clean_window, "is_clean_window")
        self._observed_window_count += 1
        if not self._enabled or not is_clean_window:
            return
        anc_vec = np.asarray(anchor_probs, dtype=np.float64)
        cand_vec = np.asarray(candidate_probs, dtype=np.float64)
        max_anchor = float(np.max(anc_vec))
        if max_anchor >= self._anchor_confidence_floor:
            obs_weight = min(1.0, (max_anchor - self._anchor_confidence_floor) / self._anchor_confidence_span)
            excess_dot = float(np.dot(anc_vec, cand_vec)) - self._chance_agreement
            self._weighted_count += obs_weight
            self._weighted_excess_sum += obs_weight * excess_dot
