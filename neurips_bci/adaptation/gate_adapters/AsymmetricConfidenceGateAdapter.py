"""Causal B=1 decoupled asymmetric confidence-ceiling damper (LowDM_Asym) for ANCHOR-MM."""

from __future__ import annotations

import numpy as np

from neurips_bci.adaptation.gate_adapters.GateAdapter import GateAdapter
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class AsymmetricConfidenceGateAdapter(GateAdapter):
    """Computes strictly causal (0..t-1 prefix) decoupled ceiling gates (g_dm, g_proto) at each window."""

    def __init__(
        self,
        prefix_mode: str = "unbiased_ema",
        k_prior: float = 2.0,
        prior_confidence: float = 0.50,
        start_closed: bool = True,
        t0_uses_current: bool = False,
        ema_half_life: int = 24,
        min_prefix_windows: int = 8,
        instant_uncertainty_mode: str = "zero_maxp",
        tau_instant: float = 0.94,
        instant_power: float = 1.0,
        instant_cap: float = 1.6,
        tau_dm: float = 0.68,
        tau_proto: float = 0.80,
        width_dm: float = 0.06,
        width_proto: float = 0.08,
        min_gate_dm: float = 0.0,
        min_gate_proto: float = 0.0,
    ) -> None:
        super().__init__(adapter_role="asymmetric_confidence_ceiling_gate")
        self._prefix_mode = ParameterTypeValidator.require_choice(
            prefix_mode, ("unbiased_ema", "r21_biased"), "prefix_mode"
        )
        self._k_prior = ParameterTypeValidator.require_float(k_prior, "k_prior", minimum_bound=0.0)
        self._prior_confidence = ParameterTypeValidator.require_float(
            prior_confidence, "prior_confidence", minimum_bound=0.0, maximum_bound=1.0
        )
        self._start_closed = ParameterTypeValidator.require_boolean(start_closed, "start_closed")
        self._t0_uses_current = ParameterTypeValidator.require_boolean(t0_uses_current, "t0_uses_current")
        self._ema_half_life = ParameterTypeValidator.require_integer(ema_half_life, "ema_half_life", minimum_bound=0)
        self._min_prefix_windows = ParameterTypeValidator.require_integer(
            min_prefix_windows, "min_prefix_windows", minimum_bound=1
        )
        self._instant_uncertainty_mode = ParameterTypeValidator.require_choice(
            instant_uncertainty_mode,
            ("zero_maxp", "zero_margin", "r21_floor15", "none"),
            "instant_uncertainty_mode",
        )
        self._tau_instant = ParameterTypeValidator.require_float(tau_instant, "tau_instant", minimum_bound=0.0)
        self._instant_power = ParameterTypeValidator.require_float(instant_power, "instant_power", minimum_bound=0.0)
        self._instant_cap = ParameterTypeValidator.require_float(
            instant_cap, "instant_cap", minimum_bound=0.0, strict_lower=True
        )
        self._tau_dm = ParameterTypeValidator.require_float(tau_dm, "tau_dm", minimum_bound=0.0, maximum_bound=1.0)
        self._tau_proto = ParameterTypeValidator.require_float(
            tau_proto, "tau_proto", minimum_bound=0.0, maximum_bound=1.0
        )
        self._width_dm = ParameterTypeValidator.require_float(
            width_dm, "width_dm", minimum_bound=0.0, strict_lower=True
        )
        self._width_proto = ParameterTypeValidator.require_float(
            width_proto, "width_proto", minimum_bound=0.0, strict_lower=True
        )
        self._min_gate_dm = ParameterTypeValidator.require_float(
            min_gate_dm, "min_gate_dm", minimum_bound=0.0, maximum_bound=1.0
        )
        self._min_gate_proto = ParameterTypeValidator.require_float(
            min_gate_proto, "min_gate_proto", minimum_bound=0.0, maximum_bound=1.0
        )
        self._alpha_decay = (
            float(np.exp(-np.log(2.0) / float(max(1, self._ema_half_life)))) if self._ema_half_life > 0 else 0.0
        )

        self._clean_step_count: int = 0
        self._biased_state_dm: float = 0.0
        self._biased_state_proto: float = 0.0
        self._ema_num_dm: float = 0.0
        self._ema_den_dm: float = 0.0
        self._cum_sum_dm: float = 0.0
        self._ema_num_proto: float = 0.0
        self._ema_den_proto: float = 0.0
        self._cum_sum_proto: float = 0.0
        self.reset_session_state()

    def reset_session_state(self) -> None:
        self._clean_step_count = 0
        self._biased_state_dm = self._tau_dm if self._start_closed else self._prior_confidence
        self._biased_state_proto = self._tau_proto if self._start_closed else self._prior_confidence
        self._ema_num_dm = 0.0
        self._ema_den_dm = 0.0
        self._cum_sum_dm = 0.0
        self._ema_num_proto = 0.0
        self._ema_den_proto = 0.0
        self._cum_sum_proto = 0.0

    def _instantaneous_uncertainty_factor(self, probs_vec: np.ndarray) -> float:
        if self._instant_power <= 0.0 or self._instant_uncertainty_mode == "none":
            return 1.0
        max_prob = float(np.max(probs_vec))
        if self._instant_uncertainty_mode == "r21_floor15":
            return float(np.clip((1.0 - max_prob) / 0.50, 0.15, 2.0) ** self._instant_power)
        if self._instant_uncertainty_mode == "zero_maxp":
            denom = max(1e-6, self._tau_instant - 0.50)
            raw_ratio = float(np.clip((self._tau_instant - max_prob) / denom, 0.0, self._instant_cap))
            return float(raw_ratio ** self._instant_power) if raw_ratio > 0.0 else 0.0
        sorted_probs = np.sort(probs_vec)
        top_margin = float(sorted_probs[-1] - sorted_probs[-2])
        denom = max(1e-6, self._tau_instant - 0.25)
        raw_ratio = float(np.clip((self._tau_instant - top_margin) / denom, 0.0, self._instant_cap))
        return float(raw_ratio ** self._instant_power) if raw_ratio > 0.0 else 0.0

    def step_gates(
        self,
        reference_probs: np.ndarray,
        is_bad_window: bool = False,
        proto_reference_probs: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Compute causal (g_dm, g_proto) using 0..t-1 clean prefix, then update prefix if clean."""
        ParameterTypeValidator.require_numpy_array(reference_probs, "reference_probs", expected_ndim=1)
        ParameterTypeValidator.require_boolean(is_bad_window, "is_bad_window")
        probs_dm = np.asarray(reference_probs, dtype=np.float64)
        probs_proto = (
            probs_dm
            if proto_reference_probs is None
            else np.asarray(
                ParameterTypeValidator.require_numpy_array(
                    proto_reference_probs, "proto_reference_probs", expected_ndim=1
                ),
                dtype=np.float64,
            )
        )
        conf_dm = float(np.max(probs_dm))
        conf_proto = float(np.max(probs_proto))

        prior_dm = self._tau_dm if self._start_closed else self._prior_confidence
        prior_proto = self._tau_proto if self._start_closed else self._prior_confidence

        if self._prefix_mode == "r21_biased":
            weight_eff = (
                float(self._clean_step_count) / float(self._clean_step_count + self._min_prefix_windows)
                if self._clean_step_count > 0
                else 0.0
            )
            hist_dm = (1.0 - weight_eff) * prior_dm + weight_eff * self._biased_state_dm
            hist_proto = (1.0 - weight_eff) * prior_proto + weight_eff * self._biased_state_proto
            if not is_bad_window:
                self._biased_state_dm = self._alpha_decay * self._biased_state_dm + (1.0 - self._alpha_decay) * conf_dm
                self._biased_state_proto = (
                    self._alpha_decay * self._biased_state_proto + (1.0 - self._alpha_decay) * conf_proto
                )
        else:
            if self._clean_step_count == 0:
                hist_dm = conf_dm if (self._t0_uses_current and not is_bad_window) else prior_dm
                hist_proto = conf_proto if (self._t0_uses_current and not is_bad_window) else prior_proto
            else:
                ema_dm = (
                    (self._ema_num_dm / self._ema_den_dm)
                    if self._ema_half_life > 0
                    else (self._cum_sum_dm / float(self._clean_step_count))
                )
                ema_proto = (
                    (self._ema_num_proto / self._ema_den_proto)
                    if self._ema_half_life > 0
                    else (self._cum_sum_proto / float(self._clean_step_count))
                )
                if self._k_prior > 0.0:
                    count_f = float(self._clean_step_count)
                    hist_dm = (self._k_prior * prior_dm + count_f * ema_dm) / (self._k_prior + count_f)
                    hist_proto = (self._k_prior * prior_proto + count_f * ema_proto) / (self._k_prior + count_f)
                else:
                    hist_dm = ema_dm
                    hist_proto = ema_proto

            if not is_bad_window:
                self._ema_num_dm = self._alpha_decay * self._ema_num_dm + conf_dm
                self._ema_den_dm = self._alpha_decay * self._ema_den_dm + 1.0
                self._cum_sum_dm += conf_dm
                self._ema_num_proto = self._alpha_decay * self._ema_num_proto + conf_proto
                self._ema_den_proto = self._alpha_decay * self._ema_den_proto + 1.0
                self._cum_sum_proto += conf_proto

        if not is_bad_window:
            self._clean_step_count += 1

        gate_dm = float(np.clip((self._tau_dm - hist_dm) / self._width_dm, self._min_gate_dm, 1.0)) * (
            self._instantaneous_uncertainty_factor(probs_dm)
        )
        gate_proto = float(np.clip((self._tau_proto - hist_proto) / self._width_proto, self._min_gate_proto, 1.0)) * (
            self._instantaneous_uncertainty_factor(probs_proto)
        )
        return gate_dm, gate_proto
