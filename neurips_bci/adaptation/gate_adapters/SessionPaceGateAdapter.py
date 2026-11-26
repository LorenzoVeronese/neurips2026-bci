"""Affine session credit and global run-clock deadline governor (r35 and r45 PaceGuard)."""

from __future__ import annotations

from neurips_bci.adaptation.gate_adapters.GateAdapter import GateAdapter
from neurips_bci.configuration.section_configurations.FusionPaceSectionConfiguration import (
    FusionPaceSectionConfiguration,
)
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class SessionPaceGateAdapter(GateAdapter):
    """Governs monotonic per-session cost tiers (0..3) and self-training skip flags under wall-clock budgets."""

    def __init__(
        self,
        pace_configuration: FusionPaceSectionConfiguration,
        mode_probe_window: int = 40,
        min_self_train_credit_s: float = 1.0,
        self_train_credit_multiplier: float = 1.02,
    ) -> None:
        if not isinstance(pace_configuration, FusionPaceSectionConfiguration):
            raise TypeError(
                f"pace_configuration must be FusionPaceSectionConfiguration, got {type(pace_configuration).__name__}"
            )
        super().__init__(adapter_role="session_and_global_pace_deadline_gate")
        self._pace_configuration = pace_configuration
        self._mode_probe_window = ParameterTypeValidator.require_integer(
            mode_probe_window, "mode_probe_window", minimum_bound=8
        )
        self._min_self_train_credit_s = ParameterTypeValidator.require_float(
            min_self_train_credit_s, "min_self_train_credit_s", minimum_bound=0.0
        )
        self._self_train_credit_multiplier = ParameterTypeValidator.require_float(
            self_train_credit_multiplier, "self_train_credit_multiplier", minimum_bound=0.0
        )

        self._active_session_tier: int = 0
        self._hardware_speed_mode: str | None = None
        self._ordinary_window_durations: list[float] = []
        self._last_self_train_seconds: float = 0.0
        self._last_session_credit_seconds: float = pace_configuration.pace_credit_slow_f0_s
        self._tier_transition_log: list[tuple[int, int, float, float]] = []
        self.reset_session_state()

    @property
    def pace_configuration(self) -> FusionPaceSectionConfiguration:
        return self._pace_configuration

    @property
    def active_session_tier(self) -> int:
        return self._active_session_tier

    @property
    def hardware_speed_mode(self) -> str | None:
        return self._hardware_speed_mode

    @property
    def last_session_credit_seconds(self) -> float:
        return self._last_session_credit_seconds

    @property
    def tier_transition_log(self) -> list[tuple[int, int, float, float]]:
        return list(self._tier_transition_log)

    def reset_session_state(self) -> None:
        self._active_session_tier = 0
        self._hardware_speed_mode = None
        self._ordinary_window_durations = []
        self._last_self_train_seconds = 0.0
        self._last_session_credit_seconds = self._pace_configuration.pace_credit_slow_f0_s
        self._tier_transition_log = []

    def record_ordinary_window_duration(self, step_seconds: float) -> None:
        validated_dt = ParameterTypeValidator.require_float(step_seconds, "step_seconds", minimum_bound=0.0)
        self._ordinary_window_durations.append(validated_dt)

    def record_self_training_duration(self, self_train_seconds: float) -> None:
        validated_st = ParameterTypeValidator.require_float(self_train_seconds, "self_train_seconds", minimum_bound=0.0)
        self._last_self_train_seconds = validated_st

    def _resolve_effective_f0_seconds(self, window_index: int) -> float:
        cfg = self._pace_configuration
        if (
            self._hardware_speed_mode is None
            and window_index >= self._mode_probe_window
            and len(self._ordinary_window_durations) > 8
        ):
            warmup_trimmed = sorted(self._ordinary_window_durations[8:window_index])
            if warmup_trimmed:
                median_step_s = warmup_trimmed[len(warmup_trimmed) // 2]
                self._hardware_speed_mode = (
                    "fast" if median_step_s <= cfg.pace_fast_threshold_s else "slow"
                )
        if self._hardware_speed_mode == "fast":
            return cfg.pace_credit_fast_f0_s
        return cfg.pace_credit_slow_f0_s

    def evaluate_window_tier(
        self,
        window_index: int,
        session_elapsed_s: float,
        run_elapsed_s: float,
        is_checkpoint_next: bool,
        force_deterministic: bool = False,
    ) -> tuple[int, bool]:
        """Evaluate monotonic active cost tier (0..3) and skip_self_training flag for window_index."""
        ParameterTypeValidator.require_integer(window_index, "window_index", minimum_bound=0)
        ParameterTypeValidator.require_float(session_elapsed_s, "session_elapsed_s", minimum_bound=0.0)
        ParameterTypeValidator.require_float(run_elapsed_s, "run_elapsed_s", minimum_bound=0.0)
        ParameterTypeValidator.require_boolean(is_checkpoint_next, "is_checkpoint_next")
        ParameterTypeValidator.require_boolean(force_deterministic, "force_deterministic")

        if force_deterministic:
            return 0, False

        cfg = self._pace_configuration
        f0_seconds = self._resolve_effective_f0_seconds(window_index)
        session_credit = f0_seconds + cfg.pace_budget_per_window_s * float(window_index + 1) - float(session_elapsed_s)
        self._last_session_credit_seconds = session_credit

        if session_credit < cfg.pace_tier3_credit_s:
            candidate_tier = 3
        elif session_credit < cfg.pace_tier2_credit_s:
            candidate_tier = 2
        elif session_credit < cfg.pace_tier1_credit_s:
            candidate_tier = 1
        else:
            candidate_tier = 0

        if run_elapsed_s > cfg.global_tier3_seconds:
            candidate_tier = max(candidate_tier, 3)
        elif run_elapsed_s > cfg.global_tier2_seconds:
            candidate_tier = max(candidate_tier, 2)
        elif cfg.global_tier1_seconds is not None and run_elapsed_s > cfg.global_tier1_seconds:
            candidate_tier = max(candidate_tier, 1)

        if candidate_tier > self._active_session_tier:
            self._active_session_tier = candidate_tier
            self._tier_transition_log.append(
                (window_index, candidate_tier, round(session_elapsed_s, 3), round(session_credit, 3))
            )

        active_tier = self._active_session_tier
        st_shutoff_tier = 2 if cfg.tier1_keep_self_training else 1
        required_st_credit = max(
            self._min_self_train_credit_s,
            self._self_train_credit_multiplier * self._last_self_train_seconds,
        )
        insufficient_credit = (session_credit < required_st_credit) if is_checkpoint_next else (
            session_credit < required_st_credit
        )
        skip_self_training = bool((active_tier >= st_shutoff_tier) or insufficient_credit)
        return active_tier, skip_self_training
