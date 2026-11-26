"""Typed configuration for Trajectory A/B fusion, balancers, G1 reliability gate, and PaceGuard."""

from __future__ import annotations

from neurips_bci.configuration.section_configurations.SectionConfiguration import SectionConfiguration
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class FusionPaceSectionConfiguration(SectionConfiguration):
    """Defines dual-trajectory fusion weights, Sinkhorn/Kalman balancers, G1 gate, and r35/r45 pace guard."""

    def __init__(
        self,
        trajectory_a_weight: float = 0.40,
        traj_a_cnn_weight: float = 0.75,
        traj_a_lat52_weight: float = 0.06,
        traj_a_rms56_weight: float = 0.12,
        traj_a_tsr4_weight: float = 0.07,
        traj_b_cnn_weight: float = 0.90,
        traj_b_lat52_weight: float = 0.05,
        traj_b_rms56_weight: float = 0.05,
        traj_b_rsnse_weight: float = 0.08,
        balance_std_prior: float = 10.0,
        kalman_n0: float = 5.0,
        kalman_q_drift: float = 0.002,
        pc_sinkhorn_tau: float = 1.3,
        pc_sinkhorn_prior: float = 10.0,
        pc_sinkhorn_iterations: int = 4,
        g1_gate_enabled: bool = True,
        g1_gate_min: float = 0.25,
        g1_gate_max: float = 1.00,
        g1_clean_s2c_replacement: bool = False,
        pace_budget_per_window_s: float = 0.29,
        pace_fast_threshold_s: float = 0.30,
        pace_credit_fast_f0_s: float = 14.0,
        pace_credit_slow_f0_s: float = 6.0,
        pace_tier1_credit_s: float = 1.2,
        pace_tier2_credit_s: float = 0.6,
        pace_tier3_credit_s: float = 0.2,
        global_tier1_seconds: float | None = 3550.0,
        global_tier2_seconds: float = 4300.0,
        global_tier3_seconds: float = 4420.0,
        tier1_keep_self_training: bool = True,
    ) -> None:
        super().__init__(configuration_scope="dual_trajectory_fusion_and_pace_guard")
        self._trajectory_a_weight = ParameterTypeValidator.require_float(
            trajectory_a_weight, "trajectory_a_weight", 0.0, 1.0
        )
        self._traj_a_cnn_weight = ParameterTypeValidator.require_float(traj_a_cnn_weight, "traj_a_cnn_weight", 0.0, 1.0)
        self._traj_a_lat52_weight = ParameterTypeValidator.require_float(
            traj_a_lat52_weight, "traj_a_lat52_weight", 0.0, 1.0
        )
        self._traj_a_rms56_weight = ParameterTypeValidator.require_float(
            traj_a_rms56_weight, "traj_a_rms56_weight", 0.0, 1.0
        )
        self._traj_a_tsr4_weight = ParameterTypeValidator.require_float(
            traj_a_tsr4_weight, "traj_a_tsr4_weight", 0.0, 1.0
        )
        self._traj_b_cnn_weight = ParameterTypeValidator.require_float(traj_b_cnn_weight, "traj_b_cnn_weight", 0.0, 1.0)
        self._traj_b_lat52_weight = ParameterTypeValidator.require_float(
            traj_b_lat52_weight, "traj_b_lat52_weight", 0.0, 1.0
        )
        self._traj_b_rms56_weight = ParameterTypeValidator.require_float(
            traj_b_rms56_weight, "traj_b_rms56_weight", 0.0, 1.0
        )
        self._traj_b_rsnse_weight = ParameterTypeValidator.require_float(
            traj_b_rsnse_weight, "traj_b_rsnse_weight", 0.0, 1.0
        )
        self._balance_std_prior = ParameterTypeValidator.require_float(balance_std_prior, "balance_std_prior", 0.0)
        self._kalman_n0 = ParameterTypeValidator.require_float(kalman_n0, "kalman_n0", 0.0, strict_lower=True)
        self._kalman_q_drift = ParameterTypeValidator.require_float(kalman_q_drift, "kalman_q_drift", 0.0)
        self._pc_sinkhorn_tau = ParameterTypeValidator.require_float(pc_sinkhorn_tau, "pc_sinkhorn_tau", 0.0)
        self._pc_sinkhorn_prior = ParameterTypeValidator.require_float(pc_sinkhorn_prior, "pc_sinkhorn_prior", 0.0)
        self._pc_sinkhorn_iterations = ParameterTypeValidator.require_integer(
            pc_sinkhorn_iterations, "pc_sinkhorn_iterations", 1
        )
        self._g1_gate_enabled = ParameterTypeValidator.require_boolean(g1_gate_enabled, "g1_gate_enabled")
        self._g1_gate_min = ParameterTypeValidator.require_float(g1_gate_min, "g1_gate_min", 0.0, 1.0)
        self._g1_gate_max = ParameterTypeValidator.require_float(g1_gate_max, "g1_gate_max", 0.0, 1.0)
        self._g1_clean_s2c_replacement = ParameterTypeValidator.require_boolean(
            g1_clean_s2c_replacement, "g1_clean_s2c_replacement"
        )
        self._pace_budget_per_window_s = ParameterTypeValidator.require_float(
            pace_budget_per_window_s, "pace_budget_per_window_s", 0.0, strict_lower=True
        )
        self._pace_fast_threshold_s = ParameterTypeValidator.require_float(
            pace_fast_threshold_s, "pace_fast_threshold_s", 0.0, strict_lower=True
        )
        self._pace_credit_fast_f0_s = ParameterTypeValidator.require_float(
            pace_credit_fast_f0_s, "pace_credit_fast_f0_s", 0.0
        )
        self._pace_credit_slow_f0_s = ParameterTypeValidator.require_float(
            pace_credit_slow_f0_s, "pace_credit_slow_f0_s", 0.0
        )
        self._pace_tier1_credit_s = ParameterTypeValidator.require_float(pace_tier1_credit_s, "pace_tier1_credit_s", 0.0)
        self._pace_tier2_credit_s = ParameterTypeValidator.require_float(pace_tier2_credit_s, "pace_tier2_credit_s", 0.0)
        self._pace_tier3_credit_s = ParameterTypeValidator.require_float(pace_tier3_credit_s, "pace_tier3_credit_s", 0.0)
        self._global_tier1_seconds = (
            None
            if global_tier1_seconds is None
            else ParameterTypeValidator.require_float(global_tier1_seconds, "global_tier1_seconds", 0.0)
        )
        self._global_tier2_seconds = ParameterTypeValidator.require_float(
            global_tier2_seconds, "global_tier2_seconds", 0.0
        )
        self._global_tier3_seconds = ParameterTypeValidator.require_float(
            global_tier3_seconds, "global_tier3_seconds", 0.0
        )
        self._tier1_keep_self_training = ParameterTypeValidator.require_boolean(
            tier1_keep_self_training, "tier1_keep_self_training"
        )

    @property
    def trajectory_a_weight(self) -> float:
        return self._trajectory_a_weight

    @property
    def traj_a_cnn_weight(self) -> float:
        return self._traj_a_cnn_weight

    @property
    def traj_a_lat52_weight(self) -> float:
        return self._traj_a_lat52_weight

    @property
    def traj_a_rms56_weight(self) -> float:
        return self._traj_a_rms56_weight

    @property
    def traj_a_tsr4_weight(self) -> float:
        return self._traj_a_tsr4_weight

    @property
    def traj_b_cnn_weight(self) -> float:
        return self._traj_b_cnn_weight

    @property
    def traj_b_lat52_weight(self) -> float:
        return self._traj_b_lat52_weight

    @property
    def traj_b_rms56_weight(self) -> float:
        return self._traj_b_rms56_weight

    @property
    def traj_b_rsnse_weight(self) -> float:
        return self._traj_b_rsnse_weight

    @property
    def balance_std_prior(self) -> float:
        return self._balance_std_prior

    @property
    def kalman_n0(self) -> float:
        return self._kalman_n0

    @property
    def kalman_q_drift(self) -> float:
        return self._kalman_q_drift

    @property
    def pc_sinkhorn_tau(self) -> float:
        return self._pc_sinkhorn_tau

    @property
    def pc_sinkhorn_prior(self) -> float:
        return self._pc_sinkhorn_prior

    @property
    def pc_sinkhorn_iterations(self) -> int:
        return self._pc_sinkhorn_iterations

    @property
    def g1_gate_enabled(self) -> bool:
        return self._g1_gate_enabled

    @property
    def g1_gate_min(self) -> float:
        return self._g1_gate_min

    @property
    def g1_gate_max(self) -> float:
        return self._g1_gate_max

    @property
    def g1_clean_s2c_replacement(self) -> bool:
        return self._g1_clean_s2c_replacement

    @property
    def pace_budget_per_window_s(self) -> float:
        return self._pace_budget_per_window_s

    @property
    def pace_fast_threshold_s(self) -> float:
        return self._pace_fast_threshold_s

    @property
    def pace_credit_fast_f0_s(self) -> float:
        return self._pace_credit_fast_f0_s

    @property
    def pace_credit_slow_f0_s(self) -> float:
        return self._pace_credit_slow_f0_s

    @property
    def pace_tier1_credit_s(self) -> float:
        return self._pace_tier1_credit_s

    @property
    def pace_tier2_credit_s(self) -> float:
        return self._pace_tier2_credit_s

    @property
    def pace_tier3_credit_s(self) -> float:
        return self._pace_tier3_credit_s

    @property
    def global_tier1_seconds(self) -> float | None:
        return self._global_tier1_seconds

    @property
    def global_tier2_seconds(self) -> float:
        return self._global_tier2_seconds

    @property
    def global_tier3_seconds(self) -> float:
        return self._global_tier3_seconds

    @property
    def tier1_keep_self_training(self) -> bool:
        return self._tier1_keep_self_training

    def to_dictionary(self) -> dict[str, object]:
        return {
            "trajectory_a_weight": self._trajectory_a_weight,
            "traj_a_cnn_weight": self._traj_a_cnn_weight,
            "traj_a_lat52_weight": self._traj_a_lat52_weight,
            "traj_a_rms56_weight": self._traj_a_rms56_weight,
            "traj_a_tsr4_weight": self._traj_a_tsr4_weight,
            "traj_b_cnn_weight": self._traj_b_cnn_weight,
            "traj_b_lat52_weight": self._traj_b_lat52_weight,
            "traj_b_rms56_weight": self._traj_b_rms56_weight,
            "traj_b_rsnse_weight": self._traj_b_rsnse_weight,
            "balance_std_prior": self._balance_std_prior,
            "kalman_n0": self._kalman_n0,
            "kalman_q_drift": self._kalman_q_drift,
            "pc_sinkhorn_tau": self._pc_sinkhorn_tau,
            "pc_sinkhorn_prior": self._pc_sinkhorn_prior,
            "pc_sinkhorn_iterations": self._pc_sinkhorn_iterations,
            "g1_gate_enabled": self._g1_gate_enabled,
            "g1_gate_min": self._g1_gate_min,
            "g1_gate_max": self._g1_gate_max,
            "g1_clean_s2c_replacement": self._g1_clean_s2c_replacement,
            "pace_budget_per_window_s": self._pace_budget_per_window_s,
            "pace_fast_threshold_s": self._pace_fast_threshold_s,
            "pace_credit_fast_f0_s": self._pace_credit_fast_f0_s,
            "pace_credit_slow_f0_s": self._pace_credit_slow_f0_s,
            "pace_tier1_credit_s": self._pace_tier1_credit_s,
            "pace_tier2_credit_s": self._pace_tier2_credit_s,
            "pace_tier3_credit_s": self._pace_tier3_credit_s,
            "global_tier1_seconds": self._global_tier1_seconds,
            "global_tier2_seconds": self._global_tier2_seconds,
            "global_tier3_seconds": self._global_tier3_seconds,
            "tier1_keep_self_training": self._tier1_keep_self_training,
        }
