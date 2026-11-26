"""Causal PseudoChrono run-neighborhood Sinkhorn-Knopp cell balancer (pc_sk)."""

from __future__ import annotations

import numpy as np

from neurips_bci.adaptation.probability_adapters.ProbabilityAdapter import ProbabilityAdapter
from neurips_bci.configuration.section_configurations.FusionPaceSectionConfiguration import (
    FusionPaceSectionConfiguration,
)
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class PseudoChronoSinkhornProbabilityAdapter(ProbabilityAdapter):
    """Causal B=1 Sinkhorn scaling over RBF-weighted run-neighborhood probability histories."""

    def __init__(
        self,
        kernel_tau: float = 1.3,
        prior_strength: float = 10.0,
        global_pseudo_count: float = 2.0,
        self_weight: float = 0.25,
        sinkhorn_iterations: int = 4,
        subspace_dim: int = 6,
        max_history_windows: int = 512,
        class_count: int = 3,
    ) -> None:
        super().__init__(adapter_role="pseudochrono_sinkhorn_probability_balancer", class_count=class_count)
        self._kernel_tau = ParameterTypeValidator.require_float(kernel_tau, "kernel_tau", minimum_bound=0.0)
        self._prior_strength = ParameterTypeValidator.require_float(prior_strength, "prior_strength", minimum_bound=0.0)
        self._global_pseudo_count = ParameterTypeValidator.require_float(
            global_pseudo_count, "global_pseudo_count", minimum_bound=0.0
        )
        self._self_weight = ParameterTypeValidator.require_float(self_weight, "self_weight", minimum_bound=0.0)
        self._sinkhorn_iterations = ParameterTypeValidator.require_integer(
            sinkhorn_iterations, "sinkhorn_iterations", minimum_bound=1
        )
        self._subspace_dim = ParameterTypeValidator.require_integer(subspace_dim, "subspace_dim", minimum_bound=1)
        self._max_history_windows = ParameterTypeValidator.require_integer(
            max_history_windows, "max_history_windows", minimum_bound=1
        )

        self._total_window_count: int = 0
        self._global_probability_sum: np.ndarray = np.zeros(class_count, dtype=np.float64)
        self._clean_u_coordinates: list[np.ndarray] = []
        self._clean_probabilities: list[np.ndarray] = []
        self.reset_session_state()

    @classmethod
    def from_fusion_configuration(
        cls,
        fusion_configuration: FusionPaceSectionConfiguration,
        subspace_dim: int = 6,
        class_count: int = 3,
    ) -> PseudoChronoSinkhornProbabilityAdapter:
        if not isinstance(fusion_configuration, FusionPaceSectionConfiguration):
            raise TypeError(
                f"fusion_configuration must be FusionPaceSectionConfiguration, got {type(fusion_configuration).__name__}"
            )
        return cls(
            kernel_tau=fusion_configuration.pc_sinkhorn_tau,
            prior_strength=fusion_configuration.pc_sinkhorn_prior,
            sinkhorn_iterations=fusion_configuration.pc_sinkhorn_iterations,
            subspace_dim=subspace_dim,
            class_count=class_count,
        )

    def reset_session_state(self) -> None:
        self._total_window_count = 0
        self._global_probability_sum = np.zeros(self._class_count, dtype=np.float64)
        self._clean_u_coordinates = []
        self._clean_probabilities = []

    def _compute_past_weights(self, u_coordinate: np.ndarray | None) -> np.ndarray:
        if u_coordinate is None or not self._clean_u_coordinates or self._kernel_tau <= 0.0:
            return np.zeros(0, dtype=np.float64)
        coord_vec = np.asarray(u_coordinate, dtype=np.float64)
        past_mat = np.stack(self._clean_u_coordinates, axis=0)
        sq_dist = np.sum((past_mat - coord_vec[None, :]) ** 2, axis=1)
        inv_2dt2 = 0.5 / (float(self._subspace_dim) * (self._kernel_tau ** 2))
        return np.exp(-inv_2dt2 * sq_dist)

    def adapt_probabilities(
        self,
        raw_probabilities: np.ndarray,
        u_coordinate: np.ndarray | None = None,
        is_bad_window: bool = False,
    ) -> np.ndarray:
        unit_probs = self._validate_and_normalize_simplex(raw_probabilities)
        self._total_window_count += 1
        self._global_probability_sum = self._global_probability_sum + unit_probs
        global_mean_probs = self._global_probability_sum / float(self._total_window_count)

        past_weights = self._compute_past_weights(u_coordinate)
        if len(past_weights) > 0:
            past_prob_mat = np.stack(self._clean_probabilities, axis=0)
            if self._self_weight > 0.0:
                weight_vec = np.concatenate([past_weights, np.array([self._self_weight], dtype=np.float64)])
                prob_mat = np.vstack([past_prob_mat, unit_probs[None, :]])
            else:
                weight_vec = past_weights
                prob_mat = past_prob_mat
        elif self._self_weight > 0.0:
            weight_vec = np.array([self._self_weight], dtype=np.float64)
            prob_mat = unit_probs[None, :]
        else:
            weight_vec = np.zeros(0, dtype=np.float64)
            prob_mat = np.zeros((0, self._class_count), dtype=np.float64)

        total_mass = self._prior_strength + self._global_pseudo_count + float(np.sum(weight_vec))
        col_scales = np.ones(self._class_count, dtype=np.float64)
        class_f = float(self._class_count)

        for _ in range(self._sinkhorn_iterations):
            q_prior = col_scales / max(float(np.sum(col_scales)), 1e-15)
            ug_scaled = global_mean_probs * col_scales
            q_global = ug_scaled / max(float(np.sum(ug_scaled)), 1e-15)
            if len(weight_vec) > 0:
                scaled_rows = prob_mat * col_scales[None, :]
                scaled_rows = scaled_rows / np.maximum(np.sum(scaled_rows, axis=-1, keepdims=True), 1e-15)
                local_marginal = weight_vec @ scaled_rows
            else:
                local_marginal = 0.0
            combined_marginal = (
                self._prior_strength * q_prior + self._global_pseudo_count * q_global + local_marginal
            ) / max(total_mass, 1e-8)
            col_scales = col_scales / (class_f * np.clip(combined_marginal, 1e-6, 1.0))
            col_scales = col_scales / float(np.mean(col_scales))

        scaled_out = unit_probs * col_scales
        balanced_probs = scaled_out / float(np.sum(scaled_out))

        if (not is_bad_window) and (u_coordinate is not None):
            coord_copy = np.asarray(u_coordinate, dtype=np.float64).copy()
            self._clean_u_coordinates.append(coord_copy)
            self._clean_probabilities.append(unit_probs.copy())
            if len(self._clean_u_coordinates) > self._max_history_windows:
                self._clean_u_coordinates.pop(0)
                self._clean_probabilities.pop(0)

        return balanced_probs

    def step(
        self,
        raw_probabilities: np.ndarray,
        u_coordinate: np.ndarray | None = None,
        is_bad_window: bool = False,
    ) -> np.ndarray:
        return self.adapt_probabilities(
            raw_probabilities=raw_probabilities,
            u_coordinate=u_coordinate,
            is_bad_window=is_bad_window,
        )
