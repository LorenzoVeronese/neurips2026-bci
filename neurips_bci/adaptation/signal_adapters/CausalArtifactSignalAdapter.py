"""Causal B=1 artifact guard with dead-channel latching, loose-electrode zeroing, Welford gate, and MAD clipping."""

from __future__ import annotations

import numpy as np

from neurips_bci.adaptation.signal_adapters.SignalAdapter import SignalAdapter
from neurips_bci.configuration.section_configurations.ClassicalHeadsSectionConfiguration import (
    ClassicalHeadsSectionConfiguration,
)
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class CausalArtifactSignalAdapter(SignalAdapter):
    """Sanitizes incoming (C, T) windows causally and flags extreme amplitude/covariance outliers."""

    def __init__(
        self,
        z_amplitude_threshold: float = 8.0,
        z_logdet_threshold: float = 8.0,
        prior_pseudo_count: int = 10,
        dead_channel_fraction: float = 0.01,
        dead_latch_windows: int = 10,
        dead_check_every_window: bool = True,
        loose_variance_ratio: float | None = 16.0,
        loose_consecutive_windows: int = 3,
        spike_clip_z: float | None = 32.0,
        spike_clip_target_z: float | None = 8.0,
        repair_dead_channels: bool = False,
        interpolation_matrix: np.ndarray | None = None,
        prior_statistics: dict[str, object] | None = None,
    ) -> None:
        super().__init__(adapter_role="causal_artifact_signal_guard")
        self._z_amplitude_threshold = ParameterTypeValidator.require_float(
            z_amplitude_threshold, "z_amplitude_threshold", minimum_bound=1.0
        )
        self._z_logdet_threshold = ParameterTypeValidator.require_float(
            z_logdet_threshold, "z_logdet_threshold", minimum_bound=1.0
        )
        self._prior_pseudo_count = ParameterTypeValidator.require_integer(
            prior_pseudo_count, "prior_pseudo_count", minimum_bound=1
        )
        self._dead_channel_fraction = ParameterTypeValidator.require_float(
            dead_channel_fraction, "dead_channel_fraction", minimum_bound=0.0, maximum_bound=1.0
        )
        self._dead_latch_windows = ParameterTypeValidator.require_integer(
            dead_latch_windows, "dead_latch_windows", minimum_bound=1
        )
        self._dead_check_every_window = ParameterTypeValidator.require_boolean(
            dead_check_every_window, "dead_check_every_window"
        )
        self._loose_variance_ratio = (
            None
            if loose_variance_ratio is None
            else ParameterTypeValidator.require_float(
                loose_variance_ratio, "loose_variance_ratio", minimum_bound=1.0, strict_lower=True
            )
        )
        self._loose_consecutive_windows = ParameterTypeValidator.require_integer(
            loose_consecutive_windows, "loose_consecutive_windows", minimum_bound=1
        )
        self._spike_clip_z = (
            None
            if spike_clip_z is None
            else ParameterTypeValidator.require_float(spike_clip_z, "spike_clip_z", minimum_bound=1.0)
        )
        self._spike_clip_target_z = (
            None
            if spike_clip_target_z is None
            else ParameterTypeValidator.require_float(spike_clip_target_z, "spike_clip_target_z", minimum_bound=1.0)
        )
        self._repair_dead_channels = ParameterTypeValidator.require_boolean(
            repair_dead_channels, "repair_dead_channels"
        )
        self._interpolation_matrix = (
            None
            if interpolation_matrix is None
            else np.asarray(
                ParameterTypeValidator.require_numpy_array(
                    interpolation_matrix, "interpolation_matrix", expected_ndim=2
                ),
                dtype=np.float64,
            )
        )
        if prior_statistics is not None and not isinstance(prior_statistics, dict):
            raise TypeError(f"prior_statistics must be dict or None, got {type(prior_statistics).__name__}")
        self._prior_statistics = None if prior_statistics is None else dict(prior_statistics)

        self._step_counter: int = 0
        self._variance_sum: np.ndarray | None = None
        self._dead_channel_mask: np.ndarray | None = None
        self._welford_state: dict[str, list[float]] = {}
        self._log_rel_var_count: np.ndarray | None = None
        self._log_rel_var_mean: np.ndarray | None = None
        self._recent_log_rel_vars: list[np.ndarray] = []
        self._mad_scale_sum: np.ndarray | None = None
        self._mad_scale_count: np.ndarray | None = None
        self._latest_diagnostics: dict[str, object] = {}
        self.reset_session_state()

    @property
    def dead_channel_mask(self) -> np.ndarray | None:
        return None if self._dead_channel_mask is None else self._dead_channel_mask.copy()

    @property
    def latest_diagnostics(self) -> dict[str, object]:
        return dict(self._latest_diagnostics)

    def reset_session_state(self) -> None:
        self._step_counter = 0
        self._variance_sum = None
        self._dead_channel_mask = None
        n0_float = float(self._prior_pseudo_count)
        prior_map = self._prior_statistics
        self._welford_state = {}
        for metric_key in ("lrms", "ld"):
            if prior_map is not None and metric_key in prior_map:
                pair_stats = prior_map[metric_key]
                prior_mean = float(pair_stats[0])  # type: ignore[index]
                prior_sd = float(pair_stats[1])  # type: ignore[index]
                self._welford_state[metric_key] = [n0_float, prior_mean, n0_float * (prior_sd ** 2)]
            else:
                self._welford_state[metric_key] = [0.0, 0.0, 0.0]

        lv_prior = None if prior_map is None else prior_map.get("lv")
        if lv_prior is not None:
            lv_arr = np.asarray(lv_prior, dtype=np.float64)
            self._log_rel_var_count = np.full(len(lv_arr), n0_float, dtype=np.float64)
            self._log_rel_var_mean = lv_arr.copy()
        else:
            self._log_rel_var_count = None
            self._log_rel_var_mean = None
        self._recent_log_rel_vars = []
        self._mad_scale_sum = None
        self._mad_scale_count = None
        self._latest_diagnostics = {}

    @staticmethod
    def compute_window_measures(
        window_matrix: np.ndarray,
        live_channel_mask: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Compute (log RMS amplitude, mean covariance log-eigenvalue per live channel)."""
        centered = np.asarray(window_matrix, dtype=np.float64)
        centered = centered - centered.mean(axis=-1, keepdims=True)
        if live_channel_mask is not None:
            centered = centered[live_channel_mask]
        if len(centered) == 0:
            return -np.inf, -np.inf
        det_sign, cov_logdet = np.linalg.slogdet((centered @ centered.T) / float(centered.shape[-1]))
        log_rms = 0.5 * float(np.log(np.mean(centered * centered) + 1e-300))
        mean_log_eig = float(cov_logdet / float(len(centered))) if det_sign > 0 else -np.inf
        return log_rms, mean_log_eig

    @staticmethod
    def compute_log_relative_variance(window_matrix: np.ndarray) -> np.ndarray:
        chan_var = np.var(np.asarray(window_matrix, dtype=np.float64), axis=-1)
        median_var = max(float(np.median(chan_var)), 1e-300)
        return np.log(np.maximum(chan_var, 1e-300) / median_var)

    @classmethod
    def fit_training_prior(
        cls,
        training_windows: np.ndarray,
        session_identifiers: np.ndarray | None = None,
    ) -> dict[str, object]:
        """Fit training distribution priors {'lrms': [mean, sd], 'ld': [mean, sd], 'lv': [...]}."""
        ParameterTypeValidator.require_numpy_array(training_windows, "training_windows", expected_ndim=3)
        if len(training_windows) == 0:
            raise ValueError("training_windows must be non-empty")
        measures_mat = np.array([cls.compute_window_measures(win_arr) for win_arr in training_windows], dtype=np.float64)
        prior_dict: dict[str, object] = {}
        for col_idx, metric_key in enumerate(("lrms", "ld")):
            col_series = measures_mat[:, col_idx]
            finite_mask = np.isfinite(col_series)
            usable_series = col_series[finite_mask] if np.any(finite_mask) else np.zeros(1, dtype=np.float64)
            if session_identifiers is not None:
                sess_arr = np.asarray(session_identifiers)
                sess_sds = [
                    float(col_series[(sess_arr == sess_k) & finite_mask].std())
                    for sess_k in np.unique(sess_arr)
                    if int(np.sum((sess_arr == sess_k) & finite_mask)) >= 2
                ]
                spread_sd = float(np.median(sess_sds)) if sess_sds else float(usable_series.std())
            else:
                spread_sd = float(usable_series.std())
            prior_dict[metric_key] = [float(usable_series.mean()), max(spread_sd, 1e-6)]
        mean_lv = np.mean([cls.compute_log_relative_variance(win_arr) for win_arr in training_windows], axis=0)
        prior_dict["lv"] = mean_lv.astype(np.float64).tolist()
        return prior_dict

    @classmethod
    def from_prior_dictionary(
        cls,
        config_section: ClassicalHeadsSectionConfiguration,
        prior_dict: dict[str, object] | None,
    ) -> CausalArtifactSignalAdapter:
        if not isinstance(config_section, ClassicalHeadsSectionConfiguration):
            raise TypeError(
                f"config_section must be ClassicalHeadsSectionConfiguration, got {type(config_section).__name__}"
            )
        return cls(
            z_amplitude_threshold=config_section.artifact_z_amplitude,
            z_logdet_threshold=config_section.artifact_z_logdet,
            prior_pseudo_count=config_section.artifact_prior_n0,
            spike_clip_z=config_section.artifact_clip_z,
            spike_clip_target_z=config_section.artifact_clip_target,
            prior_statistics=prior_dict,
        )

    def _compute_welford_z(self, metric_key: str, metric_scalar: float) -> float:
        if not np.isfinite(metric_scalar):
            return 0.0
        obs_count, running_mean, running_m2 = self._welford_state[metric_key]
        if obs_count < float(self._prior_pseudo_count) or running_m2 <= 0.0:
            return 0.0
        return float((metric_scalar - running_mean) / max(np.sqrt(running_m2 / obs_count), 1e-12))

    def _append_welford_observation(self, metric_key: str, metric_scalar: float) -> None:
        state_vec = self._welford_state[metric_key]
        state_vec[0] += 1.0
        delta_pre = metric_scalar - state_vec[1]
        state_vec[1] += delta_pre / state_vec[0]
        state_vec[2] += delta_pre * (metric_scalar - state_vec[1])

    def _detect_dead_channels(self, finite_window: np.ndarray, allow_latch: bool = True) -> np.ndarray:
        centered = finite_window - finite_window.mean(axis=-1, keepdims=True)
        chan_var = centered.var(axis=-1)
        if self._dead_channel_mask is None or self._variance_sum is None:
            self._variance_sum = np.zeros_like(chan_var, dtype=np.float64)
            self._dead_channel_mask = np.zeros(len(chan_var), dtype=bool)
        if not allow_latch:
            return chan_var

        flat_mask = np.ptp(finite_window, axis=-1) == 0.0
        if self._step_counter <= self._dead_latch_windows:
            self._variance_sum = self._variance_sum + chan_var
            mean_var = self._variance_sum / float(self._step_counter)
            self._dead_channel_mask |= mean_var < (self._dead_channel_fraction * float(np.median(mean_var)))
        if self._dead_check_every_window:
            med_var = float(np.median(chan_var))
            self._dead_channel_mask |= chan_var < (self._dead_channel_fraction * med_var)
            self._latest_diagnostics["dead_min_ratio"] = float(chan_var.min() / max(med_var, 1e-300))
        self._dead_channel_mask |= flat_mask
        return chan_var

    def _detect_loose_channels(self, finite_window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cur_lv = self.compute_log_relative_variance(finite_window)
        if self._log_rel_var_mean is None or self._log_rel_var_count is None:
            self._log_rel_var_mean = np.zeros(len(cur_lv), dtype=np.float64)
            self._log_rel_var_count = np.zeros(len(cur_lv), dtype=np.float64)
        self._recent_log_rel_vars = (self._recent_log_rel_vars + [cur_lv])[-self._loose_consecutive_windows :]
        loose_mask = np.zeros(len(cur_lv), dtype=bool)
        if len(self._recent_log_rel_vars) == self._loose_consecutive_windows and self._dead_channel_mask is not None:
            excess_log = np.min(self._recent_log_rel_vars, axis=0) - self._log_rel_var_mean
            excess_log[self._dead_channel_mask] = -np.inf
            self._latest_diagnostics["loose_max_ratio"] = float(np.exp(np.max(excess_log)))
            if self._loose_variance_ratio is not None:
                loose_mask = (excess_log > np.log(self._loose_variance_ratio)) & (
                    self._log_rel_var_count >= float(self._prior_pseudo_count)
                )
        return loose_mask, cur_lv

    def _compute_channel_mad_scale(self, finite_window: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        chan_med = np.median(finite_window, axis=-1, keepdims=True)
        win_mad = 1.4826 * np.median(np.abs(finite_window - chan_med), axis=-1)
        running_mad = win_mad
        if self._mad_scale_sum is not None and self._mad_scale_count is not None:
            running_mad = np.where(
                self._mad_scale_count > 0,
                self._mad_scale_sum / np.maximum(self._mad_scale_count, 1.0),
                win_mad,
            )
        return chan_med, win_mad, np.maximum(running_mad, win_mad)

    def _repair_dead_electrodes(self, finite_window: np.ndarray) -> np.ndarray:
        assert self._dead_channel_mask is not None and self._interpolation_matrix is not None
        live_mask = ~self._dead_channel_mask
        repaired = np.array(finite_window, dtype=np.float64, copy=True)
        for dead_ch in np.flatnonzero(self._dead_channel_mask):
            row_weights = self._interpolation_matrix[dead_ch] * live_mask
            weight_total = float(row_weights.sum())
            if weight_total > 0.2:
                repaired[dead_ch] = (row_weights / weight_total) @ finite_window
        return repaired

    def _clip_extreme_spikes(
        self,
        working_window: np.ndarray,
        reference_raw_window: np.ndarray,
        muted_channel_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._spike_clip_z is not None
        chan_med, win_mad, eff_scale = self._compute_channel_mad_scale(reference_raw_window)
        abs_dev = np.abs(reference_raw_window - chan_med)
        if not bool(np.all(muted_channel_mask)):
            live_scales = np.maximum(eff_scale[~muted_channel_mask], 1e-300)[:, None]
            self._latest_diagnostics["max_spike_z"] = float(np.max(abs_dev[~muted_channel_mask] / live_scales))
        spike_hits = abs_dev > (self._spike_clip_z * eff_scale[:, None])
        spike_hits[muted_channel_mask] = False
        clipped_out = working_window
        if np.any(spike_hits):
            target_z = self._spike_clip_z if self._spike_clip_target_z is None else self._spike_clip_target_z
            bound_half = (target_z * eff_scale)[:, None]
            clipped_out = np.array(working_window, dtype=np.float64, copy=True)
            clipped_bounds = np.clip(clipped_out, chan_med - bound_half, chan_med + bound_half)
            clipped_out[spike_hits] = clipped_bounds[spike_hits]
        return clipped_out, win_mad

    def _update_channel_baselines(
        self,
        cur_lv: np.ndarray | None,
        win_mad: np.ndarray | None,
        muted_channel_mask: np.ndarray,
    ) -> None:
        clean_channels = ~muted_channel_mask
        if cur_lv is not None and self._log_rel_var_count is not None and self._log_rel_var_mean is not None:
            self._log_rel_var_count = self._log_rel_var_count + clean_channels
            self._log_rel_var_mean = self._log_rel_var_mean + np.where(
                clean_channels,
                (cur_lv - self._log_rel_var_mean) / np.maximum(self._log_rel_var_count, 1.0),
                0.0,
            )
        if win_mad is not None:
            if self._mad_scale_sum is None or self._mad_scale_count is None:
                self._mad_scale_sum = np.zeros(len(clean_channels), dtype=np.float64)
                self._mad_scale_count = np.zeros(len(clean_channels), dtype=np.float64)
            self._mad_scale_sum = self._mad_scale_sum + np.where(clean_channels, win_mad, 0.0)
            self._mad_scale_count = self._mad_scale_count + clean_channels

    def sanitize_and_gate_window(self, raw_window: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return (cleaned_window, dc_centered_window, is_bad_window) for one incoming (C, T) window."""
        ParameterTypeValidator.require_numpy_array(raw_window, "raw_window", expected_ndim=2)
        window_f64 = np.asarray(raw_window, dtype=np.float64)
        finite_entries = np.isfinite(window_f64)
        has_non_finite = not bool(np.all(finite_entries))

        if not bool(np.any(finite_entries)):
            zero_win = np.zeros_like(window_f64, dtype=np.float64)
            self._latest_diagnostics = {"dropout": True, "non_finite": True}
            return zero_win, zero_win.copy(), True

        sanitized_input = (
            np.nan_to_num(window_f64, nan=0.0, posinf=5000.0, neginf=-5000.0)
            if has_non_finite
            else window_f64
        )
        if bool(np.all(np.ptp(sanitized_input, axis=-1) == 0.0)):
            zero_win = np.zeros_like(window_f64, dtype=np.float64)
            self._latest_diagnostics = {"dropout": True, "non_finite": has_non_finite}
            return zero_win, zero_win.copy(), True

        self._step_counter += 1
        self._latest_diagnostics = {"non_finite": has_non_finite}
        self._detect_dead_channels(sanitized_input, allow_latch=not has_non_finite)
        assert self._dead_channel_mask is not None

        loose_mask = np.zeros(len(sanitized_input), dtype=bool)
        cur_lv: np.ndarray | None = None
        if self._loose_variance_ratio is not None and not has_non_finite:
            loose_mask, cur_lv = self._detect_loose_channels(sanitized_input)

        non_finite_channels = ~np.all(finite_entries, axis=-1)
        muted_mask = self._dead_channel_mask | loose_mask | non_finite_channels
        cleaned_window = np.array(sanitized_input, dtype=np.float64, copy=True)
        zero_mask = muted_mask
        if self._repair_dead_channels and np.any(self._dead_channel_mask) and self._interpolation_matrix is not None:
            cleaned_window = self._repair_dead_electrodes(sanitized_input)
            zero_mask = loose_mask | non_finite_channels
        if np.any(zero_mask):
            cleaned_window[zero_mask] = 0.0

        live_selector = None if not np.any(muted_mask) else ~muted_mask
        log_rms, mean_log_eig = self.compute_window_measures(cleaned_window, live_selector)
        z_amp = self._compute_welford_z("lrms", log_rms)
        z_ld = self._compute_welford_z("ld", mean_log_eig)
        self._latest_diagnostics["z_amp"] = z_amp
        self._latest_diagnostics["z_ld"] = z_ld
        is_bad_window = bool(has_non_finite or (z_amp > self._z_amplitude_threshold) or (z_ld > self._z_logdet_threshold))

        win_mad: np.ndarray | None = None
        if self._spike_clip_z is not None:
            cleaned_window, win_mad = self._clip_extreme_spikes(cleaned_window, sanitized_input, muted_mask)

        if not is_bad_window:
            self._append_welford_observation("lrms", log_rms)
            if np.isfinite(mean_log_eig):
                self._append_welford_observation("ld", mean_log_eig)
            self._update_channel_baselines(cur_lv, win_mad, muted_mask)

        dc_centered_window = cleaned_window - np.mean(cleaned_window, axis=-1, keepdims=True)
        return cleaned_window, dc_centered_window, is_bad_window
