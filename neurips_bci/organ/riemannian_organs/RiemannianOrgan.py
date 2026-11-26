"""Abstract base class for causal Riemannian and spectral-manifold decoding organs in Layer 7."""

from __future__ import annotations

from abc import ABC
import numpy as np
from scipy.signal import sosfiltfilt

from neurips_bci.operator.matrix_operators.CovarianceMatrixOperator import CovarianceMatrixOperator
from neurips_bci.operator.signal_operators.SpectralSignalOperator import SpectralSignalOperator
from neurips_bci.organ.Organ import Organ


class RiemannianOrgan(Organ, ABC):
    """Shared SPD covariance, Butterworth filter-bank, and causal balancing helpers for Riemannian organs."""

    def __init__(self, organ_identifier: str) -> None:
        super().__init__(organ_identifier=organ_identifier)

    @staticmethod
    def window_band_covariances(
        demeaned_window: np.ndarray,
        frequency_bands_hz: tuple[tuple[float, float], ...] | list[list[float]],
        sos_filters: list[np.ndarray],
        known_covariances: dict[tuple[float, float] | tuple[int, int] | None, np.ndarray] | None = None,
        include_broadband: bool = True,
        sampling_rate_hz: float = 120.0,
    ) -> np.ndarray:
        known_lookup = known_covariances if known_covariances is not None else {}
        band_tuples = [tuple(band_pair) for band_pair in frequency_bands_hz]
        slot_keys: list[tuple | None] = list(band_tuples) + ([None] if include_broadband else [])
        win_f64 = np.asarray(demeaned_window, dtype=np.float64)
        chan_count, sample_count = win_f64.shape
        cov_out = np.empty((len(slot_keys), chan_count, chan_count), dtype=np.float64)
        precomp_ops = (
            SpectralSignalOperator.precomputed_sos_operators(
                tuple((float(b[0]), float(b[1])) for b in band_tuples),
                sample_count=480,
                sampling_rate_hz=sampling_rate_hz,
            )
            if sample_count == 480 and len(band_tuples) > 0
            else None
        )
        for slot_idx, slot_key in enumerate(slot_keys):
            alt_int_key = (
                (int(slot_key[0]), int(slot_key[1]))
                if (slot_key is not None and float(slot_key[0]).is_integer() and float(slot_key[1]).is_integer())
                else None
            )
            if slot_key in known_lookup:
                cov_out[slot_idx] = known_lookup[slot_key]
            elif alt_int_key is not None and alt_int_key in known_lookup:
                cov_out[slot_idx] = known_lookup[alt_int_key]
            elif slot_key is None:
                cov_out[slot_idx] = CovarianceMatrixOperator.ledoit_wolf_covariance(win_f64[None])[0]
            elif precomp_ops is not None:
                filtered_win = win_f64 @ precomp_ops[slot_idx]
                cov_out[slot_idx] = CovarianceMatrixOperator.ledoit_wolf_covariance(filtered_win[None])[0]
            else:
                filtered_win = sosfiltfilt(sos_filters[slot_idx], win_f64[None], axis=-1)
                cov_out[slot_idx] = CovarianceMatrixOperator.ledoit_wolf_covariance(filtered_win)[0]
        return cov_out

    @staticmethod
    def balance_probability_step(
        raw_probs: np.ndarray,
        running_prob_sum: np.ndarray,
        step_index: int,
        prior_pseudo_count: float = 10.0,
    ) -> np.ndarray:
        class_count = len(raw_probs)
        marginal_est = (float(prior_pseudo_count) / float(class_count) + running_prob_sum) / (
            float(prior_pseudo_count) + float(step_index)
        )
        reweighted = raw_probs / np.maximum(marginal_est, 1e-12)
        return reweighted / np.sum(reweighted)
