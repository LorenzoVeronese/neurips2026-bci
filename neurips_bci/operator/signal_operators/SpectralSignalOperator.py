"""Spectral filtering, band-covariance, resampling, and non-motor feature operator."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, firwin, sosfiltfilt

from neurips_bci.operator.matrix_operators.CovarianceMatrixOperator import CovarianceMatrixOperator
from neurips_bci.operator.signal_operators.SignalOperator import SignalOperator
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class SpectralSignalOperator(SignalOperator):
    """Provides Butterworth/FIR linear filter matrices, multi-rate resamplers, and band covariances."""

    _SOS_MATRIX_CACHE: dict[tuple, np.ndarray] = {}
    _FIR_MATRIX_CACHE: dict[tuple, np.ndarray] = {}
    _RESAMPLE_200_CACHE: dict[tuple[int, int], np.ndarray] = {}

    def __init__(self, sampling_rate_hz: float = 120.0) -> None:
        super().__init__(operator_domain="electroencephalography_spectral_processing")
        self._sampling_rate_hz = ParameterTypeValidator.require_float(
            sampling_rate_hz, "sampling_rate_hz", minimum_bound=1.0
        )

    @property
    def sampling_rate_hz(self) -> float:
        return self._sampling_rate_hz

    def describe_operator(self) -> str:
        return f"Spectral EEG operator (sampling_rate_hz={self._sampling_rate_hz})"

    @staticmethod
    def build_butterworth_sos_bank(
        frequency_bands_hz: list[list[float]] | tuple[tuple[float, float], ...],
        sampling_rate_hz: float = 120.0,
        filter_order: int = 4,
    ) -> list[np.ndarray]:
        return [
            butter(filter_order, band_pair, btype="bandpass", fs=sampling_rate_hz, output="sos")
            for band_pair in frequency_bands_hz
        ]

    @classmethod
    def precomputed_sos_operators(
        cls,
        frequency_bands_hz: tuple[tuple[float, float], ...],
        sample_count: int = 480,
        sampling_rate_hz: float = 120.0,
        filter_order: int = 4,
    ) -> np.ndarray:
        cache_key = (tuple(tuple(map(float, b)) for b in frequency_bands_hz), int(sample_count), float(sampling_rate_hz), int(filter_order))
        cached_ops = cls._SOS_MATRIX_CACHE.get(cache_key)
        if cached_ops is not None:
            return cached_ops
        sos_list = cls.build_butterworth_sos_bank(list(frequency_bands_hz), sampling_rate_hz, filter_order)
        identity_time = np.eye(sample_count, dtype=np.float64)
        linear_ops = np.stack([sosfiltfilt(sos_mat, identity_time, axis=-1) for sos_mat in sos_list], axis=0)
        cls._SOS_MATRIX_CACHE[cache_key] = linear_ops
        return linear_ops

    @staticmethod
    def design_bandpass_fir(band_hz: tuple[float, float] | list[float], tap_count: int = 121, sampling_rate_hz: float = 120.0) -> np.ndarray:
        return firwin(tap_count, band_hz, pass_zero=False, fs=sampling_rate_hz)

    @classmethod
    def apply_zero_phase_fir(cls, signal_tensor: np.ndarray, fir_taps: np.ndarray) -> np.ndarray:
        signal_f64 = np.asarray(signal_tensor, dtype=np.float64)
        sample_count = signal_f64.shape[-1]
        tap_key = (sample_count, fir_taps.tobytes())
        linear_op = cls._FIR_MATRIX_CACHE.get(tap_key)
        if linear_op is None:
            identity_time = np.eye(sample_count, dtype=np.float64)
            linear_op = fftconvolve(
                identity_time, np.asarray(fir_taps, dtype=np.float64).reshape(1, -1), mode="same", axes=-1
            )
            cls._FIR_MATRIX_CACHE[tap_key] = linear_op
        return signal_f64 @ linear_op

    @classmethod
    def compute_band_covariances(
        cls,
        window_batch: np.ndarray,
        frequency_bands_hz: list[list[float]] | tuple[tuple[float, float], ...],
        sampling_rate_hz: float = 120.0,
        broadband_filtered_batch: np.ndarray | None = None,
        broadband_estimator: str = "lw",
        chunk_size: int = 512,
    ) -> np.ndarray:
        band_tuples = tuple((float(b[0]), float(b[1])) for b in frequency_bands_hz)
        batch_len, channel_count, sample_count = window_batch.shape
        if batch_len == 1 and sample_count == 480:
            single_window = np.asarray(window_batch[0], dtype=np.float64)
            linear_ops = cls.precomputed_sos_operators(band_tuples, 480, sampling_rate_hz)
            filtered_bands = np.matmul(single_window[None], linear_ops)
            single_output = np.empty((1, len(band_tuples) + 1, channel_count, channel_count), dtype=np.float64)
            single_output[0, :-1] = CovarianceMatrixOperator.ledoit_wolf_covariance(filtered_bands)
            last_input = (
                single_window[None]
                if broadband_filtered_batch is None
                else np.asarray(broadband_filtered_batch[:1], dtype=np.float64)
            )
            cov_fn = (
                CovarianceMatrixOperator.ledoit_wolf_covariance
                if broadband_estimator == "lw"
                else CovarianceMatrixOperator.sample_covariance
            )
            single_output[0, -1] = cov_fn(last_input)[0]
            return single_output

        sos_list = cls.build_butterworth_sos_bank(list(band_tuples), sampling_rate_hz)
        output_covs = np.empty((batch_len, len(band_tuples) + 1, channel_count, channel_count), dtype=np.float64)
        cov_fn = (
            CovarianceMatrixOperator.ledoit_wolf_covariance
            if broadband_estimator == "lw"
            else CovarianceMatrixOperator.sample_covariance
        )
        for chunk_start in range(0, batch_len, chunk_size):
            chunk_end = min(batch_len, chunk_start + chunk_size)
            chunk_x = np.asarray(window_batch[chunk_start:chunk_end], dtype=np.float64)
            for band_idx, sos_mat in enumerate(sos_list):
                output_covs[chunk_start:chunk_end, band_idx] = CovarianceMatrixOperator.ledoit_wolf_covariance(
                    sosfiltfilt(sos_mat, chunk_x, axis=-1)
                )
            last_chunk = chunk_x if broadband_filtered_batch is None else broadband_filtered_batch[chunk_start:chunk_end]
            output_covs[chunk_start:chunk_end, -1] = cov_fn(np.asarray(last_chunk, dtype=np.float64))
        return output_covs

    @classmethod
    def extract_non_motor_246_features(cls, window_matrix: np.ndarray) -> np.ndarray:
        centered_window = np.asarray(window_matrix, dtype=np.float64)
        centered_window = centered_window - centered_window.mean(axis=-1, keepdims=True)
        car_window = centered_window - centered_window.mean(axis=0, keepdims=True)
        raw_broadband_logvar = np.log(np.var(centered_window, axis=-1) + 1e-6)
        nm_bands = ((0.5, 4.0), (32.0, 48.0), (0.5, 48.0))
        if centered_window.shape[-1] == 480:
            nm_ops = cls.precomputed_sos_operators(nm_bands, 480, 120.0, 4)
            car_filtered = np.matmul(car_window[None], nm_ops)
            raw_filtered = np.matmul(centered_window[None], nm_ops[:2])
            car_logvars = np.log(np.var(car_filtered, axis=-1) + 1e-6).reshape(-1)
            raw_logvars = np.log(np.var(raw_filtered, axis=-1) + 1e-6).reshape(-1)
            feature_vec = np.concatenate([car_logvars, raw_logvars, raw_broadband_logvar], axis=0)
        else:
            sos_nm = cls.build_butterworth_sos_bank(list(nm_bands), 120.0, 4)
            car_list = [np.log(np.var(sosfiltfilt(s_mat, car_window, axis=-1), axis=-1) + 1e-6) for s_mat in sos_nm]
            raw_list = [np.log(np.var(sosfiltfilt(s_mat, centered_window, axis=-1), axis=-1) + 1e-6) for s_mat in sos_nm[:2]]
            feature_vec = np.concatenate(car_list + raw_list + [raw_broadband_logvar], axis=0)
        if not np.isfinite(feature_vec).all():
            feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=0.0, neginf=0.0)
        return feature_vec

    @staticmethod
    def mirror_rfft_resample(
        window_matrix: np.ndarray,
        target_samples: int = 1024,
        source_rate_hz: float = 120.0,
        lowpass_cutoff_hz: float | None = 52.0,
        mirror_pad: int = 30,
    ) -> np.ndarray:
        signal_f64 = np.asarray(window_matrix, dtype=np.float64)
        source_samples = signal_f64.shape[-1]
        padded_signal = np.pad(signal_f64, ((0, 0), (mirror_pad, mirror_pad)), mode="reflect")
        padded_len = padded_signal.shape[-1]
        freq_bins = np.fft.rfftfreq(padded_len, d=1.0 / source_rate_hz)
        spectrum = np.fft.rfft(padded_signal, axis=-1)
        if lowpass_cutoff_hz is not None:
            spectrum = spectrum * (freq_bins <= float(lowpass_cutoff_hz))[None, :]
        target_padded_len = int(round(padded_len * float(target_samples) / float(source_samples)))
        resampled_padded = np.fft.irfft(spectrum, n=target_padded_len, axis=-1) * (
            float(target_padded_len) / float(padded_len)
        )
        pad_out = int(round(mirror_pad * float(target_samples) / float(source_samples)))
        return resampled_padded[:, pad_out : pad_out + target_samples]

    @classmethod
    def linear_resample_matrix(cls, source_samples: int = 480, target_samples: int = 800) -> np.ndarray:
        key_pair = (int(source_samples), int(target_samples))
        cached_mat = cls._RESAMPLE_200_CACHE.get(key_pair)
        if cached_mat is not None:
            return cached_mat
        src_grid = np.linspace(0.0, 1.0, source_samples, endpoint=False, dtype=np.float64)
        tgt_grid = np.linspace(0.0, 1.0, target_samples, endpoint=False, dtype=np.float64)
        interp_mat = np.zeros((source_samples, target_samples), dtype=np.float64)
        scaled_pos = tgt_grid * float(source_samples)
        left_idx = np.clip(np.floor(scaled_pos).astype(int), 0, source_samples - 1)
        right_idx = np.clip(left_idx + 1, 0, source_samples - 1)
        frac = np.clip(scaled_pos - left_idx, 0.0, 1.0)
        for t_col in range(target_samples):
            interp_mat[left_idx[t_col], t_col] += 1.0 - frac[t_col]
            interp_mat[right_idx[t_col], t_col] += frac[t_col]
        cls._RESAMPLE_200_CACHE[key_pair] = interp_mat
        return interp_mat
