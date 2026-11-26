"""Abstract base class for EEG corpus downloaders and window extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence
import mne
import numpy as np

from neurips_bci.acquisition.Acquirer import Acquirer
from neurips_bci.configuration.section_configurations.DatasetSectionConfiguration import DatasetSectionConfiguration
from neurips_bci.operator.matrix_operators.SubspaceMatrixOperator import SubspaceMatrixOperator
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class CorpusAcquirer(Acquirer, ABC):
    """Shared filtering, resampling, and 3D spherical-spline montage mapping for EEG corpora."""

    def __init__(
        self,
        dataset_configuration: DatasetSectionConfiguration,
        acquirer_source: str,
    ) -> None:
        if not isinstance(dataset_configuration, DatasetSectionConfiguration):
            raise TypeError(
                f"dataset_configuration must be DatasetSectionConfiguration, got {type(dataset_configuration).__name__}"
            )
        super().__init__(acquirer_source=acquirer_source)
        self._dataset_configuration = dataset_configuration

    @property
    def dataset_configuration(self) -> DatasetSectionConfiguration:
        return self._dataset_configuration

    @staticmethod
    def notch_frequencies_below_nyquist(
        sampling_rate_hz: float,
        mains_harmonics_hz: tuple[float, float] = (50.0, 60.0),
    ) -> list[float]:
        ParameterTypeValidator.require_float(sampling_rate_hz, "sampling_rate_hz", minimum_bound=1.0)
        nyquist_cap_hz = min(float(sampling_rate_hz) / 2.0, 301.0)
        harmonic_list: list[float] = []
        for base_hz in mains_harmonics_hz:
            harmonic_list.extend(np.arange(base_hz, nyquist_cap_hz, base_hz).tolist())
        return sorted(set(float(freq_hz) for freq_hz in harmonic_list if freq_hz < float(sampling_rate_hz) / 2.0))

    @classmethod
    def preprocess_continuous_recording_uv(
        cls,
        continuous_microvolts: np.ndarray,
        source_sampling_rate_hz: float,
        source_channel_labels: Sequence[str],
        target_channel_labels: Sequence[str],
        target_sampling_rate_hz: float = 120.0,
        low_cut_hz: float = 0.1,
        high_cut_hz: float = 75.0,
    ) -> np.ndarray:
        """Apply zero-phase 50/60 Hz notch, 0.1-75 Hz FIR bandpass, 120 Hz resample, and spline mapping."""
        ParameterTypeValidator.require_numpy_array(continuous_microvolts, "continuous_microvolts", expected_ndim=2)
        ParameterTypeValidator.require_float(source_sampling_rate_hz, "source_sampling_rate_hz", minimum_bound=1.0)
        ParameterTypeValidator.require_float(target_sampling_rate_hz, "target_sampling_rate_hz", minimum_bound=1.0)
        source_labels = [str(ch_str) for ch_str in source_channel_labels]
        target_labels = [str(ch_str) for ch_str in target_channel_labels]
        if continuous_microvolts.shape[0] != len(source_labels):
            raise ValueError(
                f"continuous_microvolts channel axis ({continuous_microvolts.shape[0]}) != {len(source_labels)}"
            )

        mne_info = mne.create_info(source_labels, float(source_sampling_rate_hz), ch_types="eeg")
        raw_recording = mne.io.RawArray(
            np.asarray(continuous_microvolts, dtype=np.float64) * 1e-6,
            mne_info,
            verbose=False,
        )
        notch_list = cls.notch_frequencies_below_nyquist(source_sampling_rate_hz)
        if notch_list:
            raw_recording.notch_filter(
                notch_list,
                picks="all",
                phase="zero",
                filter_length="auto",
                verbose=False,
            )
        effective_high_hz = high_cut_hz if high_cut_hz < float(source_sampling_rate_hz) / 2.0 else None
        raw_recording.filter(
            low_cut_hz,
            effective_high_hz,
            picks="all",
            skip_by_annotation=("edge", "bad_acq_skip"),
            verbose=False,
        )
        if abs(float(source_sampling_rate_hz) - float(target_sampling_rate_hz)) > 1e-6:
            raw_recording.resample(float(target_sampling_rate_hz), verbose=False)
        filtered_uv = raw_recording.get_data() * 1e6

        if source_labels == target_labels:
            return filtered_uv.astype(np.float32)
        source_lookup = {ch_str.lower(): idx for idx, ch_str in enumerate(source_labels)}
        if all(tgt_str.lower() in source_lookup for tgt_str in target_labels):
            perm_indices = [source_lookup[tgt_str.lower()] for tgt_str in target_labels]
            return filtered_uv[perm_indices].astype(np.float32)
        spline_matrix = SubspaceMatrixOperator.build_spherical_spline_mapping(source_labels, target_labels)
        return (spline_matrix @ filtered_uv).astype(np.float32)

    def resolve_target_proteus_channels(self) -> tuple[str, ...]:
        proteus_npz_path = self._dataset_configuration.extracted_cache_directory / "proteus_windows.npz"
        if proteus_npz_path.is_file():
            with np.load(proteus_npz_path, allow_pickle=False) as cached_npz:
                if "ch_names" in cached_npz:
                    return tuple(str(ch_str) for ch_str in cached_npz["ch_names"].tolist())
        return SubspaceMatrixOperator.PROTEUS_41_CHANNELS

    @abstractmethod
    def acquire(self) -> Path | list[Path]:
        """Download raw EEG recordings if needed, extract standardized windows, and write .npz cache(s)."""
