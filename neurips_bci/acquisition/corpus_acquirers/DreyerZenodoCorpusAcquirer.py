"""Dreyer 2023 Zenodo/MOABB 87-participant motor imagery downloader and 41-channel spline extractor."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import mne
import numpy as np

from neurips_bci.acquisition.corpus_acquirers.CorpusAcquirer import CorpusAcquirer
from neurips_bci.configuration.section_configurations.DatasetSectionConfiguration import DatasetSectionConfiguration
from neurips_bci.container.stream_containers.WindowedCorpusStreamContainer import WindowedCorpusStreamContainer
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class DreyerZenodoCorpusAcquirer(CorpusAcquirer):
    """Downloads Zenodo record 8089820 via MOABB and extracts 41-channel spherical-spline mapped windows."""

    ZENODO_RECORD_URL = "https://zenodo.org/records/8089820"
    DREYER_27_CHANNELS: tuple[str, ...] = (
        "Fz", "FCz", "Cz", "CPz", "Pz",
        "C1", "C3", "C5", "C2", "C4", "C6",
        "F4", "FC2", "FC4", "FC6", "CP2", "CP4", "CP6", "P4",
        "F3", "FC1", "FC3", "FC5", "CP1", "CP3", "CP5", "P3",
    )
    _CUE_CODE_TO_CLASS: dict[int, int] = {769: 0, 770: 1}
    _BASELINE_START_CODE = 32775
    _BASELINE_STOP_CODE = 32776

    def __init__(self, dataset_configuration: DatasetSectionConfiguration) -> None:
        super().__init__(
            dataset_configuration=dataset_configuration,
            acquirer_source=self.ZENODO_RECORD_URL,
        )

    def acquire(self) -> Path:
        cache_npz_path = self._dataset_configuration.extracted_cache_directory / "dreyer2023_windows.npz"
        if cache_npz_path.is_file():
            WindowedCorpusStreamContainer.load_npz_archive(cache_npz_path)
            return cache_npz_path
        bids_root = self._ensure_dreyer2023_downloaded()
        corpus_container = self.extract_from_bids_directory(bids_root)
        return corpus_container.save_npz_archive(cache_npz_path)

    def _ensure_dreyer2023_downloaded(self) -> Path:
        mne_staging_dir = self._dataset_configuration.raw_download_directory / "Dreyer2023"
        candidate_roots = (
            mne_staging_dir / "MNE-dreyer2023-data",
            mne_staging_dir,
        )
        for root_dir in candidate_roots:
            if root_dir.is_dir() and list(root_dir.glob("sub-*")):
                return root_dir

        mne_staging_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MNE_DATA", str(mne_staging_dir))
        os.environ.setdefault("MNE_DATASETS_DREYER2023_PATH", str(mne_staging_dir))
        from moabb.datasets import Dreyer2023

        moabb_dataset = Dreyer2023()
        for subject_number in moabb_dataset.subject_list:
            moabb_dataset.data_path(subject_number, path=str(mne_staging_dir))

        for root_dir in candidate_roots:
            if root_dir.is_dir() and list(root_dir.glob("sub-*")):
                return root_dir
        raise FileNotFoundError(f"Dreyer2023 BIDS folders not found in {mne_staging_dir}")

    @staticmethod
    def _read_events_table(events_tsv_path: Path) -> list[dict[str, str]]:
        with events_tsv_path.open("r", encoding="utf-8-sig", newline="") as file_handle:
            reader = csv.DictReader(file_handle, delimiter="\t")
            return [dict(row_map) for row_map in reader]

    def extract_from_bids_directory(
        self,
        bids_root_directory: str | Path,
        target_channels: tuple[str, ...] | None = None,
    ) -> WindowedCorpusStreamContainer:
        resolved_root = ParameterTypeValidator.require_path(bids_root_directory, "bids_root_directory", must_exist=True)
        resolved_target_ch = self.resolve_target_proteus_channels() if target_channels is None else tuple(target_channels)
        target_sfreq = float(self._dataset_configuration.sampling_rate_hz)
        window_samples = int(self._dataset_configuration.window_sample_count)
        window_duration_s = float(window_samples) / target_sfreq

        ordered_records: list[tuple[tuple[int, int, float], np.ndarray, int, str, str, int]] = []
        rest_windows: list[np.ndarray] = []
        rest_sessions: list[str] = []

        for sub_idx in range(1, 88):
            eeg_dir = resolved_root / f"sub-{sub_idx:02d}" / "eeg"
            if not eeg_dir.is_dir():
                continue
            available_runs = [
                run_num
                for run_num in range(1, 7)
                if list(eeg_dir.glob(f"sub-{sub_idx:02d}_*task-R{run_num}*_eeg.edf"))
            ]
            if not available_runs:
                continue
            participant_id = f"Dreyer2023/S{sub_idx:02d}"
            session_id = f"{participant_id}/1"

            for run_num in available_runs:
                edf_file = sorted(eeg_dir.glob(f"sub-{sub_idx:02d}_*task-R{run_num}*_eeg.edf"))[0]
                events_file = Path(str(edf_file)[: -len("_eeg.edf")] + "_events.tsv")
                if not events_file.is_file():
                    continue
                raw_edf = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
                present_27 = [ch_str for ch_str in self.DREYER_27_CHANNELS if ch_str in raw_edf.ch_names]
                raw_edf.pick(present_27)
                raw_edf.reorder_channels(present_27)
                mapped_uv = self.preprocess_continuous_recording_uv(
                    continuous_microvolts=raw_edf.get_data() * 1e6,
                    source_sampling_rate_hz=float(raw_edf.info["sfreq"]),
                    source_channel_labels=present_27,
                    target_channel_labels=resolved_target_ch,
                    target_sampling_rate_hz=target_sfreq,
                )
                for ev_row in self._read_events_table(events_file):
                    try:
                        code_int = int(float(ev_row["trial_type"]))
                    except (KeyError, ValueError):
                        continue
                    if code_int not in self._CUE_CODE_TO_CLASS:
                        continue
                    onset_s = float(ev_row["onset"])
                    start_samp = int(round((onset_s + 1.0) * target_sfreq))
                    if 0 <= start_samp and start_samp + window_samples <= mapped_uv.shape[1]:
                        ordered_records.append(
                            (
                                (sub_idx, run_num, onset_s),
                                mapped_uv[:, start_samp : start_samp + window_samples].astype(np.float32),
                                self._CUE_CODE_TO_CLASS[code_int],
                                participant_id,
                                session_id,
                                run_num,
                            )
                        )

            oe_candidates = sorted(eeg_dir.glob("*OEbaseline_eeg.edf"))
            if oe_candidates:
                oe_edf = oe_candidates[0]
                oe_events_path = Path(str(oe_edf)[: -len("_eeg.edf")] + "_events.tsv")
                raw_oe = mne.io.read_raw_edf(oe_edf, preload=True, verbose=False)
                present_27 = [ch_str for ch_str in self.DREYER_27_CHANNELS if ch_str in raw_oe.ch_names]
                raw_oe.pick(present_27)
                raw_oe.reorder_channels(present_27)
                mapped_oe = self.preprocess_continuous_recording_uv(
                    continuous_microvolts=raw_oe.get_data() * 1e6,
                    source_sampling_rate_hz=float(raw_oe.info["sfreq"]),
                    source_channel_labels=present_27,
                    target_channel_labels=resolved_target_ch,
                    target_sampling_rate_hz=target_sfreq,
                )
                oe_rows = self._read_events_table(oe_events_path) if oe_events_path.is_file() else []
                starts = [
                    float(ev_row["onset"])
                    for ev_row in oe_rows
                    if str(ev_row.get("trial_type", "")).strip() == str(self._BASELINE_START_CODE)
                ]
                stops = [
                    float(ev_row["onset"])
                    for ev_row in oe_rows
                    if str(ev_row.get("trial_type", "")).strip() == str(self._BASELINE_STOP_CODE)
                ]
                cursor_s = (starts[0] + 1.0) if starts else 1.0
                stop_s = (stops[0] - 1.0) if stops else (float(mapped_oe.shape[1]) / target_sfreq)
                while cursor_s + window_duration_s <= stop_s:
                    start_samp = int(round(cursor_s * target_sfreq))
                    if 0 <= start_samp and start_samp + window_samples <= mapped_oe.shape[1]:
                        rest_windows.append(mapped_oe[:, start_samp : start_samp + window_samples].astype(np.float32))
                        rest_sessions.append(session_id)
                    cursor_s += window_duration_s

        if not ordered_records:
            raise RuntimeError(f"No Dreyer2023 windows extracted from {resolved_root}")

        ordered_records.sort(key=lambda rec_item: rec_item[0])
        return WindowedCorpusStreamContainer(
            corpus_identifier="Dreyer2023",
            eeg_windows=np.stack([rec_item[1] for rec_item in ordered_records], axis=0),
            class_labels=np.array([rec_item[2] for rec_item in ordered_records], dtype=np.int64),
            participant_ids=np.array([rec_item[3] for rec_item in ordered_records], dtype="U32"),
            session_ids=np.array([rec_item[4] for rec_item in ordered_records], dtype="U64"),
            interface_contexts=np.full(len(ordered_records), "Graz", dtype="U32"),
            channel_labels=resolved_target_ch,
            sampling_rate_hz=target_sfreq,
            run_indices=np.array([rec_item[5] for rec_item in ordered_records], dtype=np.int64),
            resting_windows=np.stack(rest_windows, axis=0) if rest_windows else None,
            resting_session_ids=np.array(rest_sessions, dtype="U64") if rest_sessions else None,
        )
