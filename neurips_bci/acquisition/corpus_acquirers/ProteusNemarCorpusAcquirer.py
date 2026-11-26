"""PROTEUS Track 2 NEMAR dataset downloader, BIDS window/rest extractor, and synthetic smoke generator."""

from __future__ import annotations

import csv
from pathlib import Path
import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

from neurips_bci.acquisition.corpus_acquirers.CorpusAcquirer import CorpusAcquirer
from neurips_bci.configuration.section_configurations.DatasetSectionConfiguration import DatasetSectionConfiguration
from neurips_bci.container.stream_containers.WindowedCorpusStreamContainer import WindowedCorpusStreamContainer
from neurips_bci.operator.matrix_operators.SubspaceMatrixOperator import SubspaceMatrixOperator
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class ProteusNemarCorpusAcquirer(CorpusAcquirer):
    """Acquires NEMAR nm000290 v1.0.0 BIDS recordings, extracts 2,655 cued windows + rest, and builds smoke caches."""

    _INTERFACE_BY_BIDS_TASK: dict[str, str] = {
        "AcquisitionBH": "BrainHero",
        "OnlineRawBH": "BrainHero",
        "AcquisitionGraz": "Graz",
        "OnlineRawGraz": "Graz",
    }
    _CLASS_BY_TRIAL_TYPE: dict[str, int] = {
        "mi": 0,
        "sub": 1,
        "word": 2,
    }
    _BASELINE_START_CODE = 32775
    _BASELINE_STOP_CODE = 32776

    def __init__(self, dataset_configuration: DatasetSectionConfiguration) -> None:
        source_tag = (
            f"nemar:{dataset_configuration.nemar_dataset_id}@{dataset_configuration.nemar_dataset_version}"
        )
        super().__init__(dataset_configuration=dataset_configuration, acquirer_source=source_tag)

    def acquire(self) -> Path:
        cache_npz_path = self._dataset_configuration.extracted_cache_directory / "proteus_windows.npz"
        if cache_npz_path.is_file():
            WindowedCorpusStreamContainer.load_npz_archive(cache_npz_path)
            return cache_npz_path
        bids_root = self._ensure_bids_dataset_present()
        corpus_container = self.extract_from_bids_directory(bids_root)
        return corpus_container.save_npz_archive(cache_npz_path)

    def acquire_corpus_container(self) -> WindowedCorpusStreamContainer:
        produced_path = self.acquire()
        return WindowedCorpusStreamContainer.load_npz_archive(produced_path)

    def _ensure_bids_dataset_present(self) -> Path:
        proteus_staging_root = self._dataset_configuration.raw_download_directory / "Dreyer2026Proteus"
        candidate_roots = (
            proteus_staging_root / "download" / self._dataset_configuration.nemar_dataset_id,
            proteus_staging_root / self._dataset_configuration.nemar_dataset_id,
            proteus_staging_root,
        )
        for candidate_dir in candidate_roots:
            if candidate_dir.is_dir() and list(candidate_dir.glob("sub-*")):
                return candidate_dir

        proteus_staging_root.mkdir(parents=True, exist_ok=True)
        download_dir = proteus_staging_root / "download"
        download_dir.mkdir(parents=True, exist_ok=True)
        try:
            import nemar

            nemar.download(
                self._dataset_configuration.nemar_dataset_id,
                data_dir=str(download_dir),
            )
        except Exception as nemar_exc:
            try:
                from neuralfetch.studies import Dreyer2026Proteus

                study_instance = Dreyer2026Proteus(path=str(proteus_staging_root))
                study_instance.download()
            except Exception as fetch_exc:
                raise RuntimeError(
                    f"Unable to download {self._dataset_configuration.nemar_dataset_id} into {proteus_staging_root}: "
                    f"nemar error={nemar_exc}; neuralfetch error={fetch_exc}"
                ) from fetch_exc

        for candidate_dir in candidate_roots:
            if candidate_dir.is_dir() and list(candidate_dir.glob("sub-*")):
                return candidate_dir
        raise FileNotFoundError(f"BIDS subject folders not found under {proteus_staging_root}")

    @staticmethod
    def _read_events_tsv(events_tsv_path: Path) -> list[dict[str, str]]:
        with events_tsv_path.open("r", encoding="utf-8-sig", newline="") as file_handle:
            reader = csv.DictReader(file_handle, delimiter="\t")
            return [dict(row_map) for row_map in reader]

    def extract_from_bids_directory(self, bids_root_directory: str | Path) -> WindowedCorpusStreamContainer:
        resolved_bids = ParameterTypeValidator.require_path(bids_root_directory, "bids_root_directory", must_exist=True)
        target_sfreq = float(self._dataset_configuration.sampling_rate_hz)
        window_samples = int(self._dataset_configuration.window_sample_count)
        window_duration_s = float(window_samples) / target_sfreq
        cue_offset_s = 1.0

        subject_dirs = sorted(sub_dir for sub_dir in resolved_bids.glob("sub-*") if sub_dir.is_dir())
        if not subject_dirs:
            raise ValueError(f"No sub-* directories found inside {resolved_bids}")

        canonical_channels: tuple[str, ...] | None = None
        task_records: list[tuple[tuple[str, int, int, float], np.ndarray, int, str, str, str, int]] = []
        rest_windows_list: list[np.ndarray] = []
        rest_sessions_list: list[str] = []

        for sub_dir in subject_dirs:
            participant_code = sub_dir.name.replace("sub-", "")
            participant_key = f"Dreyer2026Proteus/{participant_code}"
            session_dirs = sorted(
                (ses_dir for ses_dir in sub_dir.glob("ses-*") if ses_dir.is_dir()),
                key=lambda ses_path: int(ses_path.name.replace("ses-", "")),
            )
            for ses_dir in session_dirs:
                session_num_str = ses_dir.name.replace("ses-", "")
                session_num_int = int(session_num_str)
                session_key = f"{participant_key}/{session_num_str}"
                eeg_dir = ses_dir / "eeg"
                if not eeg_dir.is_dir():
                    continue

                for edf_path in sorted(eeg_dir.glob("*_eeg.edf")):
                    stem_prefix = edf_path.name[: -len("_eeg.edf")]
                    events_path = eeg_dir / f"{stem_prefix}_events.tsv"
                    if not events_path.is_file():
                        continue
                    task_token = ""
                    run_token = "0"
                    for entity_part in stem_prefix.split("_"):
                        if entity_part.startswith("task-"):
                            task_token = entity_part[len("task-") :]
                        elif entity_part.startswith("run-"):
                            run_token = entity_part[len("run-") :]

                    raw_edf = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                    eeg_picks = mne.pick_types(raw_edf.info, eeg=True, eog=False, emg=False, stim=False, exclude=[])
                    edf_channels = tuple(raw_edf.ch_names[idx] for idx in eeg_picks)
                    if canonical_channels is None:
                        canonical_channels = edf_channels
                    raw_uv = raw_edf.get_data(picks=eeg_picks) * 1e6
                    proc_uv = self.preprocess_continuous_recording_uv(
                        continuous_microvolts=raw_uv,
                        source_sampling_rate_hz=float(raw_edf.info["sfreq"]),
                        source_channel_labels=edf_channels,
                        target_channel_labels=canonical_channels,
                        target_sampling_rate_hz=target_sfreq,
                    )
                    event_rows = self._read_events_tsv(events_path)

                    if task_token in self._INTERFACE_BY_BIDS_TASK:
                        interface_label = self._INTERFACE_BY_BIDS_TASK[task_token]
                        run_num = int(run_token)
                        for ev_row in event_rows:
                            trial_kind = str(ev_row.get("trial_type", "")).strip()
                            if trial_kind not in self._CLASS_BY_TRIAL_TYPE:
                                continue
                            onset_sec = float(ev_row["onset"])
                            start_idx = int(round((onset_sec + cue_offset_s) * target_sfreq))
                            if start_idx < 0 or start_idx + window_samples > proc_uv.shape[1]:
                                continue
                            win_slice = proc_uv[:, start_idx : start_idx + window_samples].astype(np.float32)
                            sort_tuple = (participant_key, session_num_int, run_num, onset_sec)
                            task_records.append(
                                (
                                    sort_tuple,
                                    win_slice,
                                    self._CLASS_BY_TRIAL_TYPE[trial_kind],
                                    participant_key,
                                    session_key,
                                    interface_label,
                                    run_num,
                                )
                            )
                    elif task_token in ("BaselineOE", "BaselineCE"):
                        start_onsets = [
                            float(ev_row["onset"])
                            for ev_row in event_rows
                            if str(ev_row.get("value", "")).strip() == str(self._BASELINE_START_CODE)
                            or "BaselineStart" in str(ev_row.get("trial_type", ""))
                        ]
                        stop_onsets = [
                            float(ev_row["onset"])
                            for ev_row in event_rows
                            if str(ev_row.get("value", "")).strip() == str(self._BASELINE_STOP_CODE)
                            or "BaselineStop" in str(ev_row.get("trial_type", ""))
                        ]
                        cursor_s = (start_onsets[0] + 1.0) if start_onsets else 1.0
                        end_limit_s = (
                            (stop_onsets[0] - 1.0)
                            if stop_onsets
                            else (float(proc_uv.shape[1]) / target_sfreq - 1.0)
                        )
                        while cursor_s + window_duration_s <= end_limit_s:
                            start_idx = int(round(cursor_s * target_sfreq))
                            if 0 <= start_idx and start_idx + window_samples <= proc_uv.shape[1]:
                                rest_windows_list.append(
                                    proc_uv[:, start_idx : start_idx + window_samples].astype(np.float32)
                                )
                                rest_sessions_list.append(session_key)
                            cursor_s += window_duration_s

        if not task_records or canonical_channels is None:
            raise RuntimeError(f"No cued PROTEUS windows extracted from {resolved_bids}")

        task_records.sort(key=lambda rec_item: rec_item[0])
        stacked_windows = np.stack([rec_item[1] for rec_item in task_records], axis=0)
        stacked_labels = np.array([rec_item[2] for rec_item in task_records], dtype=np.int64)
        stacked_participants = np.array([rec_item[3] for rec_item in task_records], dtype="U32")
        stacked_sessions = np.array([rec_item[4] for rec_item in task_records], dtype="U64")
        stacked_interfaces = np.array([rec_item[5] for rec_item in task_records], dtype="U32")
        stacked_runs = np.array([rec_item[6] for rec_item in task_records], dtype=np.int64)
        stacked_rest = np.stack(rest_windows_list, axis=0) if rest_windows_list else None
        stacked_rest_sessions = np.array(rest_sessions_list, dtype="U64") if rest_sessions_list else None

        return WindowedCorpusStreamContainer(
            corpus_identifier="Dreyer2026Proteus",
            eeg_windows=stacked_windows,
            class_labels=stacked_labels,
            participant_ids=stacked_participants,
            session_ids=stacked_sessions,
            interface_contexts=stacked_interfaces,
            channel_labels=canonical_channels,
            sampling_rate_hz=target_sfreq,
            run_indices=stacked_runs,
            resting_windows=stacked_rest,
            resting_session_ids=stacked_rest_sessions,
        )

    @classmethod
    def generate_synthetic_smoke_corpus(
        cls,
        output_npz_path: str | Path,
        random_seed: int = 20261123,
    ) -> WindowedCorpusStreamContainer:
        """Create a realistic 4-participant (2 single-session, 2 multi-session, 144 windows + rest) EEG corpus."""
        resolved_out = ParameterTypeValidator.require_path(output_npz_path, "output_npz_path")
        validated_seed = ParameterTypeValidator.require_integer(random_seed, "random_seed", minimum_bound=0)
        rng = np.random.default_rng(validated_seed)

        channels = SubspaceMatrixOperator.PROTEUS_41_CHANNELS
        channel_count = len(channels)
        sample_count = 480
        sampling_rate_hz = 120.0

        participant_session_plan: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("Dreyer2026Proteus/P01", ("1",)),
            ("Dreyer2026Proteus/P02", ("1",)),
            ("Dreyer2026Proteus/P03", ("1", "2")),
            ("Dreyer2026Proteus/P04", ("1", "2")),
        )
        run_interface_schedule: tuple[tuple[int, str], ...] = (
            (1, "Graz"),
            (2, "BrainHero"),
            (3, "Graz"),
            (4, "Graz"),
            (5, "BrainHero"),
            (6, "BrainHero"),
        )

        sos_delta = butter(4, (1.0, 4.0), btype="bandpass", fs=sampling_rate_hz, output="sos")
        sos_mu = butter(4, (8.0, 13.0), btype="bandpass", fs=sampling_rate_hz, output="sos")
        sos_beta = butter(4, (14.0, 28.0), btype="bandpass", fs=sampling_rate_hz, output="sos")
        sos_gamma = butter(4, (32.0, 44.0), btype="bandpass", fs=sampling_rate_hz, output="sos")

        spatial_class_templates = rng.standard_normal((3, channel_count, 4)) * 0.45
        interface_templates = {
            "Graz": rng.standard_normal((channel_count, 3)) * 0.35,
            "BrainHero": rng.standard_normal((channel_count, 3)) * 0.55,
            "Rest": rng.standard_normal((channel_count, 3)) * 0.40,
        }

        windows_accum: list[np.ndarray] = []
        labels_accum: list[int] = []
        participants_accum: list[str] = []
        sessions_accum: list[str] = []
        interfaces_accum: list[str] = []
        runs_accum: list[int] = []
        rest_windows_accum: list[np.ndarray] = []
        rest_sessions_accum: list[str] = []

        for participant_id, session_numbers in participant_session_plan:
            subj_mixin = np.eye(channel_count, dtype=np.float64) + 0.12 * rng.standard_normal(
                (channel_count, channel_count)
            )
            for session_str in session_numbers:
                session_key = f"{participant_id}/{session_str}"
                session_gain = np.exp(0.18 * rng.standard_normal((channel_count, 1)))
                graz_labels = np.tile(np.array([0, 1, 2], dtype=np.int64), 4)
                bh_labels = np.tile(np.array([0, 1, 2], dtype=np.int64), 4)
                rng.shuffle(graz_labels)
                rng.shuffle(bh_labels)
                graz_cursor = 0
                bh_cursor = 0

                for run_idx, iface_label in run_interface_schedule:
                    run_drift = 1.0 + 0.08 * float(run_idx - 3)
                    for _ in range(4):
                        if iface_label == "Graz":
                            cls_id = int(graz_labels[graz_cursor])
                            graz_cursor += 1
                        else:
                            cls_id = int(bh_labels[bh_cursor])
                            bh_cursor += 1

                        lat_cls = rng.standard_normal((4, sample_count))
                        lat_iface = rng.standard_normal((3, sample_count))
                        bg_noise = rng.standard_normal((channel_count, sample_count)) * 8.0

                        cls_osc = (
                            sosfiltfilt(sos_mu, spatial_class_templates[0] @ lat_cls, axis=-1) * (22.0 if cls_id == 0 else 10.0)
                            + sosfiltfilt(sos_beta, spatial_class_templates[1] @ lat_cls, axis=-1) * (22.0 if cls_id == 1 else 10.0)
                            + sosfiltfilt(sos_gamma, spatial_class_templates[2] @ lat_cls, axis=-1) * (20.0 if cls_id == 2 else 9.0)
                        )
                        iface_osc = sosfiltfilt(
                            sos_delta if iface_label == "Graz" else sos_gamma,
                            interface_templates[iface_label] @ lat_iface,
                            axis=-1,
                        ) * (14.0 * run_drift)
                        synth_win = session_gain * (subj_mixin @ (cls_osc + iface_osc + bg_noise))
                        windows_accum.append(synth_win.astype(np.float32))
                        labels_accum.append(cls_id)
                        participants_accum.append(participant_id)
                        sessions_accum.append(session_key)
                        interfaces_accum.append(iface_label)
                        runs_accum.append(run_idx)

                for _ in range(4):
                    lat_rest = rng.standard_normal((3, sample_count))
                    rest_bg = rng.standard_normal((channel_count, sample_count)) * 9.0
                    rest_alpha = sosfiltfilt(sos_mu, interface_templates["Rest"] @ lat_rest, axis=-1) * 18.0
                    rest_win = session_gain * (subj_mixin @ (rest_alpha + rest_bg))
                    rest_windows_accum.append(rest_win.astype(np.float32))
                    rest_sessions_accum.append(session_key)

        smoke_container = WindowedCorpusStreamContainer(
            corpus_identifier="Dreyer2026Proteus_Smoke",
            eeg_windows=np.stack(windows_accum, axis=0),
            class_labels=np.array(labels_accum, dtype=np.int64),
            participant_ids=np.array(participants_accum, dtype="U32"),
            session_ids=np.array(sessions_accum, dtype="U64"),
            interface_contexts=np.array(interfaces_accum, dtype="U32"),
            channel_labels=channels,
            sampling_rate_hz=sampling_rate_hz,
            run_indices=np.array(runs_accum, dtype=np.int64),
            resting_windows=np.stack(rest_windows_accum, axis=0),
            resting_session_ids=np.array(rest_sessions_accum, dtype="U64"),
        )
        smoke_container.save_npz_archive(resolved_out)
        return smoke_container
