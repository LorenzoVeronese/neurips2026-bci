"""Cognitive non-motor auxiliary corpora acquirer (TU Berlin Dataset C word generation & PhysioNet EEGMAT)."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shutil
import tempfile
import urllib.request
import zipfile
import mne
import numpy as np

from neurips_bci.acquisition.corpus_acquirers.CorpusAcquirer import CorpusAcquirer
from neurips_bci.configuration.section_configurations.DatasetSectionConfiguration import DatasetSectionConfiguration
from neurips_bci.container.stream_containers.WindowedCorpusStreamContainer import WindowedCorpusStreamContainer
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class CognitiveAuxiliaryCorpusAcquirer(CorpusAcquirer):
    """Downloads TU Berlin Dataset C (word generation) and PhysioNet EEGMAT (subtraction) and builds 41-ch caches."""

    SHIN2018_BASE_URL = "https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/EEG"
    ZYMA2019_ARCHIVE_URL = (
        "https://physionet.org/static/published-projects/eegmat/"
        "eeg-during-mental-arithmetic-tasks-1.0.0.zip"
    )

    _SHIN2018_RENAME: dict[str, str] = {
        "FP1": "Fp1",
        "FP2": "Fp2",
        "AFF5": "AFF5h",
        "AFF6": "AFF6h",
    }
    _SHIN2018_EOG: tuple[str, str] = ("HEOG", "VEOG")
    _ZYMA2019_SCALP: tuple[str, ...] = (
        "Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4",
        "C3", "C4", "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz",
    )
    _ZYMA2019_RENAME: dict[str, str] = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

    def __init__(self, dataset_configuration: DatasetSectionConfiguration) -> None:
        super().__init__(
            dataset_configuration=dataset_configuration,
            acquirer_source=f"{self.SHIN2018_BASE_URL}|{self.ZYMA2019_ARCHIVE_URL}",
        )

    def acquire(self) -> list[Path]:
        produced_caches: list[Path] = []
        if self._dataset_configuration.include_shin2018_corpus:
            produced_caches.append(self.acquire_shin2018_corpus())
        if self._dataset_configuration.include_zyma2019_corpus:
            produced_caches.append(self.acquire_zyma2019_corpus())
        return produced_caches

    def acquire_shin2018_corpus(self) -> Path:
        out_npz_path = self._dataset_configuration.extracted_cache_directory / "shin2018_windows.npz"
        if out_npz_path.is_file():
            WindowedCorpusStreamContainer.load_npz_archive(out_npz_path)
            return out_npz_path
        shin_raw_dir = self._ensure_shin2018_downloaded()
        shin_container = self.extract_shin2018_directory(shin_raw_dir)
        return shin_container.save_npz_archive(out_npz_path)

    def acquire_zyma2019_corpus(self) -> Path:
        out_npz_path = self._dataset_configuration.extracted_cache_directory / "zyma2019_windows.npz"
        if out_npz_path.is_file():
            WindowedCorpusStreamContainer.load_npz_archive(out_npz_path)
            return out_npz_path
        zyma_raw_dir = self._ensure_zyma2019_downloaded()
        zyma_container = self.extract_zyma2019_directory(zyma_raw_dir)
        return zyma_container.save_npz_archive(out_npz_path)

    def _ensure_shin2018_downloaded(self) -> Path:
        shin_root = self._dataset_configuration.raw_download_directory / "shin2018"
        if shin_root.is_dir() and list(shin_root.glob("vp*/word1.vhdr")):
            return shin_root
        shin_root.mkdir(parents=True, exist_ok=True)
        member_regex = re.compile(r"(?:^|.*/)word([123])\.(eeg|vhdr|vmrk)$", re.IGNORECASE)

        for participant_idx in range(1, 27):
            vp_folder = shin_root / f"vp{participant_idx:03d}"
            if all((vp_folder / f"word{ses_i}.vhdr").is_file() for ses_i in (1, 2, 3)):
                continue
            vp_folder.mkdir(parents=True, exist_ok=True)
            zip_url = f"{self.SHIN2018_BASE_URL}/VP{participant_idx:03d}.zip"
            tmp_zip = shin_root / f"VP{participant_idx:03d}.zip.part"
            urllib.request.urlretrieve(zip_url, tmp_zip)
            try:
                with zipfile.ZipFile(tmp_zip, "r") as zip_archive:
                    for member_info in zip_archive.infolist():
                        if member_info.is_dir():
                            continue
                        regex_hit = member_regex.match(member_info.filename)
                        if regex_hit is None:
                            continue
                        dest_file = vp_folder / f"word{regex_hit.group(1)}.{regex_hit.group(2).lower()}"
                        with zip_archive.open(member_info) as src_stream, dest_file.open("wb") as dst_stream:
                            shutil.copyfileobj(src_stream, dst_stream)
            finally:
                if tmp_zip.exists():
                    tmp_zip.unlink()
        return shin_root

    def _ensure_zyma2019_downloaded(self) -> Path:
        zyma_root = self._dataset_configuration.raw_download_directory / "zyma2019"
        if zyma_root.is_dir() and list(zyma_root.glob("Subject*_2.edf")):
            return zyma_root
        zyma_root.mkdir(parents=True, exist_ok=True)
        allowed_pattern = re.compile(
            r"^(Subject\d\d_[12]\.edf|README\.txt|RECORDS|SHA256SUMS\.txt|subject-info\.csv|LICENSE\.txt)$"
        )
        tmp_zip = zyma_root / "eegmat.zip.part"
        urllib.request.urlretrieve(self.ZYMA2019_ARCHIVE_URL, tmp_zip)
        try:
            with zipfile.ZipFile(tmp_zip, "r") as zip_archive:
                for member_info in zip_archive.infolist():
                    if member_info.is_dir():
                        continue
                    basename_str = member_info.filename.rsplit("/", 1)[-1]
                    if not allowed_pattern.match(basename_str):
                        continue
                    dest_file = zyma_root / basename_str
                    with zip_archive.open(member_info) as src_stream, dest_file.open("wb") as dst_stream:
                        shutil.copyfileobj(src_stream, dst_stream)
        finally:
            if tmp_zip.exists():
                tmp_zip.unlink()

        checksum_file = zyma_root / "SHA256SUMS.txt"
        if checksum_file.is_file():
            for line_text in checksum_file.read_text(encoding="utf-8").splitlines():
                parts = line_text.split()
                if len(parts) == 2 and (zyma_root / parts[1]).is_file():
                    computed_hex = hashlib.sha256((zyma_root / parts[1]).read_bytes()).hexdigest()
                    if computed_hex != parts[0]:
                        raise ValueError(f"SHA-256 mismatch for {parts[1]} in Zyma2019 archive")
        return zyma_root

    @staticmethod
    def _read_brainvision_recording(vhdr_path: Path) -> mne.io.BaseRaw:
        header_lines = vhdr_path.read_text(encoding="utf-8", errors="replace").replace("\r", "").splitlines()
        referenced_files = {
            header_key: next(
                line_str[len(header_key) + 1 :]
                for line_str in header_lines
                if line_str.startswith(header_key + "=")
            )
            for header_key in ("DataFile", "MarkerFile")
        }
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            for header_key, ext_suffix in (("DataFile", ".eeg"), ("MarkerFile", ".vmrk")):
                (temp_dir / referenced_files[header_key]).symlink_to(vhdr_path.with_suffix(ext_suffix).resolve())
            shutil.copy2(vhdr_path, temp_dir / vhdr_path.name)
            return mne.io.read_raw_brainvision(temp_dir / vhdr_path.name, preload=True, verbose=False)

    def extract_shin2018_directory(
        self,
        shin_root_directory: str | Path,
        target_channels: tuple[str, ...] | None = None,
    ) -> WindowedCorpusStreamContainer:
        resolved_root = ParameterTypeValidator.require_path(shin_root_directory, "shin_root_directory", must_exist=True)
        resolved_target_ch = self.resolve_target_proteus_channels() if target_channels is None else tuple(target_channels)
        target_sfreq = float(self._dataset_configuration.sampling_rate_hz)
        window_samples = int(self._dataset_configuration.window_sample_count)

        task_windows: list[np.ndarray] = []
        task_labels: list[int] = []
        participant_list: list[str] = []
        session_list: list[str] = []
        run_list: list[int] = []
        rest_windows: list[np.ndarray] = []
        rest_sessions: list[str] = []

        for vp_idx in range(1, 27):
            vp_dir = resolved_root / f"vp{vp_idx:03d}"
            if not vp_dir.is_dir():
                continue
            participant_id = f"Shin2018/S{vp_idx:02d}"
            for ses_idx in (1, 2, 3):
                vhdr_file = vp_dir / f"word{ses_idx}.vhdr"
                if not vhdr_file.is_file():
                    continue
                session_id = f"{participant_id}/{ses_idx}"
                raw_bv = self._read_brainvision_recording(vhdr_file)
                eog_present = [ch_str for ch_str in self._SHIN2018_EOG if ch_str in raw_bv.ch_names]
                if eog_present:
                    raw_bv.drop_channels(eog_present)
                renamed_src = [self._SHIN2018_RENAME.get(ch_str, ch_str) for ch_str in raw_bv.ch_names]
                mapped_uv = self.preprocess_continuous_recording_uv(
                    continuous_microvolts=raw_bv.get_data() * 1e6,
                    source_sampling_rate_hz=float(raw_bv.info["sfreq"]),
                    source_channel_labels=renamed_src,
                    target_channel_labels=resolved_target_ch,
                    target_sampling_rate_hz=target_sfreq,
                )
                for onset_sec, desc_str in zip(raw_bv.annotations.onset, raw_bv.annotations.description):
                    start_samp = int(round((float(onset_sec) + 1.0) * target_sfreq))
                    if start_samp < 0 or start_samp + window_samples > mapped_uv.shape[1]:
                        continue
                    win_arr = mapped_uv[:, start_samp : start_samp + window_samples].astype(np.float32)
                    if desc_str == "Stimulus/S 16":
                        task_windows.append(win_arr)
                        task_labels.append(2)
                        participant_list.append(participant_id)
                        session_list.append(session_id)
                        run_list.append(ses_idx)
                    elif desc_str == "Stimulus/S 32":
                        rest_windows.append(win_arr)
                        rest_sessions.append(session_id)

        if not task_windows:
            raise RuntimeError(f"No Shin2018 word generation windows extracted from {resolved_root}")

        return WindowedCorpusStreamContainer(
            corpus_identifier="Shin2018",
            eeg_windows=np.stack(task_windows, axis=0),
            class_labels=np.array(task_labels, dtype=np.int64),
            participant_ids=np.array(participant_list, dtype="U32"),
            session_ids=np.array(session_list, dtype="U64"),
            interface_contexts=np.full(len(task_windows), "ext", dtype="U32"),
            channel_labels=resolved_target_ch,
            sampling_rate_hz=target_sfreq,
            run_indices=np.array(run_list, dtype=np.int64),
            resting_windows=np.stack(rest_windows, axis=0) if rest_windows else None,
            resting_session_ids=np.array(rest_sessions, dtype="U64") if rest_sessions else None,
        )

    def extract_zyma2019_directory(
        self,
        zyma_root_directory: str | Path,
        target_channels: tuple[str, ...] | None = None,
    ) -> WindowedCorpusStreamContainer:
        resolved_root = ParameterTypeValidator.require_path(zyma_root_directory, "zyma_root_directory", must_exist=True)
        resolved_target_ch = self.resolve_target_proteus_channels() if target_channels is None else tuple(target_channels)
        target_sfreq = float(self._dataset_configuration.sampling_rate_hz)
        window_samples = int(self._dataset_configuration.window_sample_count)
        window_duration_s = float(window_samples) / target_sfreq
        renamed_scalp = [self._ZYMA2019_RENAME.get(ch_str, ch_str) for ch_str in self._ZYMA2019_SCALP]

        sub_windows: list[np.ndarray] = []
        sub_labels: list[int] = []
        participant_list: list[str] = []
        session_list: list[str] = []
        run_list: list[int] = []
        rest_windows: list[np.ndarray] = []
        rest_sessions: list[str] = []

        for subj_idx in range(36):
            rest_edf = resolved_root / f"Subject{subj_idx:02d}_1.edf"
            task_edf = resolved_root / f"Subject{subj_idx:02d}_2.edf"
            if not task_edf.is_file():
                continue
            participant_id = f"Zyma2019/S{subj_idx:02d}"
            session_id = f"{participant_id}/1"

            for part_num, edf_path, usable_sec, is_sub_task in (
                (1, rest_edf, 180.0, False),
                (2, task_edf, 60.0, True),
            ):
                if not edf_path.is_file():
                    continue
                raw_edf = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                src_sfreq = float(raw_edf.info["sfreq"])
                edf_picks = [
                    f"EEG {ch_str}" if f"EEG {ch_str}" in raw_edf.ch_names else ch_str
                    for ch_str in self._ZYMA2019_SCALP
                ]
                usable_samples = min(raw_edf.n_times, int(round(usable_sec * src_sfreq)))
                raw_uv = raw_edf.get_data(picks=edf_picks)[:, :usable_samples] * 1e6
                mapped_uv = self.preprocess_continuous_recording_uv(
                    continuous_microvolts=raw_uv,
                    source_sampling_rate_hz=src_sfreq,
                    source_channel_labels=renamed_scalp,
                    target_channel_labels=resolved_target_ch,
                    target_sampling_rate_hz=target_sfreq,
                )
                tile_count = int((usable_sec - 1.0) // window_duration_s)
                for tile_idx in range(tile_count):
                    onset_s = float(tile_idx) * window_duration_s
                    start_samp = int(round((onset_s + 1.0) * target_sfreq))
                    if start_samp + window_samples > mapped_uv.shape[1]:
                        continue
                    win_arr = mapped_uv[:, start_samp : start_samp + window_samples].astype(np.float32)
                    if is_sub_task:
                        sub_windows.append(win_arr)
                        sub_labels.append(1)
                        participant_list.append(participant_id)
                        session_list.append(session_id)
                        run_list.append(part_num)
                    else:
                        rest_windows.append(win_arr)
                        rest_sessions.append(session_id)

        if not sub_windows:
            raise RuntimeError(f"No Zyma2019 subtraction windows extracted from {resolved_root}")

        return WindowedCorpusStreamContainer(
            corpus_identifier="Zyma2019",
            eeg_windows=np.stack(sub_windows, axis=0),
            class_labels=np.array(sub_labels, dtype=np.int64),
            participant_ids=np.array(participant_list, dtype="U32"),
            session_ids=np.array(session_list, dtype="U64"),
            interface_contexts=np.full(len(sub_windows), "ext", dtype="U32"),
            channel_labels=resolved_target_ch,
            sampling_rate_hz=target_sfreq,
            run_indices=np.array(run_list, dtype=np.int64),
            resting_windows=np.stack(rest_windows, axis=0) if rest_windows else None,
            resting_session_ids=np.array(rest_sessions, dtype="U64") if rest_sessions else None,
        )
