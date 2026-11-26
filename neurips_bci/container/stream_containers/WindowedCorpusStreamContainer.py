"""Typed container for multi-session windowed EEG corpora with safe NumPy persistence."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from neurips_bci.container.stream_containers.StreamContainer import StreamContainer
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class WindowedCorpusStreamContainer(StreamContainer):
    """Holds (N, C, T) EEG windows, labels, participant/session/interface metadata, and rest segments."""

    def __init__(
        self,
        corpus_identifier: str,
        eeg_windows: np.ndarray,
        class_labels: np.ndarray,
        participant_ids: np.ndarray,
        session_ids: np.ndarray,
        interface_contexts: np.ndarray,
        channel_labels: tuple[str, ...],
        sampling_rate_hz: float = 120.0,
        run_indices: np.ndarray | None = None,
        resting_windows: np.ndarray | None = None,
        resting_session_ids: np.ndarray | None = None,
    ) -> None:
        super().__init__(container_kind="windowed_corpus")
        self._corpus_identifier = ParameterTypeValidator.require_non_empty_string(
            corpus_identifier, "corpus_identifier"
        )
        self._eeg_windows = np.asarray(
            ParameterTypeValidator.require_numpy_array(eeg_windows, "eeg_windows", expected_ndim=3),
            dtype=np.float32,
        )
        sample_count = self._eeg_windows.shape[0]
        if sample_count == 0:
            raise ValueError("eeg_windows must contain at least one window")
        self._class_labels = np.asarray(
            ParameterTypeValidator.require_numpy_array(class_labels, "class_labels", expected_shape=(sample_count,)),
            dtype=np.int64,
        )
        self._participant_ids = np.asarray(participant_ids, dtype="U32")
        self._session_ids = np.asarray(session_ids, dtype="U64")
        self._interface_contexts = np.asarray(interface_contexts, dtype="U32")
        if (
            len(self._participant_ids) != sample_count
            or len(self._session_ids) != sample_count
            or len(self._interface_contexts) != sample_count
        ):
            raise ValueError("Metadata arrays must match eeg_windows length")
        if len(channel_labels) != self._eeg_windows.shape[1]:
            raise ValueError(
                f"channel_labels length ({len(channel_labels)}) must equal C ({self._eeg_windows.shape[1]})"
            )
        self._channel_labels = tuple(str(ch_str) for ch_str in channel_labels)
        self._sampling_rate_hz = ParameterTypeValidator.require_float(
            sampling_rate_hz, "sampling_rate_hz", minimum_bound=1.0
        )
        self._run_indices = (
            np.zeros(sample_count, dtype=np.int64)
            if run_indices is None
            else np.asarray(run_indices, dtype=np.int64)
        )
        self._resting_windows = (
            None if resting_windows is None else np.asarray(resting_windows, dtype=np.float32)
        )
        self._resting_session_ids = (
            None if resting_session_ids is None else np.asarray(resting_session_ids, dtype="U64")
        )

    @property
    def corpus_identifier(self) -> str:
        return self._corpus_identifier

    @property
    def eeg_windows(self) -> np.ndarray:
        return self._eeg_windows

    @property
    def class_labels(self) -> np.ndarray:
        return self._class_labels

    @property
    def participant_ids(self) -> np.ndarray:
        return self._participant_ids

    @property
    def session_ids(self) -> np.ndarray:
        return self._session_ids

    @property
    def interface_contexts(self) -> np.ndarray:
        return self._interface_contexts

    @property
    def channel_labels(self) -> tuple[str, ...]:
        return self._channel_labels

    @property
    def sampling_rate_hz(self) -> float:
        return self._sampling_rate_hz

    @property
    def run_indices(self) -> np.ndarray:
        return self._run_indices

    @property
    def resting_windows(self) -> np.ndarray | None:
        return self._resting_windows

    @property
    def resting_session_ids(self) -> np.ndarray | None:
        return self._resting_session_ids

    def window_count(self) -> int:
        return int(self._eeg_windows.shape[0])

    def slice_indices(self, selected_indices: np.ndarray, subset_identifier: str) -> WindowedCorpusStreamContainer:
        idx_arr = np.asarray(selected_indices, dtype=np.int64)
        selected_sessions = set(self._session_ids[idx_arr].tolist())
        sliced_rest = None
        sliced_rest_sess = None
        if self._resting_windows is not None and self._resting_session_ids is not None:
            keep_rest = np.array([sess_key in selected_sessions for sess_key in self._resting_session_ids], dtype=bool)
            if np.any(keep_rest):
                sliced_rest = self._resting_windows[keep_rest]
                sliced_rest_sess = self._resting_session_ids[keep_rest]
        return WindowedCorpusStreamContainer(
            corpus_identifier=subset_identifier,
            eeg_windows=self._eeg_windows[idx_arr],
            class_labels=self._class_labels[idx_arr],
            participant_ids=self._participant_ids[idx_arr],
            session_ids=self._session_ids[idx_arr],
            interface_contexts=self._interface_contexts[idx_arr],
            channel_labels=self._channel_labels,
            sampling_rate_hz=self._sampling_rate_hz,
            run_indices=self._run_indices[idx_arr],
            resting_windows=sliced_rest,
            resting_session_ids=sliced_rest_sess,
        )

    def save_npz_archive(self, destination_npz_path: str | Path) -> Path:
        resolved_out = ParameterTypeValidator.require_path(destination_npz_path, "destination_npz_path")
        resolved_out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "corpus_identifier": np.array(self._corpus_identifier),
            "X": self._eeg_windows.astype(np.float32),
            "y": self._class_labels.astype(np.int64),
            "subjects": self._participant_ids.astype("U32"),
            "sessions": self._session_ids.astype("U64"),
            "contexts": self._interface_contexts.astype("U32"),
            "ch_names": np.array(self._channel_labels, dtype="U16"),
            "sfreq": np.array(self._sampling_rate_hz, dtype=np.float64),
            "runs": self._run_indices.astype(np.int64),
        }
        if self._resting_windows is not None and self._resting_session_ids is not None:
            payload["rest_X"] = self._resting_windows.astype(np.float32)
            payload["rest_sessions"] = self._resting_session_ids.astype("U64")
        np.savez_compressed(resolved_out, **payload)
        return resolved_out

    @classmethod
    def load_npz_archive(cls, source_npz_path: str | Path) -> WindowedCorpusStreamContainer:
        resolved_src = ParameterTypeValidator.require_path(source_npz_path, "source_npz_path", must_exist=True)
        with np.load(resolved_src, allow_pickle=False) as archive:
            corpus_id = str(archive["corpus_identifier"].item()) if "corpus_identifier" in archive else resolved_src.stem
            windows = archive["X"]
            labels = archive["y"]
            participants = archive["subjects"] if "subjects" in archive else archive["participants"]
            sessions = archive["sessions"]
            contexts = archive["contexts"] if "contexts" in archive else np.full(len(windows), "Graz", dtype="U32")
            channels = tuple(str(ch_str) for ch_str in archive["ch_names"].tolist())
            sfreq_hz = float(archive["sfreq"].item()) if "sfreq" in archive else 120.0
            runs = archive["runs"] if "runs" in archive else np.zeros(len(windows), dtype=np.int64)
            rest_x = archive["rest_X"] if "rest_X" in archive else None
            rest_s = archive["rest_sessions"] if "rest_sessions" in archive else None
        return cls(
            corpus_identifier=corpus_id,
            eeg_windows=windows,
            class_labels=labels,
            participant_ids=participants,
            session_ids=sessions,
            interface_contexts=contexts,
            channel_labels=channels,
            sampling_rate_hz=sfreq_hz,
            run_indices=runs,
            resting_windows=rest_x,
            resting_session_ids=rest_s,
        )
