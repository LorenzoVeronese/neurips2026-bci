"""Single-session B=1 causal streaming container under chronological or shuffled window order."""

from __future__ import annotations

from typing import Iterator
import numpy as np

from neurips_bci.container.stream_containers.StreamContainer import StreamContainer
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class SessionWindowStreamContainer(StreamContainer):
    """Iterates one (participant, session) EEG stream strictly one window (1, C, T) at a time."""

    def __init__(
        self,
        session_identifier: str,
        participant_identifier: str,
        order_identifier: str,
        ordered_windows: np.ndarray,
        ordered_labels: np.ndarray,
        ordered_interfaces: np.ndarray,
    ) -> None:
        super().__init__(container_kind="single_session_causal_stream")
        self._session_identifier = ParameterTypeValidator.require_non_empty_string(
            session_identifier, "session_identifier"
        )
        self._participant_identifier = ParameterTypeValidator.require_non_empty_string(
            participant_identifier, "participant_identifier"
        )
        self._order_identifier = ParameterTypeValidator.require_non_empty_string(order_identifier, "order_identifier")
        self._ordered_windows = np.asarray(
            ParameterTypeValidator.require_numpy_array(ordered_windows, "ordered_windows", expected_ndim=3),
            dtype=np.float32,
        )
        total_len = self._ordered_windows.shape[0]
        self._ordered_labels = np.asarray(
            ParameterTypeValidator.require_numpy_array(ordered_labels, "ordered_labels", expected_shape=(total_len,)),
            dtype=np.int64,
        )
        self._ordered_interfaces = np.asarray(ordered_interfaces, dtype="U32")
        if len(self._ordered_interfaces) != total_len:
            raise ValueError("ordered_interfaces length must match ordered_windows")

    @property
    def session_identifier(self) -> str:
        return self._session_identifier

    @property
    def participant_identifier(self) -> str:
        return self._participant_identifier

    @property
    def order_identifier(self) -> str:
        return self._order_identifier

    @property
    def ordered_windows(self) -> np.ndarray:
        return self._ordered_windows

    @property
    def ordered_labels(self) -> np.ndarray:
        return self._ordered_labels

    @property
    def ordered_interfaces(self) -> np.ndarray:
        return self._ordered_interfaces

    def window_count(self) -> int:
        return int(self._ordered_windows.shape[0])

    def iter_causal_single_windows(self) -> Iterator[tuple[int, np.ndarray]]:
        for step_index in range(self.window_count()):
            yield step_index, self._ordered_windows[step_index : step_index + 1]
