"""Validation fold container pairing a training corpus subset with test session streams."""

from __future__ import annotations

import numpy as np

from neurips_bci.container.stream_containers.SessionWindowStreamContainer import SessionWindowStreamContainer
from neurips_bci.container.stream_containers.StreamContainer import StreamContainer
from neurips_bci.container.stream_containers.WindowedCorpusStreamContainer import WindowedCorpusStreamContainer
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class ValidationFoldStreamContainer(StreamContainer):
    """Represents one cross-validation fold (LOSO, XS4, XS1, or Warmup) and builds ordered test streams."""

    def __init__(
        self,
        fold_index: int,
        fold_label: str,
        fold_regime: str,
        training_indices: np.ndarray,
        testing_indices: np.ndarray,
    ) -> None:
        super().__init__(container_kind="validation_fold_split")
        self._fold_index = ParameterTypeValidator.require_integer(fold_index, "fold_index", minimum_bound=0)
        self._fold_label = ParameterTypeValidator.require_non_empty_string(fold_label, "fold_label")
        self._fold_regime = ParameterTypeValidator.require_choice(
            fold_regime, ("loso", "xsess4", "xsess1", "warmup"), "fold_regime"
        )
        self._training_indices = np.asarray(training_indices, dtype=np.int64)
        self._testing_indices = np.asarray(testing_indices, dtype=np.int64)
        if len(self._training_indices) == 0 or len(self._testing_indices) == 0:
            raise ValueError("training_indices and testing_indices must be non-empty")
        if np.intersect1d(self._training_indices, self._testing_indices).size > 0:
            raise ValueError(f"Fold {fold_label} leaks indices between train and test")

    @property
    def fold_index(self) -> int:
        return self._fold_index

    @property
    def fold_label(self) -> str:
        return self._fold_label

    @property
    def fold_regime(self) -> str:
        return self._fold_regime

    @property
    def training_indices(self) -> np.ndarray:
        return self._training_indices

    @property
    def testing_indices(self) -> np.ndarray:
        return self._testing_indices

    def window_count(self) -> int:
        return int(len(self._testing_indices))

    def build_ordered_test_streams(
        self,
        full_corpus_container: WindowedCorpusStreamContainer,
        order_identifier: str = "O0",
    ) -> list[SessionWindowStreamContainer]:
        ParameterTypeValidator.require_choice(order_identifier, ("O0", "W0", "W1", "W2", "W3", "W4"), "order_identifier")
        test_corpus = full_corpus_container.slice_indices(self._testing_indices, f"{self._fold_label}_test")
        unique_sessions = sorted(set(test_corpus.session_ids.tolist()))
        session_streams: list[SessionWindowStreamContainer] = []
        for sess_rank, sess_key in enumerate(unique_sessions):
            sess_mask = np.flatnonzero(test_corpus.session_ids == sess_key)
            sess_len = len(sess_mask)
            if order_identifier == "O0":
                perm_order = np.arange(sess_len, dtype=np.int64)
            else:
                perm_seed_k = int(order_identifier[1:])
                rng_perm = np.random.default_rng([31, int(self._fold_index), perm_seed_k, int(sess_rank)])
                perm_order = rng_perm.permutation(sess_len)
            chosen_rows = sess_mask[perm_order]
            session_streams.append(
                SessionWindowStreamContainer(
                    session_identifier=sess_key,
                    participant_identifier=str(test_corpus.participant_ids[chosen_rows[0]]),
                    order_identifier=order_identifier,
                    ordered_windows=test_corpus.eeg_windows[chosen_rows],
                    ordered_labels=test_corpus.class_labels[chosen_rows],
                    ordered_interfaces=test_corpus.interface_contexts[chosen_rows],
                )
            )
        return session_streams
