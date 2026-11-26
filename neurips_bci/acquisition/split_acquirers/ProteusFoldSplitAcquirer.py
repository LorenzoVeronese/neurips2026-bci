"""Builder for the 16 PROTEUS cross-validation folds (LOSO, XS4, XS1) and official split33 warmup fold."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from neurips_bci.acquisition.split_acquirers.SplitAcquirer import SplitAcquirer
from neurips_bci.container.stream_containers.ValidationFoldStreamContainer import ValidationFoldStreamContainer
from neurips_bci.container.stream_containers.WindowedCorpusStreamContainer import WindowedCorpusStreamContainer
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class ProteusFoldSplitAcquirer(SplitAcquirer):
    """Constructs LOSO (0..9), XS4 (10..13), and XS1 (14..15) folds, adapting cleanly to smoke corpora."""

    def __init__(self, corpus_container: WindowedCorpusStreamContainer) -> None:
        super().__init__(
            corpus_container=corpus_container,
            acquirer_source=f"proteus_cv_splits:{corpus_container.corpus_identifier}",
        )

    @staticmethod
    def _short_participant_code(participant_identifier: str) -> str:
        return participant_identifier.rsplit("/", 1)[-1]

    @staticmethod
    def _session_sort_tuple(session_identifier: str) -> tuple[int, str]:
        tail_token = session_identifier.rsplit("/", 1)[-1]
        try:
            return (int(tail_token), tail_token)
        except ValueError:
            return (0, tail_token)

    def acquire(self) -> list[ValidationFoldStreamContainer]:
        participants = self._corpus_container.participant_ids
        sessions = self._corpus_container.session_ids
        unique_participants = sorted(set(participants.tolist()))
        if len(unique_participants) < 2:
            raise ValueError("ProteusFoldSplitAcquirer requires at least 2 distinct participants")

        sessions_by_participant: dict[str, list[str]] = {}
        for subj_key in unique_participants:
            subj_mask = participants == subj_key
            ordered_subj_sessions = sorted(
                set(sessions[subj_mask].tolist()),
                key=self._session_sort_tuple,
            )
            sessions_by_participant[subj_key] = ordered_subj_sessions

        folds_list: list[ValidationFoldStreamContainer] = []

        # 1. Leave-One-Subject-Out (LOSO: 0..9 on full 10-participant PROTEUS)
        for subj_key in unique_participants:
            test_mask = participants == subj_key
            train_mask = ~test_mask
            folds_list.append(
                ValidationFoldStreamContainer(
                    fold_index=len(folds_list),
                    fold_label=self._short_participant_code(subj_key),
                    fold_regime="loso",
                    training_indices=np.flatnonzero(train_mask),
                    testing_indices=np.flatnonzero(test_mask),
                )
            )

        # 2. Cross-Session Cumulative (XS4: 10..13 on full PROTEUS)
        for subj_key in unique_participants:
            subj_sessions = sessions_by_participant[subj_key]
            if len(subj_sessions) < 2:
                continue
            is_subj = participants == subj_key
            short_code = self._short_participant_code(subj_key)
            for rank_k, target_sess in enumerate(subj_sessions[1:], start=1):
                earlier_mask = is_subj & np.isin(sessions, subj_sessions[:rank_k])
                test_mask = is_subj & (sessions == target_sess)
                train_mask = (~is_subj) | earlier_mask
                sess_tail = target_sess.rsplit("/", 1)[-1]
                folds_list.append(
                    ValidationFoldStreamContainer(
                        fold_index=len(folds_list),
                        fold_label=f"{short_code}/{sess_tail}",
                        fold_regime="xsess4",
                        training_indices=np.flatnonzero(train_mask),
                        testing_indices=np.flatnonzero(test_mask),
                    )
                )

        # 3. Single-Known-Session Gap (XS1: 14..15 on full PROTEUS; adapts if max sessions == 2)
        three_plus_participants = [
            subj_key for subj_key in unique_participants if len(sessions_by_participant[subj_key]) >= 3
        ]
        xs1_candidates = (
            three_plus_participants
            if three_plus_participants
            else [subj_key for subj_key in unique_participants if len(sessions_by_participant[subj_key]) >= 2]
        )
        for subj_key in xs1_candidates:
            subj_sessions = sessions_by_participant[subj_key]
            first_sess = subj_sessions[0]
            final_sess = subj_sessions[-1]
            is_subj = participants == subj_key
            known_mask = is_subj & (sessions == first_sess)
            test_mask = is_subj & (sessions == final_sess)
            train_mask = (~is_subj) | known_mask
            short_code = self._short_participant_code(subj_key)
            final_tail = final_sess.rsplit("/", 1)[-1]
            folds_list.append(
                ValidationFoldStreamContainer(
                    fold_index=len(folds_list),
                    fold_label=f"{short_code}/{final_tail}*",
                    fold_regime="xsess1",
                    training_indices=np.flatnonzero(train_mask),
                    testing_indices=np.flatnonzero(test_mask),
                )
            )

        return folds_list

    def build_official_warmup_fold(
        self,
        split_random_state: int = 33,
        test_split_ratio: float = 0.2,
    ) -> ValidationFoldStreamContainer:
        """Build the subject-disjoint NeuralBench warmup split (random_state=33, test_ratio=0.2)."""
        validated_seed = ParameterTypeValidator.require_integer(split_random_state, "split_random_state", 0)
        validated_ratio = ParameterTypeValidator.require_float(
            test_split_ratio, "test_split_ratio", minimum_bound=0.0, maximum_bound=0.9, strict_lower=True
        )
        participants = self._corpus_container.participant_ids
        unique_participants = sorted(set(participants.tolist()))
        if len(unique_participants) < 2:
            raise ValueError("build_official_warmup_fold requires at least 2 distinct participants")

        test_count = max(1, int(round(len(unique_participants) * validated_ratio)))
        test_count = min(test_count, len(unique_participants) - 1)
        _, held_out_participants = train_test_split(
            unique_participants,
            test_size=test_count,
            random_state=validated_seed,
        )
        test_mask = np.isin(participants, held_out_participants)
        train_mask = ~test_mask
        return ValidationFoldStreamContainer(
            fold_index=0,
            fold_label=f"split{validated_seed}",
            fold_regime="warmup",
            training_indices=np.flatnonzero(train_mask),
            testing_indices=np.flatnonzero(test_mask),
        )
