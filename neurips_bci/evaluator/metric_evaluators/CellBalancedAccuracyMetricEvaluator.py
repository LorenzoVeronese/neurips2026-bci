"""Competition cell-mean balanced accuracy, 14/16-fold k05, jackknife, and sign-flip Evaluator."""

from __future__ import annotations

import itertools
from typing import Sequence
import numpy as np

from neurips_bci.evaluator.metric_evaluators.MetricEvaluator import MetricEvaluator
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class CellBalancedAccuracyMetricEvaluator(MetricEvaluator):
    """Computes Track 2 cell-mean balanced accuracy, k05 fold summaries, jackknife SE, and sign-flip p-values."""

    _STANDARD_PARTICIPANTS: tuple[str, ...] = tuple(f"P{idx:02d}" for idx in range(1, 11))
    _LOSO_INDICES: tuple[int, ...] = tuple(range(10))
    _XS4_INDICES: tuple[int, ...] = (10, 11, 12, 13)
    _XS1_PAIR_INDICES: tuple[int, ...] = (14, 15)
    _XS1_FOUR_INDICES: tuple[int, ...] = (10, 12, 14, 15)
    _SUMMARY_14_INDICES: tuple[int, ...] = tuple(range(14))
    _STANDARD_FOLD_PARTICIPANTS: tuple[str, ...] = (
        "P01",
        "P02",
        "P03",
        "P04",
        "P05",
        "P06",
        "P07",
        "P08",
        "P09",
        "P10",
        "P04",
        "P04",
        "P10",
        "P10",
        "P04",
        "P10",
    )
    _SIGN_FLIP_MATRIX: np.ndarray = np.array(
        list(itertools.product((1.0, -1.0), repeat=10)),
        dtype=np.float64,
    )

    def __init__(
        self,
        known_participant_weight: float = 0.5,
        true_labels: np.ndarray | None = None,
        pred_labels: np.ndarray | None = None,
        session_ids: np.ndarray | None = None,
        interface_contexts: np.ndarray | None = None,
    ) -> None:
        super().__init__(evaluation_protocol="cell_mean_balanced_accuracy_k05")
        self._known_participant_weight = ParameterTypeValidator.require_float(
            known_participant_weight,
            "known_participant_weight",
            minimum_bound=0.0,
            maximum_bound=1.0,
        )
        self._true_labels = None if true_labels is None else np.asarray(true_labels, dtype=np.int64).ravel()
        self._pred_labels = None if pred_labels is None else np.asarray(pred_labels, dtype=np.int64).ravel()
        self._session_ids = None if session_ids is None else np.asarray(session_ids, dtype="U64").ravel()
        self._interface_contexts = (
            None if interface_contexts is None else np.asarray(interface_contexts, dtype="U32").ravel()
        )
        provided_flags = (
            self._true_labels is not None,
            self._pred_labels is not None,
            self._session_ids is not None,
            self._interface_contexts is not None,
        )
        if any(provided_flags) and not all(provided_flags):
            raise ValueError(
                "true_labels, pred_labels, session_ids, and interface_contexts must all be provided together or omitted"
            )
        if all(provided_flags):
            assert self._true_labels is not None
            assert self._pred_labels is not None
            assert self._session_ids is not None
            assert self._interface_contexts is not None
            window_total = len(self._true_labels)
            if (
                len(self._pred_labels) != window_total
                or len(self._session_ids) != window_total
                or len(self._interface_contexts) != window_total
            ):
                raise ValueError("All window-level label and metadata arrays must have identical length")

    @property
    def known_participant_weight(self) -> float:
        return self._known_participant_weight

    def evaluate(self) -> dict[str, object]:
        if (
            self._true_labels is None
            or self._pred_labels is None
            or self._session_ids is None
            or self._interface_contexts is None
        ):
            return {
                "evaluation_protocol": self.evaluation_protocol,
                "known_participant_weight": self._known_participant_weight,
                "sign_flip_pattern_count": int(self._SIGN_FLIP_MATRIX.shape[0]),
            }
        return self.score_predictions(
            true_labels=self._true_labels,
            pred_labels=self._pred_labels,
            session_ids=self._session_ids,
            interface_contexts=self._interface_contexts,
        )

    @staticmethod
    def balanced_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """Exact competition balanced accuracy over union(true_labels, pred_labels) in [0.0, 1.0]."""
        true_flat = np.asarray(true_labels).ravel()
        pred_flat = np.asarray(pred_labels).ravel()
        if len(true_flat) != len(pred_flat):
            raise ValueError(
                f"true_labels length ({len(true_flat)}) must equal pred_labels length ({len(pred_flat)})"
            )
        union_classes = np.union1d(true_flat, pred_flat)
        if len(union_classes) == 0:
            return float("nan")
        per_class_recalls = [
            float(np.mean(pred_flat[true_flat == cls_id] == cls_id))
            if np.any(true_flat == cls_id)
            else 0.0
            for cls_id in union_classes
        ]
        return float(np.mean(per_class_recalls))

    @classmethod
    def cell_mean_balanced_accuracy(
        cls,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        session_ids: np.ndarray,
        interface_contexts: np.ndarray,
    ) -> float:
        """Unweighted mean of competition balanced_accuracy across distinct (session_id, interface) cells."""
        true_flat = np.asarray(true_labels).ravel()
        pred_flat = np.asarray(pred_labels).ravel()
        session_flat = np.asarray(session_ids, dtype=str).ravel()
        context_flat = np.asarray(interface_contexts, dtype=str).ravel()
        sample_total = len(true_flat)
        if sample_total == 0:
            return float("nan")
        if (
            len(pred_flat) != sample_total
            or len(session_flat) != sample_total
            or len(context_flat) != sample_total
        ):
            raise ValueError("true_labels, pred_labels, session_ids, and interface_contexts must match in length")
        composite_keys = np.char.add(np.char.add(session_flat, "|"), context_flat)
        distinct_cells = sorted(set(composite_keys.tolist()))
        cell_scores = [
            cls.balanced_accuracy(true_flat[composite_keys == cell_key], pred_flat[composite_keys == cell_key])
            for cell_key in distinct_cells
        ]
        return float(np.mean(cell_scores))

    def score_predictions(
        self,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        session_ids: np.ndarray,
        interface_contexts: np.ndarray,
    ) -> dict[str, object]:
        true_flat = np.asarray(true_labels).ravel()
        pred_flat = np.asarray(pred_labels).ravel()
        session_flat = np.asarray(session_ids, dtype=str).ravel()
        context_flat = np.asarray(interface_contexts, dtype=str).ravel()
        if len(true_flat) == 0:
            raise ValueError("Cannot score empty prediction arrays")
        if (
            len(pred_flat) != len(true_flat)
            or len(session_flat) != len(true_flat)
            or len(context_flat) != len(true_flat)
        ):
            raise ValueError("Prediction and metadata arrays must have equal length")

        composite_keys = np.char.add(np.char.add(session_flat, "|"), context_flat)
        distinct_cells = sorted(set(composite_keys.tolist()))
        per_cell_map: dict[str, float] = {}
        graz_scores: list[float] = []
        brainhero_scores: list[float] = []
        for cell_key in distinct_cells:
            cell_mask = composite_keys == cell_key
            cell_bacc = self.balanced_accuracy(true_flat[cell_mask], pred_flat[cell_mask])
            per_cell_map[cell_key] = cell_bacc
            if cell_key.endswith("|Graz"):
                graz_scores.append(cell_bacc)
            elif cell_key.endswith("|BrainHero"):
                brainhero_scores.append(cell_bacc)

        cell_mean = float(np.mean(list(per_cell_map.values())))
        pooled_bacc = self.balanced_accuracy(true_flat, pred_flat)
        raw_acc = float(np.mean(true_flat == pred_flat))
        return {
            "cell_mean_balanced_accuracy": cell_mean,
            "pooled_balanced_accuracy": pooled_bacc,
            "raw_accuracy": raw_acc,
            "graz_cell_mean": float(np.mean(graz_scores)) if graz_scores else float("nan"),
            "brainhero_cell_mean": float(np.mean(brainhero_scores)) if brainhero_scores else float("nan"),
            "cell_count": len(distinct_cells),
            "window_count": int(len(true_flat)),
            "per_cell_balanced_accuracy": per_cell_map,
        }

    @classmethod
    def summarize_fold_scores(
        cls,
        fold_scores: Sequence[float] | np.ndarray,
        fold_regimes: Sequence[str] | None = None,
        known_participant_weight: float = 0.5,
    ) -> dict[str, float]:
        """Compute LOSO, XS4, XS1, A+B, k05, woP04, and woP10 from 14/16-fold or regime-tagged scores."""
        weight_k = ParameterTypeValidator.require_float(
            known_participant_weight,
            "known_participant_weight",
            minimum_bound=0.0,
            maximum_bound=1.0,
        )
        scores_vec = np.asarray(fold_scores, dtype=np.float64).ravel()
        fold_total = len(scores_vec)
        if fold_total == 0:
            raise ValueError("fold_scores must be non-empty")

        if fold_regimes is None and fold_total in (14, 16):
            loso_mean = float(np.mean(scores_vec[list(cls._LOSO_INDICES)]))
            xs4_mean = float(np.mean(scores_vec[list(cls._XS4_INDICES)]))
            k05_score = 0.5 * loso_mean + 0.5 * xs4_mean
            kmix_score = weight_k * xs4_mean + (1.0 - weight_k) * loso_mean
            ab_mean = float(np.mean(scores_vec[list(cls._SUMMARY_14_INDICES)]))
            wo_p04 = cls._k_mixture_omitting_participant(scores_vec, "P04", weight_k=0.5)
            wo_p10 = cls._k_mixture_omitting_participant(scores_vec, "P10", weight_k=0.5)
            if fold_total == 16:
                xs1_pair_mean = float(np.mean(scores_vec[list(cls._XS1_PAIR_INDICES)]))
                xs1_four_mean = float(np.mean(scores_vec[list(cls._XS1_FOUR_INDICES)]))
            else:
                xs1_pair_mean = float("nan")
                xs1_four_mean = float("nan")
            return {
                "k05": k05_score,
                "k_mixture": kmix_score,
                "LOSO": loso_mean,
                "XS4": xs4_mean,
                "XS1": xs1_four_mean,
                "XS1_pair": xs1_pair_mean,
                "A_plus_B": ab_mean,
                "woP04": wo_p04,
                "woP10": wo_p10,
            }

        if fold_regimes is None:
            raise ValueError(
                "fold_regimes must be provided when fold_scores length is neither 14 nor 16"
            )
        if len(fold_regimes) != fold_total:
            raise ValueError("fold_regimes length must match fold_scores length")

        loso_rows = [scores_vec[idx] for idx, reg in enumerate(fold_regimes) if reg == "loso"]
        xs4_rows = [scores_vec[idx] for idx, reg in enumerate(fold_regimes) if reg in ("xsess4", "xsess")]
        xs1_rows = [scores_vec[idx] for idx, reg in enumerate(fold_regimes) if reg == "xsess1"]
        loso_mean = float(np.mean(loso_rows)) if loso_rows else float("nan")
        xs4_mean = float(np.mean(xs4_rows)) if xs4_rows else float("nan")
        xs1_mean = float(np.mean(xs1_rows)) if xs1_rows else float("nan")
        k05_score = (
            0.5 * loso_mean + 0.5 * xs4_mean
            if np.isfinite(loso_mean) and np.isfinite(xs4_mean)
            else float("nan")
        )
        kmix_score = (
            weight_k * xs4_mean + (1.0 - weight_k) * loso_mean
            if np.isfinite(loso_mean) and np.isfinite(xs4_mean)
            else float("nan")
        )
        combined_14 = loso_rows + xs4_rows
        return {
            "k05": k05_score,
            "k_mixture": kmix_score,
            "LOSO": loso_mean,
            "XS4": xs4_mean,
            "XS1": xs1_mean,
            "XS1_pair": xs1_mean,
            "A_plus_B": float(np.mean(combined_14)) if combined_14 else float("nan"),
            "woP04": float("nan"),
            "woP10": float("nan"),
        }

    @classmethod
    def _k_mixture_omitting_participant(
        cls,
        fold_scores: np.ndarray,
        omitted_participant: str | None = None,
        weight_k: float = 0.5,
    ) -> float:
        kept_loso = [
            fold_idx
            for fold_idx in cls._LOSO_INDICES
            if cls._STANDARD_FOLD_PARTICIPANTS[fold_idx] != omitted_participant
        ]
        kept_xs4 = [
            fold_idx
            for fold_idx in cls._XS4_INDICES
            if cls._STANDARD_FOLD_PARTICIPANTS[fold_idx] != omitted_participant
        ]
        return float(
            weight_k * float(np.mean(fold_scores[kept_xs4]))
            + (1.0 - weight_k) * float(np.mean(fold_scores[kept_loso]))
        )

    @classmethod
    def compute_participant_contributions(
        cls,
        fold_deltas: Sequence[float] | np.ndarray,
        known_participant_weight: float = 0.5,
    ) -> np.ndarray:
        """Return (10,) per-participant contributions whose sum equals k_mixture(fold_deltas)."""
        weight_k = ParameterTypeValidator.require_float(
            known_participant_weight,
            "known_participant_weight",
            minimum_bound=0.0,
            maximum_bound=1.0,
        )
        deltas_vec = np.asarray(fold_deltas, dtype=np.float64).ravel()
        if len(deltas_vec) < 14:
            raise ValueError(f"fold_deltas must contain at least the 14 summary folds, got {len(deltas_vec)}")
        contributions = np.zeros(len(cls._STANDARD_PARTICIPANTS), dtype=np.float64)
        for fold_idx in cls._LOSO_INDICES:
            subj_slot = cls._STANDARD_PARTICIPANTS.index(cls._STANDARD_FOLD_PARTICIPANTS[fold_idx])
            contributions[subj_slot] += (1.0 - weight_k) * deltas_vec[fold_idx] / len(cls._LOSO_INDICES)
        for fold_idx in cls._XS4_INDICES:
            subj_slot = cls._STANDARD_PARTICIPANTS.index(cls._STANDARD_FOLD_PARTICIPANTS[fold_idx])
            contributions[subj_slot] += weight_k * deltas_vec[fold_idx] / len(cls._XS4_INDICES)
        return contributions

    @classmethod
    def compute_participant_jackknife(
        cls,
        fold_deltas: Sequence[float] | np.ndarray,
        known_participant_weight: float = 0.5,
    ) -> dict[str, object]:
        """Compute 10-participant clustered delete-1 jackknife standard error, z-score, and omit estimates."""
        weight_k = ParameterTypeValidator.require_float(
            known_participant_weight,
            "known_participant_weight",
            minimum_bound=0.0,
            maximum_bound=1.0,
        )
        deltas_vec = np.asarray(fold_deltas, dtype=np.float64).ravel()
        if len(deltas_vec) < 14:
            raise ValueError(f"fold_deltas must contain at least 14 folds, got {len(deltas_vec)}")
        full_estimate = cls._k_mixture_omitting_participant(deltas_vec, None, weight_k=weight_k)
        omit_estimates = np.array(
            [
                cls._k_mixture_omitting_participant(deltas_vec, participant_code, weight_k=weight_k)
                for participant_code in cls._STANDARD_PARTICIPANTS
            ],
            dtype=np.float64,
        )
        cluster_count = len(omit_estimates)
        jackknife_se = float(
            np.sqrt(
                ((cluster_count - 1.0) / cluster_count)
                * np.sum((omit_estimates - np.mean(omit_estimates)) ** 2)
            )
        )
        z_score = float(full_estimate / jackknife_se) if jackknife_se > 0.0 else 0.0
        return {
            "full_k05_delta": full_estimate,
            "jackknife_se": jackknife_se,
            "z_score": z_score,
            "omit_one_estimates": {
                participant_code: float(omit_estimates[idx])
                for idx, participant_code in enumerate(cls._STANDARD_PARTICIPANTS)
            },
            "woP04": float(omit_estimates[cls._STANDARD_PARTICIPANTS.index("P04")]),
            "woP10": float(omit_estimates[cls._STANDARD_PARTICIPANTS.index("P10")]),
        }

    @classmethod
    def compute_participant_sign_flip_pvalue(
        cls,
        fold_deltas: Sequence[float] | np.ndarray,
        known_participant_weight: float = 0.5,
    ) -> float:
        """Exact two-sided p-value across all 2^10 = 1,024 participant-cluster sign-flip patterns."""
        contributions = cls.compute_participant_contributions(
            fold_deltas=fold_deltas,
            known_participant_weight=known_participant_weight,
        )
        observed_abs = abs(float(np.sum(contributions)))
        permuted_abs_totals = np.abs(cls._SIGN_FLIP_MATRIX @ contributions)
        return float(np.mean(permuted_abs_totals >= observed_abs - 1e-12))

    @classmethod
    def compare_paired_fold_scores(
        cls,
        candidate_fold_scores: Sequence[float] | np.ndarray,
        reference_fold_scores: Sequence[float] | np.ndarray,
        known_participant_weight: float = 0.5,
    ) -> dict[str, object]:
        """Paired statistical comparison of two 14/16-fold score vectors with jackknife z and sign-flip p."""
        cand_vec = np.asarray(candidate_fold_scores, dtype=np.float64).ravel()
        ref_vec = np.asarray(reference_fold_scores, dtype=np.float64).ravel()
        if len(cand_vec) != len(ref_vec) or len(cand_vec) < 14:
            raise ValueError("candidate_fold_scores and reference_fold_scores must have matching length >= 14")
        fold_deltas = cand_vec - ref_vec
        jackknife_report = cls.compute_participant_jackknife(
            fold_deltas=fold_deltas,
            known_participant_weight=known_participant_weight,
        )
        sign_flip_p = cls.compute_participant_sign_flip_pvalue(
            fold_deltas=fold_deltas,
            known_participant_weight=known_participant_weight,
        )
        contributions = cls.compute_participant_contributions(
            fold_deltas=fold_deltas,
            known_participant_weight=known_participant_weight,
        )
        positive_participants = int(np.sum(contributions > 1e-12))
        negative_participants = int(np.sum(contributions < -1e-12))
        return {
            "delta_k05": jackknife_report["full_k05_delta"],
            "delta_LOSO": float(np.mean(fold_deltas[list(cls._LOSO_INDICES)])),
            "delta_XS4": float(np.mean(fold_deltas[list(cls._XS4_INDICES)])),
            "jackknife_se": jackknife_report["jackknife_se"],
            "z_score": jackknife_report["z_score"],
            "sign_flip_p_value": sign_flip_p,
            "positive_participants": positive_participants,
            "negative_participants": negative_participants,
            "woP04_delta": jackknife_report["woP04"],
            "woP10_delta": jackknife_report["woP10"],
            "participant_contributions": {
                participant_code: float(contributions[idx])
                for idx, participant_code in enumerate(cls._STANDARD_PARTICIPANTS)
            },
        }
