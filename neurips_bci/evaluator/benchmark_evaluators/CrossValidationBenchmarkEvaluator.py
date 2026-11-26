"""Cross-validation streaming BenchmarkEvaluator across folds and window orders (O0, W0..W4)."""

from __future__ import annotations

import copy
import csv
import inspect
import json
from pathlib import Path
import time
from typing import Callable, Sequence
import numpy as np

from neurips_bci.container.stream_containers.ValidationFoldStreamContainer import (
    ValidationFoldStreamContainer,
)
from neurips_bci.container.stream_containers.WindowedCorpusStreamContainer import (
    WindowedCorpusStreamContainer,
)
from neurips_bci.decoder.Decoder import Decoder
from neurips_bci.evaluator.benchmark_evaluators.BenchmarkEvaluator import BenchmarkEvaluator
from neurips_bci.evaluator.metric_evaluators.CellBalancedAccuracyMetricEvaluator import (
    CellBalancedAccuracyMetricEvaluator,
)
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class CrossValidationBenchmarkEvaluator(BenchmarkEvaluator):
    """Evaluates a Decoder factory across validation folds and causal B=1 session streams."""

    _SHUFFLE_FIVE_ORDERS: tuple[str, ...] = ("W0", "W1", "W2", "W3", "W4")

    def __init__(
        self,
        corpus_container: WindowedCorpusStreamContainer,
        fold_containers: Sequence[ValidationFoldStreamContainer],
        decoder_builder: (
            Callable[[WindowedCorpusStreamContainer, ValidationFoldStreamContainer], Decoder]
            | Callable[[WindowedCorpusStreamContainer], Decoder]
            | Decoder
        ),
        evaluation_orders: Sequence[str] = ("O0", "W0", "W1", "W2", "W3", "W4"),
        output_directory: str | Path | None = None,
        known_participant_weight: float = 0.5,
        reference_fold_scores_by_order: dict[str, Sequence[float]] | None = None,
    ) -> None:
        resolved_out_dir = (
            None
            if output_directory is None
            else ParameterTypeValidator.require_path(output_directory, "output_directory")
        )
        super().__init__(evaluation_protocol="streaming_cross_validation", output_directory=resolved_out_dir)
        if not isinstance(corpus_container, WindowedCorpusStreamContainer):
            raise TypeError("corpus_container must be WindowedCorpusStreamContainer")
        if not fold_containers:
            raise ValueError("fold_containers must contain at least one ValidationFoldStreamContainer")
        for fold_item in fold_containers:
            if not isinstance(fold_item, ValidationFoldStreamContainer):
                raise TypeError("Every element of fold_containers must be ValidationFoldStreamContainer")
        if not isinstance(decoder_builder, Decoder) and not callable(decoder_builder):
            raise TypeError("decoder_builder must be a Decoder instance or a callable returning a Decoder")
        if not evaluation_orders:
            raise ValueError("evaluation_orders must be non-empty")
        validated_orders: list[str] = []
        for order_tag in evaluation_orders:
            validated_orders.append(
                ParameterTypeValidator.require_choice(
                    order_tag, ("O0", "W0", "W1", "W2", "W3", "W4"), "evaluation_orders"
                )
            )
        self._corpus_container = corpus_container
        self._fold_containers = tuple(fold_containers)
        self._decoder_builder = decoder_builder
        self._evaluation_orders = tuple(validated_orders)
        self._known_participant_weight = ParameterTypeValidator.require_float(
            known_participant_weight,
            "known_participant_weight",
            minimum_bound=0.0,
            maximum_bound=1.0,
        )
        self._reference_fold_scores_by_order = (
            None
            if reference_fold_scores_by_order is None
            else {str(ord_key): tuple(float(sc) for sc in sc_seq) for ord_key, sc_seq in reference_fold_scores_by_order.items()}
        )

    @property
    def evaluation_orders(self) -> tuple[str, ...]:
        return self._evaluation_orders

    @property
    def fold_containers(self) -> tuple[ValidationFoldStreamContainer, ...]:
        return self._fold_containers

    def _instantiate_fold_decoder(
        self,
        training_corpus: WindowedCorpusStreamContainer,
        fold_container: ValidationFoldStreamContainer,
    ) -> Decoder:
        if isinstance(self._decoder_builder, Decoder):
            return copy.deepcopy(self._decoder_builder)
        signature_params = list(inspect.signature(self._decoder_builder).parameters.values())
        positional_capable = [
            param_spec
            for param_spec in signature_params
            if param_spec.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(
            param_spec.kind == inspect.Parameter.VAR_POSITIONAL for param_spec in signature_params
        )
        if has_varargs or len(positional_capable) >= 2:
            built_decoder = self._decoder_builder(training_corpus, fold_container)  # type: ignore[call-arg]
        else:
            built_decoder = self._decoder_builder(training_corpus)  # type: ignore[call-arg]
        if not isinstance(built_decoder, Decoder):
            raise TypeError(f"decoder_builder must return a Decoder, got {type(built_decoder).__name__}")
        return built_decoder

    def evaluate(self) -> dict[str, object]:
        metric_evaluator = CellBalancedAccuracyMetricEvaluator(
            known_participant_weight=self._known_participant_weight
        )
        per_order_fold_scores: dict[str, list[float]] = {
            order_tag: [] for order_tag in self._evaluation_orders
        }
        fold_regimes: list[str] = [fold_item.fold_regime for fold_item in self._fold_containers]
        csv_row_records: list[dict[str, object]] = []
        total_stream_seconds = 0.0
        total_windows_streamed = 0

        for fold_item in self._fold_containers:
            training_subset = self._corpus_container.slice_indices(
                fold_item.training_indices, f"{fold_item.fold_label}_train"
            )
            fit_start_clock = time.perf_counter()
            fold_decoder = self._instantiate_fold_decoder(training_subset, fold_item)
            fit_duration_s = time.perf_counter() - fit_start_clock

            for order_tag in self._evaluation_orders:
                session_streams = fold_item.build_ordered_test_streams(
                    self._corpus_container, order_identifier=order_tag
                )
                true_label_buffer: list[int] = []
                pred_label_buffer: list[int] = []
                session_id_buffer: list[str] = []
                interface_buffer: list[str] = []

                stream_start_clock = time.perf_counter()
                for session_stream in session_streams:
                    session_decoder = copy.deepcopy(fold_decoder)
                    session_decoder.reset_state()
                    for step_idx, single_window_batch in session_stream.iter_causal_single_windows():
                        if single_window_batch.shape[0] != 1:
                            raise RuntimeError(
                                f"Strict B=1 violated at step {step_idx}: batch shape {single_window_batch.shape}"
                            )
                        predicted_batch = np.asarray(session_decoder.predict(single_window_batch)).ravel()
                        if len(predicted_batch) != 1:
                            raise ValueError(
                                f"Decoder.predict must return 1 label for B=1, got shape {predicted_batch.shape}"
                            )
                        predicted_class = int(predicted_batch[0])
                        true_label_buffer.append(int(session_stream.ordered_labels[step_idx]))
                        pred_label_buffer.append(predicted_class)
                        session_id_buffer.append(session_stream.session_identifier)
                        interface_buffer.append(str(session_stream.ordered_interfaces[step_idx]))
                stream_duration_s = time.perf_counter() - stream_start_clock

                total_stream_seconds += stream_duration_s
                total_windows_streamed += len(true_label_buffer)

                scored_fold = metric_evaluator.score_predictions(
                    true_labels=np.asarray(true_label_buffer, dtype=np.int64),
                    pred_labels=np.asarray(pred_label_buffer, dtype=np.int64),
                    session_ids=np.asarray(session_id_buffer, dtype="U64"),
                    interface_contexts=np.asarray(interface_buffer, dtype="U32"),
                )
                cell_bacc = float(scored_fold["cell_mean_balanced_accuracy"])  # type: ignore[arg-type]
                per_order_fold_scores[order_tag].append(cell_bacc)
                ms_per_window = (
                    1000.0 * stream_duration_s / len(true_label_buffer) if true_label_buffer else 0.0
                )
                csv_row_records.append(
                    {
                        "fold_index": fold_item.fold_index,
                        "fold_label": fold_item.fold_label,
                        "fold_regime": fold_item.fold_regime,
                        "order_identifier": order_tag,
                        "cell_mean_balanced_accuracy": cell_bacc,
                        "cell_mean_balanced_accuracy_pct": 100.0 * cell_bacc,
                        "pooled_balanced_accuracy": float(scored_fold["pooled_balanced_accuracy"]),  # type: ignore[arg-type]
                        "raw_accuracy": float(scored_fold["raw_accuracy"]),  # type: ignore[arg-type]
                        "cell_count": int(scored_fold["cell_count"]),  # type: ignore[arg-type]
                        "window_count": int(scored_fold["window_count"]),  # type: ignore[arg-type]
                        "fit_seconds": fit_duration_s,
                        "stream_seconds": stream_duration_s,
                        "milliseconds_per_window": ms_per_window,
                    }
                )

        order_summaries: dict[str, dict[str, object]] = {}
        regime_arg: Sequence[str] | None = (
            None if len(self._fold_containers) in (14, 16) else fold_regimes
        )
        for order_tag, order_scores in per_order_fold_scores.items():
            agg_metrics = metric_evaluator.summarize_fold_scores(
                fold_scores=order_scores,
                fold_regimes=regime_arg,
                known_participant_weight=self._known_participant_weight,
            )
            order_entry: dict[str, object] = {
                "per_fold_scores": list(order_scores),
                "per_fold_scores_pct": [100.0 * sc for sc in order_scores],
                "summary": agg_metrics,
                "summary_pct": {
                    metric_key: (100.0 * metric_val if np.isfinite(metric_val) else float("nan"))
                    for metric_key, metric_val in agg_metrics.items()
                },
            }
            if (
                self._reference_fold_scores_by_order is not None
                and order_tag in self._reference_fold_scores_by_order
                and len(order_scores) >= 14
            ):
                order_entry["vs_reference"] = metric_evaluator.compare_paired_fold_scores(
                    candidate_fold_scores=order_scores,
                    reference_fold_scores=self._reference_fold_scores_by_order[order_tag],
                    known_participant_weight=self._known_participant_weight,
                )
            order_summaries[order_tag] = order_entry

        if all(w_tag in per_order_fold_scores for w_tag in self._SHUFFLE_FIVE_ORDERS):
            ow5_fold_scores = np.mean(
                np.stack([per_order_fold_scores[w_tag] for w_tag in self._SHUFFLE_FIVE_ORDERS], axis=0),
                axis=0,
            ).tolist()
            ow5_agg = metric_evaluator.summarize_fold_scores(
                fold_scores=ow5_fold_scores,
                fold_regimes=regime_arg,
                known_participant_weight=self._known_participant_weight,
            )
            order_summaries["OW5"] = {
                "per_fold_scores": ow5_fold_scores,
                "per_fold_scores_pct": [100.0 * sc for sc in ow5_fold_scores],
                "summary": ow5_agg,
                "summary_pct": {
                    metric_key: (100.0 * metric_val if np.isfinite(metric_val) else float("nan"))
                    for metric_key, metric_val in ow5_agg.items()
                },
            }

        benchmark_report: dict[str, object] = {
            "evaluation_protocol": self.evaluation_protocol,
            "corpus_identifier": self._corpus_container.corpus_identifier,
            "fold_count": len(self._fold_containers),
            "evaluation_orders": list(self._evaluation_orders),
            "total_windows_streamed": total_windows_streamed,
            "total_stream_seconds": total_stream_seconds,
            "mean_milliseconds_per_window": (
                1000.0 * total_stream_seconds / total_windows_streamed
                if total_windows_streamed > 0
                else 0.0
            ),
            "order_summaries": order_summaries,
            "fold_records": csv_row_records,
        }

        if self._output_directory is not None:
            self._output_directory.mkdir(parents=True, exist_ok=True)
            json_report_path = self._output_directory / "cv_benchmark_summary.json"
            csv_report_path = self._output_directory / "cv_fold_scores.csv"
            json_report_path.write_text(
                json.dumps(benchmark_report, indent=2) + "\n",
                encoding="utf-8",
            )
            column_headers = list(csv_row_records[0].keys()) if csv_row_records else []
            with csv_report_path.open("w", encoding="utf-8", newline="") as csv_handle:
                writer_obj = csv.DictWriter(csv_handle, fieldnames=column_headers)
                writer_obj.writeheader()
                for row_record in csv_row_records:
                    writer_obj.writerow(row_record)
            benchmark_report["saved_json_path"] = str(json_report_path)
            benchmark_report["saved_csv_path"] = str(csv_report_path)

        return benchmark_report
