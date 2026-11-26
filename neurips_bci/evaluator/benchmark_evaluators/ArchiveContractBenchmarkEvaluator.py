"""Multi-gate competition archive survival and causality BenchmarkEvaluator."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import sys
import tempfile
from typing import Iterator
import uuid
import zipfile

import numpy as np
import torch

from neurips_bci.decoder.Decoder import Decoder
from neurips_bci.evaluator.benchmark_evaluators.BenchmarkEvaluator import BenchmarkEvaluator
from neurips_bci.validator.type_validators.ArchiveSafetyTypeValidator import (
    ArchiveSafetyTypeValidator,
)
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class ArchiveContractBenchmarkEvaluator(BenchmarkEvaluator):
    """Runs 6-gate verification (safety, read-only offline load, causality, reset, hostile EEG, parity)."""

    _DEFAULT_PROTEUS_CHANNELS: tuple[str, ...] = (
        "C2", "CP2", "P2", "Pz", "CPz", "Cz", "C1", "CP1", "P1", "P3",
        "PO7", "P5", "TP7", "CP5", "CP3", "C3", "PO8", "P6", "P4", "TP8",
        "CP6", "CP4", "C6", "C4", "FC6", "FC4", "FC2", "F6", "F4", "F2",
        "AF4", "AF3", "F5", "F3", "F1", "Fz", "FC5", "FC3", "FC1", "FCz", "C5",
    )

    def __init__(
        self,
        target_archive_path: str | Path,
        execution_device: str = "cpu",
        channel_count: int = 41,
        window_sample_count: int = 480,
        sampling_rate_hz: float = 120.0,
        causality_stream_length: int = 16,
        causality_cut_index: int = 12,
        reference_target: str | Path | Decoder | None = None,
        max_allowed_decision_flips: int = 0,
        output_directory: str | Path | None = None,
        random_seed: int = 20261123,
    ) -> None:
        resolved_out_dir = (
            None
            if output_directory is None
            else ParameterTypeValidator.require_path(output_directory, "output_directory")
        )
        super().__init__(
            evaluation_protocol="six_gate_archive_contract",
            output_directory=resolved_out_dir,
        )
        self._target_archive_path = ParameterTypeValidator.require_path(
            target_archive_path, "target_archive_path", must_exist=True
        )
        self._execution_device = ParameterTypeValidator.require_choice(
            execution_device, ("auto", "cuda", "cpu"), "execution_device"
        )
        self._channel_count = ParameterTypeValidator.require_integer(
            channel_count, "channel_count", minimum_bound=1
        )
        self._window_sample_count = ParameterTypeValidator.require_integer(
            window_sample_count, "window_sample_count", minimum_bound=8
        )
        self._sampling_rate_hz = ParameterTypeValidator.require_float(
            sampling_rate_hz, "sampling_rate_hz", minimum_bound=1.0
        )
        self._causality_stream_length = ParameterTypeValidator.require_integer(
            causality_stream_length, "causality_stream_length", minimum_bound=4
        )
        self._causality_cut_index = ParameterTypeValidator.require_integer(
            causality_cut_index, "causality_cut_index", minimum_bound=1
        )
        if self._causality_cut_index >= self._causality_stream_length - 1:
            raise ValueError("causality_cut_index must be strictly less than causality_stream_length - 1")
        if reference_target is not None and not isinstance(reference_target, (str, Path, Decoder)):
            raise TypeError("reference_target must be str, Path, Decoder, or None")
        self._reference_target = (
            ParameterTypeValidator.require_path(reference_target, "reference_target", must_exist=True)
            if isinstance(reference_target, (str, Path))
            else reference_target
        )
        self._max_allowed_decision_flips = ParameterTypeValidator.require_integer(
            max_allowed_decision_flips, "max_allowed_decision_flips", minimum_bound=0
        )
        self._random_seed = ParameterTypeValidator.require_integer(random_seed, "random_seed", minimum_bound=0)

    @property
    def target_archive_path(self) -> Path:
        return self._target_archive_path

    def _resolve_device_string(self) -> str:
        if self._execution_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self._execution_device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self._execution_device

    def _build_channel_labels(self) -> list[str]:
        if self._channel_count <= len(self._DEFAULT_PROTEUS_CHANNELS):
            return list(self._DEFAULT_PROTEUS_CHANNELS[: self._channel_count])
        extra_labels = [
            f"EEG{idx:03d}"
            for idx in range(len(self._DEFAULT_PROTEUS_CHANNELS), self._channel_count)
        ]
        return list(self._DEFAULT_PROTEUS_CHANNELS) + extra_labels

    @staticmethod
    @contextmanager
    def _network_socket_guard() -> Iterator[None]:
        original_socket_cls = socket.socket
        original_create_conn = socket.create_connection

        def _blocked_socket(*_pos_args: object, **_kw_args: object) -> None:
            raise RuntimeError("Outbound network socket creation is forbidden during submission execution")

        socket.socket = _blocked_socket  # type: ignore[misc,assignment]
        socket.create_connection = _blocked_socket  # type: ignore[assignment]
        try:
            yield
        finally:
            socket.socket = original_socket_cls  # type: ignore[misc,assignment]
            socket.create_connection = original_create_conn

    @staticmethod
    @contextmanager
    def _deterministic_env_guard() -> Iterator[None]:
        previous_det = os.environ.get("R31_FORCE_DET")
        previous_hf = os.environ.get("HF_HUB_OFFLINE")
        os.environ["R31_FORCE_DET"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            yield
        finally:
            if previous_det is None:
                os.environ.pop("R31_FORCE_DET", None)
            else:
                os.environ["R31_FORCE_DET"] = previous_det
            if previous_hf is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_hf

    @staticmethod
    def _stage_bundle_directory(archive_source: Path, staging_directory: Path) -> Path:
        bundle_dir = staging_directory / "readonly_submission_bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        if archive_source.is_dir():
            for src_entry in sorted(archive_source.iterdir()):
                if src_entry.is_file():
                    shutil.copy2(src_entry, bundle_dir / src_entry.name)
        else:
            with zipfile.ZipFile(archive_source, "r") as zip_handle:
                zip_handle.extractall(bundle_dir)
        return bundle_dir

    @staticmethod
    def _set_readonly_permissions(bundle_dir: Path, make_readonly: bool) -> None:
        if make_readonly:
            for child_entry in bundle_dir.rglob("*"):
                if child_entry.is_file():
                    child_entry.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                elif child_entry.is_dir():
                    child_entry.chmod(
                        stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
                    )
            bundle_dir.chmod(
                stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
            )
        else:
            bundle_dir.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
            for child_entry in bundle_dir.rglob("*"):
                if child_entry.is_dir():
                    child_entry.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
                elif child_entry.is_file():
                    child_entry.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

    def _load_submission_model(self, bundle_dir: Path, device_str: str) -> object:
        submission_py = bundle_dir / "submission.py"
        unique_mod_tag = f"_proteus_sub_audit_{uuid.uuid4().hex[:12]}"
        module_spec = importlib.util.spec_from_file_location(unique_mod_tag, submission_py)
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError(f"Unable to build import spec for {submission_py}")
        loaded_module = importlib.util.module_from_spec(module_spec)
        sys.modules[unique_mod_tag] = loaded_module
        try:
            module_spec.loader.exec_module(loaded_module)
            channel_list = self._build_channel_labels()
            solver_meta = {
                "sfreq": self._sampling_rate_hz,
                "ch_names": channel_list,
                "chs_info": [{"ch_name": ch_str} for ch_str in channel_list],
                "n_chans": self._channel_count,
                "n_times": self._window_sample_count,
                "n_classes": 3,
                "device": device_str,
                "submission_dir": bundle_dir,
            }
            solver_cls = getattr(loaded_module, "Solver", None)
            if solver_cls is not None:
                solver_instance = solver_cls()
                return solver_instance.load_model(solver_meta)
            predictor_cls = getattr(loaded_module, "Predictor", None)
            if predictor_cls is not None and hasattr(predictor_cls, "load"):
                return predictor_cls.load(bundle_dir, device_str)
            raise AttributeError("submission.py must define Solver.load_model(meta) or Predictor.load(dir, device)")
        finally:
            sys.modules.pop(unique_mod_tag, None)

    @staticmethod
    def _fresh_session_clone(base_model: object) -> object:
        cloned_model = copy.deepcopy(base_model)
        reset_hook = getattr(cloned_model, "reset_state", None)
        if callable(reset_hook):
            reset_hook()
        return cloned_model

    @staticmethod
    def _step_single_window(
        active_model: object,
        window_array_b1: np.ndarray,
        device_str: str,
    ) -> tuple[int, np.ndarray | None]:
        tensor_b1 = torch.as_tensor(window_array_b1, dtype=torch.float32, device=device_str)
        proba_vec: np.ndarray | None = None
        predict_proba_fn = getattr(active_model, "predict_proba", None)
        if callable(predict_proba_fn):
            try:
                raw_proba = predict_proba_fn(tensor_b1)
            except TypeError:
                raw_proba = predict_proba_fn(window_array_b1)
            if isinstance(raw_proba, torch.Tensor):
                proba_arr = raw_proba.detach().cpu().numpy()
            else:
                proba_arr = np.asarray(raw_proba, dtype=np.float64)
            proba_vec = np.asarray(proba_arr, dtype=np.float64).reshape(-1)
            predicted_label = int(np.argmax(proba_vec))
            return predicted_label, proba_vec

        predict_fn = getattr(active_model, "predict", None)
        if not callable(predict_fn):
            raise TypeError("Loaded submission model does not expose predict(X)")
        raw_pred = predict_fn(tensor_b1)
        if isinstance(raw_pred, torch.Tensor):
            pred_np = raw_pred.detach().cpu().numpy()
        else:
            pred_np = np.asarray(raw_pred)
        pred_flat = pred_np.reshape(-1)
        if len(pred_flat) != 1:
            raise ValueError(f"predict(X) must return shape (1,) for B=1, got {pred_np.shape}")
        return int(pred_flat[0]), None

    @classmethod
    def _stream_windows(
        cls,
        active_model: object,
        session_windows: np.ndarray,
        device_str: str,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        labels_out: list[int] = []
        probas_out: list[np.ndarray] = []
        has_probas = True
        for step_idx in range(session_windows.shape[0]):
            pred_cls, proba_row = cls._step_single_window(
                active_model,
                session_windows[step_idx : step_idx + 1],
                device_str,
            )
            labels_out.append(pred_cls)
            if proba_row is not None and has_probas:
                probas_out.append(proba_row)
            else:
                has_probas = False
        stacked_probas = np.stack(probas_out, axis=0) if (has_probas and probas_out) else None
        return np.asarray(labels_out, dtype=np.int64), stacked_probas

    def evaluate(self) -> dict[str, object]:
        gate_summaries: dict[str, dict[str, object]] = {}

        # Gate 1: Static AST + flat layout + SHA-256 manifest verification
        safety_validator = ArchiveSafetyTypeValidator(self._target_archive_path)
        gate1_inspection = safety_validator.inspect_archive()
        gate_summaries["gate1_static_safety"] = {
            "passed": bool(gate1_inspection["all_checks_passed"]),
            "entry_count": gate1_inspection["entry_count"],
            "violations": gate1_inspection["violations"],
        }

        device_str = self._resolve_device_string()
        rng_gen = np.random.default_rng(self._random_seed)

        with tempfile.TemporaryDirectory(prefix="neurips_bci_audit_") as temp_root_str:
            temp_root = Path(temp_root_str)
            bundle_dir = self._stage_bundle_directory(self._target_archive_path, temp_root)
            self._set_readonly_permissions(bundle_dir, make_readonly=True)
            try:
                with self._deterministic_env_guard(), self._network_socket_guard():
                    # Gate 2: Read-only directory (0555) + blocked network socket load & B=1 step
                    loaded_model: object | None = None
                    try:
                        loaded_model = self._load_submission_model(bundle_dir, device_str)
                        probe_window = (
                            rng_gen.standard_normal(
                                (1, self._channel_count, self._window_sample_count)
                            ).astype(np.float32)
                            * 12.0
                        )
                        gate2_clone = self._fresh_session_clone(loaded_model)
                        gate2_label, gate2_proba = self._step_single_window(
                            gate2_clone, probe_window, device_str
                        )
                        valid_gate2_label = gate2_label in (0, 1, 2)
                        valid_gate2_proba = (
                            True
                            if gate2_proba is None
                            else bool(
                                gate2_proba.shape == (3,)
                                and np.all(np.isfinite(gate2_proba))
                                and abs(float(np.sum(gate2_proba)) - 1.0) < 1e-3
                            )
                        )
                        gate_summaries["gate2_readonly_offline_load"] = {
                            "passed": bool(valid_gate2_label and valid_gate2_proba),
                            "readonly_mode_octal": oct(bundle_dir.stat().st_mode & 0o777),
                            "sample_predicted_class": gate2_label,
                        }
                    except Exception as exc_gate2:
                        gate_summaries["gate2_readonly_offline_load"] = {
                            "passed": False,
                            "error_message": repr(exc_gate2),
                        }

                    if loaded_model is None or not bool(
                        gate_summaries["gate2_readonly_offline_load"]["passed"]
                    ):
                        for skipped_gate in (
                            "gate3_strict_b1_causality",
                            "gate4_session_order_invariance",
                            "gate5_hostile_eeg_inputs",
                            "gate6_reference_parity",
                        ):
                            gate_summaries[skipped_gate] = {
                                "passed": False,
                                "error_message": "Skipped because Gate 2 failed",
                            }
                    else:
                        # Gate 3: Strict B=1 prefix-sabotage causality across self-training checkpoint
                        base_stream = (
                            rng_gen.standard_normal(
                                (
                                    self._causality_stream_length,
                                    self._channel_count,
                                    self._window_sample_count,
                                )
                            ).astype(np.float32)
                            * 15.0
                        )
                        mutated_stream = base_stream.copy()
                        mutated_stream[self._causality_cut_index + 1 :] = (
                            rng_gen.standard_normal(
                                mutated_stream[self._causality_cut_index + 1 :].shape
                            ).astype(np.float32)
                            * 250.0
                        )
                        clean_clone = self._fresh_session_clone(loaded_model)
                        sabotaged_clone = self._fresh_session_clone(loaded_model)
                        preds_clean, probas_clean = self._stream_windows(
                            clean_clone, base_stream, device_str
                        )
                        preds_sab, probas_sab = self._stream_windows(
                            sabotaged_clone, mutated_stream, device_str
                        )
                        cut_end = self._causality_cut_index + 1
                        prefix_label_flips = int(
                            np.sum(preds_clean[:cut_end] != preds_sab[:cut_end])
                        )
                        if probas_clean is not None and probas_sab is not None:
                            future_leak = float(
                                np.max(np.abs(probas_clean[:cut_end] - probas_sab[:cut_end]))
                            )
                        else:
                            future_leak = float(prefix_label_flips)
                        gate_summaries["gate3_strict_b1_causality"] = {
                            "passed": bool(prefix_label_flips == 0 and future_leak == 0.0),
                            "cut_index": self._causality_cut_index,
                            "stream_length": self._causality_stream_length,
                            "prefix_decision_flips": prefix_label_flips,
                            "future_leak": future_leak,
                        }

                        # Gate 4: reset_state() & session-order invariance ([SessA, SessB] vs [SessB, SessA])
                        sess_a_windows = (
                            rng_gen.standard_normal(
                                (8, self._channel_count, self._window_sample_count)
                            ).astype(np.float32)
                            * 10.0
                        )
                        sess_b_windows = (
                            rng_gen.standard_normal(
                                (8, self._channel_count, self._window_sample_count)
                            ).astype(np.float32)
                            * 22.0
                            + 3.0
                        )
                        # Path 1: Benchopt copy.deepcopy + reset_state() in [A, B] vs [B, A]
                        clone_ab_a = self._fresh_session_clone(loaded_model)
                        preds_ab_a, prob_ab_a = self._stream_windows(
                            clone_ab_a, sess_a_windows, device_str
                        )
                        clone_ab_b = self._fresh_session_clone(loaded_model)
                        preds_ab_b, prob_ab_b = self._stream_windows(
                            clone_ab_b, sess_b_windows, device_str
                        )

                        clone_ba_b = self._fresh_session_clone(loaded_model)
                        preds_ba_b, prob_ba_b = self._stream_windows(
                            clone_ba_b, sess_b_windows, device_str
                        )
                        clone_ba_a = self._fresh_session_clone(loaded_model)
                        preds_ba_a, prob_ba_a = self._stream_windows(
                            clone_ba_a, sess_a_windows, device_str
                        )

                        # Path 2: Sequential reset_state() on one instance
                        reused_instance = self._fresh_session_clone(loaded_model)
                        _, _ = self._stream_windows(reused_instance, sess_b_windows, device_str)
                        reset_method = getattr(reused_instance, "reset_state", None)
                        if callable(reset_method):
                            reset_method()
                        preds_seq_a, prob_seq_a = self._stream_windows(
                            reused_instance, sess_a_windows, device_str
                        )

                        order_flips = int(
                            np.sum(preds_ab_a != preds_ba_a)
                            + np.sum(preds_ab_b != preds_ba_b)
                            + np.sum(preds_ab_a != preds_seq_a)
                        )
                        if (
                            prob_ab_a is not None
                            and prob_ba_a is not None
                            and prob_ab_b is not None
                            and prob_ba_b is not None
                            and prob_seq_a is not None
                        ):
                            order_max_diff = float(
                                max(
                                    np.max(np.abs(prob_ab_a - prob_ba_a)),
                                    np.max(np.abs(prob_ab_b - prob_ba_b)),
                                    np.max(np.abs(prob_ab_a - prob_seq_a)),
                                )
                            )
                        else:
                            order_max_diff = float(order_flips)
                        gate_summaries["gate4_session_order_invariance"] = {
                            "passed": bool(order_flips == 0 and order_max_diff <= 1e-6),
                            "session_order_decision_flips": order_flips,
                            "session_order_max_diff": order_max_diff,
                        }

                        # Gate 5: Hostile EEG inputs survival
                        normal_five = (
                            rng_gen.standard_normal(
                                (5, self._channel_count, self._window_sample_count)
                            ).astype(np.float32)
                            * 12.0
                        )
                        nan_case = normal_five.copy()
                        nan_case[1, 0, 10:30] = np.nan
                        nan_case[3, :, :] = np.nan

                        pos_inf_case = normal_five.copy()
                        pos_inf_case[2, 3 % self._channel_count, 5:25] = np.inf

                        neg_inf_case = normal_five.copy()
                        neg_inf_case[2, 1 % self._channel_count, 40:60] = -np.inf

                        flatline_case = np.zeros_like(normal_five)

                        dead_channel_case = normal_five.copy()
                        dead_channel_case[:, 0, :] = 0.0

                        spike_5000uv_case = normal_five.copy()
                        spike_5000uv_case[1, :, 100:120] = 5000.0
                        spike_5000uv_case[3, :, 200:220] = -5000.0

                        hostile_suite: dict[str, np.ndarray] = {
                            "nan_injected": nan_case,
                            "pos_inf_injected": pos_inf_case,
                            "neg_inf_injected": neg_inf_case,
                            "all_zero_flatline": flatline_case,
                            "single_dead_channel": dead_channel_case,
                            "spike_5000uv": spike_5000uv_case,
                            "short_5_window_stream": normal_five.copy(),
                        }
                        hostile_outcomes: dict[str, bool] = {}
                        hostile_errors: dict[str, str] = {}
                        for scenario_key, scenario_windows in hostile_suite.items():
                            try:
                                hostile_clone = self._fresh_session_clone(loaded_model)
                                h_preds, h_probas = self._stream_windows(
                                    hostile_clone, scenario_windows, device_str
                                )
                                labels_valid = bool(
                                    h_preds.shape == (scenario_windows.shape[0],)
                                    and np.all(np.isin(h_preds, (0, 1, 2)))
                                )
                                probas_valid = (
                                    True
                                    if h_probas is None
                                    else bool(
                                        h_probas.shape == (scenario_windows.shape[0], 3)
                                        and np.all(np.isfinite(h_probas))
                                    )
                                )
                                hostile_outcomes[scenario_key] = bool(labels_valid and probas_valid)
                            except Exception as exc_hostile:
                                hostile_outcomes[scenario_key] = False
                                hostile_errors[scenario_key] = repr(exc_hostile)
                        gate_summaries["gate5_hostile_eeg_inputs"] = {
                            "passed": all(hostile_outcomes.values()),
                            "scenario_outcomes": hostile_outcomes,
                            "scenario_errors": hostile_errors,
                        }

                        # Gate 6: Optional parity check against a reference .zip, directory, or Decoder
                        if self._reference_target is None:
                            gate_summaries["gate6_reference_parity"] = {
                                "passed": True,
                                "skipped": True,
                            }
                        elif isinstance(self._reference_target, Decoder):
                            ref_clone = self._fresh_session_clone(self._reference_target)
                            cand_clone = self._fresh_session_clone(loaded_model)
                            cand_preds, cand_probas = self._stream_windows(
                                cand_clone, base_stream, device_str
                            )
                            ref_preds, ref_probas = self._stream_windows(
                                ref_clone, base_stream, device_str
                            )
                            flip_count = int(np.sum(cand_preds != ref_preds))
                            max_prob_delta = (
                                float(np.max(np.abs(cand_probas - ref_probas)))
                                if (cand_probas is not None and ref_probas is not None)
                                else float("nan")
                            )
                            gate_summaries["gate6_reference_parity"] = {
                                "passed": flip_count <= self._max_allowed_decision_flips,
                                "skipped": False,
                                "decision_flip_count": flip_count,
                                "max_probability_delta": max_prob_delta,
                            }
                        else:
                            ref_stage_dir = temp_root / "ref_stage"
                            ref_bundle_dir = self._stage_bundle_directory(
                                self._reference_target, ref_stage_dir
                            )
                            ref_model = self._load_submission_model(ref_bundle_dir, device_str)
                            ref_clone = self._fresh_session_clone(ref_model)
                            cand_clone = self._fresh_session_clone(loaded_model)
                            cand_preds, cand_probas = self._stream_windows(
                                cand_clone, base_stream, device_str
                            )
                            ref_preds, ref_probas = self._stream_windows(
                                ref_clone, base_stream, device_str
                            )
                            flip_count = int(np.sum(cand_preds != ref_preds))
                            max_prob_delta = (
                                float(np.max(np.abs(cand_probas - ref_probas)))
                                if (cand_probas is not None and ref_probas is not None)
                                else float("nan")
                            )
                            gate_summaries["gate6_reference_parity"] = {
                                "passed": flip_count <= self._max_allowed_decision_flips,
                                "skipped": False,
                                "decision_flip_count": flip_count,
                                "max_probability_delta": max_prob_delta,
                            }
            finally:
                self._set_readonly_permissions(bundle_dir, make_readonly=False)

        overall_passed = all(
            bool(gate_dict.get("passed", False)) for gate_dict in gate_summaries.values()
        )
        verification_report: dict[str, object] = {
            "evaluation_protocol": self.evaluation_protocol,
            "target_archive_path": str(self._target_archive_path),
            "execution_device": device_str,
            "all_gates_passed": overall_passed,
            "gates": gate_summaries,
        }

        if self._output_directory is not None:
            self._output_directory.mkdir(parents=True, exist_ok=True)
            report_json_path = self._output_directory / "archive_verification_report.json"
            report_json_path.write_text(
                json.dumps(verification_report, indent=2) + "\n",
                encoding="utf-8",
            )
            verification_report["saved_report_path"] = str(report_json_path)

        return verification_report
