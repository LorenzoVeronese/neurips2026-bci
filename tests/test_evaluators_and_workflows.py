"""Comprehensive unit and integration tests for Layer 9 Evaluators and Layer 10 Workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from neurips_bci.acquisition.corpus_acquirers.ProteusNemarCorpusAcquirer import (
    ProteusNemarCorpusAcquirer,
)
from neurips_bci.acquisition.split_acquirers.ProteusFoldSplitAcquirer import (
    ProteusFoldSplitAcquirer,
)
from neurips_bci.configuration.run_configurations.ExperimentRunConfiguration import (
    ExperimentRunConfiguration,
)
from neurips_bci.container.stream_containers.WindowedCorpusStreamContainer import (
    WindowedCorpusStreamContainer,
)
from neurips_bci.decoder.Decoder import Decoder
from neurips_bci.evaluator.benchmark_evaluators.ArchiveContractBenchmarkEvaluator import (
    ArchiveContractBenchmarkEvaluator,
)
from neurips_bci.evaluator.benchmark_evaluators.CrossValidationBenchmarkEvaluator import (
    CrossValidationBenchmarkEvaluator,
)
from neurips_bci.evaluator.metric_evaluators.CellBalancedAccuracyMetricEvaluator import (
    CellBalancedAccuracyMetricEvaluator,
)
from neurips_bci.workflow.cli_workflows.CommandLineInterfaceWorkflow import (
    CommandLineInterfaceWorkflow,
)
from neurips_bci.workflow.stage_workflows.DataDownloadStageWorkflow import (
    DataDownloadStageWorkflow,
)
from neurips_bci.workflow.stage_workflows.EndToEndReproductionStageWorkflow import (
    EndToEndReproductionStageWorkflow,
)
from neurips_bci.workflow.stage_workflows.EvaluationVerificationStageWorkflow import (
    EvaluationVerificationStageWorkflow,
)
from neurips_bci.workflow.stage_workflows.FeatureExtractionStageWorkflow import (
    FeatureExtractionStageWorkflow,
)
from neurips_bci.workflow.stage_workflows.FullModelTrainingStageWorkflow import (
    FullModelTrainingStageWorkflow,
)
from neurips_bci.workflow.stage_workflows.SubmissionPackagingStageWorkflow import (
    SubmissionPackagingStageWorkflow,
)


class _ToyCausalCentroidDecoder(Decoder):
    """Deterministic causal B=1 decoder used to audit streaming CV invariants."""

    def __init__(self, class_templates: np.ndarray) -> None:
        super().__init__(decoder_variant="toy_causal_centroid")
        self._class_templates = np.asarray(class_templates, dtype=np.float64)
        self._running_sum = np.zeros(self._class_templates.shape[1], dtype=np.float64)
        self._seen_steps = 0
        self.observed_batch_sizes: list[int] = []

    @classmethod
    def fit_from_corpus(cls, train_corpus: WindowedCorpusStreamContainer) -> _ToyCausalCentroidDecoder:
        win_power = np.log(np.var(train_corpus.eeg_windows.astype(np.float64), axis=-1) + 1e-6)
        templates = np.stack(
            [
                win_power[train_corpus.class_labels == cls_idx].mean(axis=0)
                for cls_idx in range(3)
            ],
            axis=0,
        )
        return cls(class_templates=templates)

    def reset_state(self) -> None:
        self._running_sum = np.zeros(self._class_templates.shape[1], dtype=np.float64)
        self._seen_steps = 0
        self.observed_batch_sizes = []

    def predict_proba(self, single_window_batch: np.ndarray) -> np.ndarray:
        batch_arr = np.asarray(single_window_batch, dtype=np.float64)
        self.observed_batch_sizes.append(int(batch_arr.shape[0]))
        if batch_arr.shape[0] != 1:
            raise AssertionError(f"Expected strict B=1 batch, got {batch_arr.shape}")
        sanitized = np.nan_to_num(batch_arr[0], nan=0.0, posinf=500.0, neginf=-500.0)
        centered = sanitized - np.mean(sanitized, axis=-1, keepdims=True)
        feat_vec = np.log(np.var(centered, axis=-1) + 1e-6)
        self._seen_steps += 1
        self._running_sum = self._running_sum + feat_vec
        causal_mean = self._running_sum / float(self._seen_steps)
        recentered = feat_vec - 0.15 * causal_mean
        neg_dists = -np.sum((self._class_templates - recentered[None, :]) ** 2, axis=-1)
        shifted = neg_dists - np.max(neg_dists)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps)
        return probs[None, :]


def _build_valid_mock_submission_zip(zip_destination: Path, tamper_network: bool = False) -> Path:
    """Create a self-contained flat submission .zip for testing ArchiveContractBenchmarkEvaluator."""
    rng = np.random.default_rng(42)
    proj_weights = rng.standard_normal((3, 41)).astype(np.float32) * 0.05
    bias_vec = np.array([0.05, -0.02, -0.03], dtype=np.float32)

    staging_dir = zip_destination.parent / f"_stage_{zip_destination.stem}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    weights_npz = staging_dir / "weights.npz"
    np.savez_compressed(weights_npz, W=proj_weights, b=bias_vec)

    socket_snippet = (
        "import socket as _sk; _sk.socket()\n        " if tamper_network else ""
    )
    submission_code = f'''"""Standalone causal B=1 mock submission."""

from pathlib import Path
import numpy as np
import torch


class _CausalMockPredictor:
    def __init__(self, proj_w: np.ndarray, bias_b: np.ndarray, device: str) -> None:
        self._proj_w = proj_w.astype(np.float64)
        self._bias_b = bias_b.astype(np.float64)
        self._device = device
        self._clean_count = 0
        self._feat_sum = np.zeros(proj_w.shape[1], dtype=np.float64)

    def reset_state(self) -> None:
        self._clean_count = 0
        self._feat_sum = np.zeros(self._proj_w.shape[1], dtype=np.float64)

    def predict_proba(self, batch_x) -> np.ndarray:
        {socket_snippet}if isinstance(batch_x, torch.Tensor):
            arr = batch_x.detach().cpu().numpy()
        else:
            arr = np.asarray(batch_x)
        win = np.asarray(arr[0], dtype=np.float64)
        bad_flag = not bool(np.all(np.isfinite(win)))
        win = np.nan_to_num(win, nan=0.0, posinf=500.0, neginf=-500.0)
        win = np.clip(win, -1000.0, 1000.0)
        win = win - np.mean(win, axis=-1, keepdims=True)
        feat = np.log(np.var(win, axis=-1) + 1e-6)
        if not bad_flag and float(np.ptp(win)) > 0.0:
            self._clean_count += 1
            self._feat_sum = self._feat_sum + feat
        ref_mean = (
            self._feat_sum / float(self._clean_count)
            if self._clean_count > 0
            else np.zeros_like(feat)
        )
        logits = self._proj_w @ (feat - 0.2 * ref_mean) + self._bias_b
        logits = logits - np.max(logits)
        exps = np.exp(logits)
        probs = exps / np.sum(exps)
        return probs[None, :]

    def predict(self, batch_x) -> torch.Tensor:
        probs = self.predict_proba(batch_x)
        pred_cls = int(np.argmax(probs[0]))
        dev = batch_x.device if isinstance(batch_x, torch.Tensor) else self._device
        return torch.tensor([pred_cls], dtype=torch.long, device=dev)


class Solver:
    def load_model(self, meta: dict) -> _CausalMockPredictor:
        sub_dir = Path(meta["submission_dir"])
        with np.load(sub_dir / "weights.npz", allow_pickle=False) as zf:
            w_arr = zf["W"].copy()
            b_arr = zf["b"].copy()
        return _CausalMockPredictor(w_arr, b_arr, str(meta.get("device", "cpu")))
'''
    (staging_dir / "submission.py").write_text(submission_code, encoding="utf-8")
    (staging_dir / "METHOD.md").write_text("# Mock Method\n", encoding="utf-8")
    (staging_dir / "REPRODUCE.md").write_text("# Mock Reproduce\n", encoding="utf-8")

    manifest_files: dict[str, dict[str, object]] = {}
    for item_path in sorted(staging_dir.iterdir()):
        if item_path.is_file():
            raw_bytes = item_path.read_bytes()
            manifest_files[item_path.name] = {
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "size_bytes": len(raw_bytes),
            }
    manifest_payload = {
        "file_count_excluding_manifest": len(manifest_files),
        "files": manifest_files,
    }
    (staging_dir / "MANIFEST_SHA256.json").write_text(
        json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8"
    )

    with zipfile.ZipFile(zip_destination, "w", compression=zipfile.ZIP_DEFLATED) as zf_out:
        for item_path in sorted(staging_dir.iterdir()):
            zf_out.write(item_path, arcname=item_path.name)
    return zip_destination


def test_cell_balanced_accuracy_union_classes_and_equal_cell_weights() -> None:
    evaluator = CellBalancedAccuracyMetricEvaluator()

    # Absent true class (class 2 predicted when true only has {0, 1}) must count as 0.0 recall
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 2, 1, 1], dtype=np.int64)
    # Recalls: class 0 -> 0.5, class 1 -> 1.0, class 2 (absent in true) -> 0.0 => mean = 0.5
    assert abs(evaluator.balanced_accuracy(y_true, y_pred) - 0.5) < 1e-12

    # Unequal cell lengths: Cell A has 9 windows (100% bacc), Cell B has 3 windows (1/3 bacc) => cell_mean = 2/3
    true_cells = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    pred_cells = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0], dtype=np.int64)
    sess_ids = np.array(["S1"] * 9 + ["S2"] * 3, dtype="U16")
    ifaces = np.array(["Graz"] * 9 + ["BrainHero"] * 3, dtype="U16")
    cell_mean = evaluator.cell_mean_balanced_accuracy(true_cells, pred_cells, sess_ids, ifaces)
    assert abs(cell_mean - (2.0 / 3.0)) < 1e-12


def test_fold_summaries_jackknife_and_exact_1024_sign_flip() -> None:
    evaluator = CellBalancedAccuracyMetricEvaluator()

    # Exact analytical verification matching test_jackknife.py:
    # P01..P03, P05..P09 = 0; P04 loso = 4, P10 loso = 8; XS4 = [2, 6, 10, 14]; XS1 = [5, 11]
    deltas_16 = np.array(
        [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 2.0, 6.0, 10.0, 14.0, 5.0, 11.0],
        dtype=np.float64,
    )
    summary = evaluator.summarize_fold_scores(deltas_16)
    assert abs(summary["LOSO"] - 1.2) < 1e-12
    assert abs(summary["XS4"] - 8.0) < 1e-12
    assert abs(summary["k05"] - 4.6) < 1e-12
    assert abs(summary["XS1_pair"] - 8.0) < 1e-12
    assert abs(summary["XS1"] - 7.0) < 1e-12  # mean([2, 10, 5, 11])

    jk = evaluator.compute_participant_jackknife(deltas_16)
    expected_se = float(np.sqrt(16569.0 / 2025.0))
    assert abs(float(jk["full_k05_delta"]) - 4.6) < 1e-12
    assert abs(float(jk["woP04"]) - (6.0 + 4.0 / 9.0)) < 1e-12
    assert abs(float(jk["woP10"]) - (2.0 + 2.0 / 9.0)) < 1e-12
    assert abs(float(jk["jackknife_se"]) - expected_se) < 1e-12
    assert abs(float(jk["z_score"]) - (4.6 / expected_se)) < 1e-12

    # When every one of the 10 participants has a positive delta, only +/- all-same (2 of 1024) match/exceed
    all_positive = np.ones(16, dtype=np.float64) * 2.0
    p_min = evaluator.compute_participant_sign_flip_pvalue(all_positive)
    assert abs(p_min - (2.0 / 1024.0)) < 1e-12

    paired = evaluator.compare_paired_fold_scores(60.0 + deltas_16, np.full(16, 60.0))
    assert abs(float(paired["jackknife_se"]) - expected_se) < 1e-12
    assert paired["positive_participants"] == 2
    assert paired["negative_participants"] == 0


def test_cross_validation_benchmark_evaluator_b1_and_exports(tmp_path: Path) -> None:
    smoke_npz = tmp_path / "smoke_proteus.npz"
    smoke_corpus = ProteusNemarCorpusAcquirer.generate_synthetic_smoke_corpus(
        output_npz_path=smoke_npz,
        random_seed=101,
    )
    folds = ProteusFoldSplitAcquirer(smoke_corpus).acquire()
    out_dir = tmp_path / "cv_out"

    cv_evaluator = CrossValidationBenchmarkEvaluator(
        corpus_container=smoke_corpus,
        fold_containers=folds,
        decoder_builder=lambda tr_corpus: _ToyCausalCentroidDecoder.fit_from_corpus(tr_corpus),
        evaluation_orders=("O0", "W0", "W1", "W2", "W3", "W4"),
        output_directory=out_dir,
    )
    cv_report = cv_evaluator.evaluate()

    assert cv_report["fold_count"] == len(folds)
    order_summaries = cv_report["order_summaries"]
    assert isinstance(order_summaries, dict)
    assert "O0" in order_summaries and "OW5" in order_summaries
    assert (out_dir / "cv_benchmark_summary.json").is_file()
    assert (out_dir / "cv_fold_scores.csv").is_file()


def test_archive_contract_benchmark_evaluator_six_gates_and_tamper_rejection(tmp_path: Path) -> None:
    good_zip = _build_valid_mock_submission_zip(tmp_path / "good_sub.zip", tamper_network=False)
    ref_zip = _build_valid_mock_submission_zip(tmp_path / "ref_sub.zip", tamper_network=False)

    contract_eval = ArchiveContractBenchmarkEvaluator(
        target_archive_path=good_zip,
        execution_device="cpu",
        causality_stream_length=14,
        causality_cut_index=9,
        reference_target=ref_zip,
        max_allowed_decision_flips=0,
        output_directory=tmp_path / "audit_out",
    )
    audit_report = contract_eval.evaluate()
    assert audit_report["all_gates_passed"] is True
    gates_map = audit_report["gates"]
    assert isinstance(gates_map, dict)
    for gate_key in (
        "gate1_static_safety",
        "gate2_readonly_offline_load",
        "gate3_strict_b1_causality",
        "gate4_session_order_invariance",
        "gate5_hostile_eeg_inputs",
        "gate6_reference_parity",
    ):
        assert gates_map[gate_key]["passed"] is True
    assert gates_map["gate3_strict_b1_causality"]["future_leak"] == 0.0
    assert gates_map["gate6_reference_parity"]["decision_flip_count"] == 0
    assert (tmp_path / "audit_out" / "archive_verification_report.json").is_file()

    # Negative check: outbound socket call inside submission.py must be caught by Gate 2
    bad_net_zip = _build_valid_mock_submission_zip(tmp_path / "bad_net.zip", tamper_network=True)
    bad_eval = ArchiveContractBenchmarkEvaluator(
        target_archive_path=bad_net_zip,
        execution_device="cpu",
    )
    bad_report = bad_eval.evaluate()
    assert bad_report["all_gates_passed"] is False
    assert bad_report["gates"]["gate2_readonly_offline_load"]["passed"] is False  # type: ignore[index]


def test_download_and_extraction_stage_workflows_in_smoke_mode(tmp_path: Path) -> None:
    smoke_cfg_src = Path(__file__).resolve().parent.parent / "configs" / "h10_smoke_test.json"
    loaded_cfg = ExperimentRunConfiguration.load_from_json_file(smoke_cfg_src)
    cfg_dict = loaded_cfg.to_dictionary()
    cfg_dict["output_run_directory"] = str(tmp_path / "run_out")
    ds_dict = dict(cfg_dict["dataset_section_configuration"])  # type: ignore[arg-type]
    ds_dict["raw_download_directory"] = str(tmp_path / "raw")
    ds_dict["extracted_cache_directory"] = str(tmp_path / "cache")
    ds_dict["pretrained_checkpoint_directory"] = str(tmp_path / "ckpts")
    cfg_dict["dataset_section_configuration"] = ds_dict
    isolated_cfg = ExperimentRunConfiguration.from_dictionary(cfg_dict)

    dl_workflow = DataDownloadStageWorkflow(isolated_cfg, synthetic_smoke_mode=True)
    dl_report = dl_workflow.execute_workflow()
    assert (tmp_path / "run_out" / "resolved_run_configuration.json").is_file()
    assert len(dl_report["acquired_corpus_paths"]) == 1  # type: ignore[arg-type]

    ex_workflow = FeatureExtractionStageWorkflow(isolated_cfg, synthetic_smoke_mode=True)
    ex_report = ex_workflow.execute_workflow()
    assert ex_report["proteus_window_count"] == 144
    assert ex_report["channel_count"] == 41
    assert (tmp_path / "run_out" / "extraction_summary.json").is_file()


def test_train_package_verify_and_cli_smoke_test_end_to_end(tmp_path: Path) -> None:
    smoke_cfg_src = Path(__file__).resolve().parent.parent / "configs" / "h10_smoke_test.json"
    loaded_cfg = ExperimentRunConfiguration.load_from_json_file(smoke_cfg_src)
    cfg_dict = loaded_cfg.to_dictionary()
    run_root = tmp_path / "e2e_smoke_run"
    cfg_dict["output_run_directory"] = str(run_root)
    ds_dict = dict(cfg_dict["dataset_section_configuration"])  # type: ignore[arg-type]
    ds_dict["raw_download_directory"] = str(run_root / "raw_downloads")
    ds_dict["extracted_cache_directory"] = str(run_root / "extracted_caches")
    ds_dict["pretrained_checkpoint_directory"] = str(run_root / "pretrained_checkpoints")
    cfg_dict["dataset_section_configuration"] = ds_dict
    # Keep smoke test super fast while exercising CNN40 + PseudoChrono + CSP + Specialists + ST checkpoint
    classical_dict = dict(cfg_dict["classical_heads_section_configuration"])  # type: ignore[arg-type]
    classical_dict["enable_tsr4"] = False
    classical_dict["enable_rsnse"] = False
    classical_dict["enable_s3d"] = False
    cfg_dict["classical_heads_section_configuration"] = classical_dict

    custom_cfg_path = tmp_path / "custom_smoke_config.json"
    isolated_cfg = ExperimentRunConfiguration.from_dictionary(cfg_dict)
    isolated_cfg.save_to_json_file(custom_cfg_path)

    cli_workflow = CommandLineInterfaceWorkflow(
        [
            "smoke-test",
            "--config",
            str(custom_cfg_path),
            "--max-folds",
            "2",
        ]
    )
    e2e_summary = cli_workflow.execute_workflow()

    assert e2e_summary["workflow_stage"] == "smoke-test"
    assert (run_root / "resolved_run_configuration.json").is_file()
    assert (run_root / "training_summary.json").is_file()
    assert (run_root / "packaging_summary.json").is_file()

    built_zip = run_root / f"{isolated_cfg.config_identifier}_submission.zip"
    assert built_zip.is_file()

    stages_map = e2e_summary["stages"]  # type: ignore[index]
    verify_eval_map = stages_map["verify_and_evaluate"]  # type: ignore[index]
    archive_audit = verify_eval_map["archive_verification"]  # type: ignore[index]
    assert archive_audit["all_gates_passed"] is True  # type: ignore[index]
    assert archive_audit["gates"]["gate3_strict_b1_causality"]["future_leak"] == 0.0  # type: ignore[index]
    assert archive_audit["gates"]["gate4_session_order_invariance"]["passed"] is True  # type: ignore[index]
    assert archive_audit["gates"]["gate5_hostile_eeg_inputs"]["passed"] is True  # type: ignore[index]

    cv_summary = verify_eval_map["cross_validation"]  # type: ignore[index]
    assert cv_summary["fold_count"] == 2  # type: ignore[index]
    assert "O0" in cv_summary["order_summaries"]  # type: ignore[index]
    assert "W0" in cv_summary["order_summaries"]  # type: ignore[index]

    # Also test standalone CLI `verify` subcommand with `--reference-archive` parity Gate 6
    verify_cli = CommandLineInterfaceWorkflow(
        [
            "verify",
            "--config",
            str(custom_cfg_path),
            "--archive-path",
            str(built_zip),
            "--reference-archive",
            str(built_zip),
        ]
    )
    verify_only_report = verify_cli.execute_workflow()
    gate6 = verify_only_report["archive_verification"]["gates"]["gate6_reference_parity"]  # type: ignore[index]
    assert gate6["passed"] is True
    assert gate6["decision_flip_count"] == 0

    # Verify individual workflow classes instantiate cleanly
    assert isinstance(FullModelTrainingStageWorkflow(isolated_cfg, synthetic_smoke_mode=True), FullModelTrainingStageWorkflow)
    assert isinstance(SubmissionPackagingStageWorkflow(isolated_cfg), SubmissionPackagingStageWorkflow)
    assert isinstance(EvaluationVerificationStageWorkflow(isolated_cfg), EvaluationVerificationStageWorkflow)
    assert isinstance(EndToEndReproductionStageWorkflow(isolated_cfg, synthetic_smoke_mode=True), EndToEndReproductionStageWorkflow)
