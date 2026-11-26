"""Stage workflow running 6-gate archive survival verification and B=1 cross-validation."""

from __future__ import annotations

import json
from pathlib import Path

from neurips_bci.acquisition.corpus_acquirers.ProteusNemarCorpusAcquirer import (
    ProteusNemarCorpusAcquirer,
)
from neurips_bci.acquisition.split_acquirers.ProteusFoldSplitAcquirer import (
    ProteusFoldSplitAcquirer,
)
from neurips_bci.configuration.run_configurations.ExperimentRunConfiguration import (
    ExperimentRunConfiguration,
)
from neurips_bci.container.stream_containers.ValidationFoldStreamContainer import (
    ValidationFoldStreamContainer,
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
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator
from neurips_bci.workflow.stage_workflows.FullModelTrainingStageWorkflow import (
    FullModelTrainingStageWorkflow,
)
from neurips_bci.workflow.stage_workflows.StageWorkflow import StageWorkflow


class EvaluationVerificationStageWorkflow(StageWorkflow):
    """Executes 6-gate archive survival verification and/or streaming cross-validation."""

    def __init__(
        self,
        run_configuration: ExperimentRunConfiguration,
        target_archive_path: str | Path | None = None,
        reference_archive_path: str | Path | None = None,
        run_archive_verification: bool = True,
        run_cross_validation: bool = True,
        synthetic_smoke_mode: bool = False,
        max_cv_folds: int | None = None,
        require_all_gates_pass: bool = True,
    ) -> None:
        stage_tag = (
            "verify"
            if (run_archive_verification and not run_cross_validation)
            else ("evaluate" if (run_cross_validation and not run_archive_verification) else "verify_and_evaluate")
        )
        super().__init__(workflow_stage=stage_tag, run_configuration=run_configuration)
        self._target_archive_path = (
            None
            if target_archive_path is None
            else ParameterTypeValidator.require_path(target_archive_path, "target_archive_path")
        )
        self._reference_archive_path = (
            None
            if reference_archive_path is None
            else ParameterTypeValidator.require_path(
                reference_archive_path, "reference_archive_path", must_exist=True
            )
        )
        self._run_archive_verification = ParameterTypeValidator.require_boolean(
            run_archive_verification, "run_archive_verification"
        )
        self._run_cross_validation = ParameterTypeValidator.require_boolean(
            run_cross_validation, "run_cross_validation"
        )
        if not self._run_archive_verification and not self._run_cross_validation:
            raise ValueError(
                "At least one of run_archive_verification or run_cross_validation must be True"
            )
        self._synthetic_smoke_mode = ParameterTypeValidator.require_boolean(
            synthetic_smoke_mode, "synthetic_smoke_mode"
        )
        self._max_cv_folds = (
            None
            if max_cv_folds is None
            else ParameterTypeValidator.require_integer(max_cv_folds, "max_cv_folds", minimum_bound=1)
        )
        self._require_all_gates_pass = ParameterTypeValidator.require_boolean(
            require_all_gates_pass, "require_all_gates_pass"
        )

    def _resolve_archive_target(self) -> Path:
        if self._target_archive_path is not None:
            return ParameterTypeValidator.require_path(
                self._target_archive_path, "target_archive_path", must_exist=True
            )
        out_root = self.run_configuration.output_run_directory
        default_zip = out_root / f"{self.run_configuration.config_identifier}_submission.zip"
        if default_zip.is_file():
            return default_zip
        default_bundle = out_root / "submission_bundle"
        if default_bundle.is_dir():
            return default_bundle
        raise FileNotFoundError(
            f"Neither {default_zip} nor {default_bundle} exists; run the 'package' stage first"
        )

    def execute_workflow(self) -> dict[str, object]:
        resolved_cfg_path = self.persist_resolved_configuration()
        dataset_cfg = self.run_configuration.dataset_section_configuration
        cnn_cfg = self.run_configuration.cnn_ensemble_section_configuration
        eval_out_dir = self.run_configuration.output_run_directory / "evaluation"
        eval_out_dir.mkdir(parents=True, exist_ok=True)

        verification_report: dict[str, object] | None = None
        if self._run_archive_verification:
            archive_target = self._resolve_archive_target()
            first_ckpt = (
                min(int(ckpt_idx) for ckpt_idx in cnn_cfg.self_train_checkpoints)
                if cnn_cfg.self_train_checkpoints
                else 12
            )
            cut_idx = max(4, first_ckpt + 1)
            stream_len = cut_idx + 4
            audit_seed = self.run_configuration.derive_component_seed("archive_contract_audit", 0)
            contract_evaluator = ArchiveContractBenchmarkEvaluator(
                target_archive_path=archive_target,
                execution_device=self.run_configuration.execution_device,
                channel_count=dataset_cfg.channel_count,
                window_sample_count=dataset_cfg.window_sample_count,
                sampling_rate_hz=dataset_cfg.sampling_rate_hz,
                causality_stream_length=stream_len,
                causality_cut_index=cut_idx,
                reference_target=self._reference_archive_path,
                output_directory=eval_out_dir,
                random_seed=audit_seed,
            )
            verification_report = contract_evaluator.evaluate()
            if self._require_all_gates_pass and not bool(verification_report["all_gates_passed"]):
                raise RuntimeError(
                    f"Archive verification failed one or more gates for {archive_target}: "
                    f"{json.dumps(verification_report['gates'])}"
                )

        cv_report: dict[str, object] | None = None
        if self._run_cross_validation:
            proteus_npz_path = dataset_cfg.extracted_cache_directory / "proteus_windows.npz"
            if self._synthetic_smoke_mode and not proteus_npz_path.is_file():
                smoke_seed = self.run_configuration.derive_component_seed("smoke_proteus_corpus", 0)
                proteus_corpus = ProteusNemarCorpusAcquirer.generate_synthetic_smoke_corpus(
                    output_npz_path=proteus_npz_path,
                    random_seed=smoke_seed,
                )
            elif proteus_npz_path.is_file():
                proteus_corpus = WindowedCorpusStreamContainer.load_npz_archive(proteus_npz_path)
            else:
                proteus_corpus = ProteusNemarCorpusAcquirer(dataset_cfg).acquire_corpus_container()

            all_folds = ProteusFoldSplitAcquirer(proteus_corpus).acquire()
            if self._max_cv_folds is not None:
                selected_folds = all_folds[: self._max_cv_folds]
            elif self._synthetic_smoke_mode:
                loso_sample = [f_item for f_item in all_folds if f_item.fold_regime == "loso"][:2]
                xs4_sample = [f_item for f_item in all_folds if f_item.fold_regime == "xsess4"][:1]
                selected_folds = (loso_sample + xs4_sample) if loso_sample else all_folds[:2]
            else:
                selected_folds = all_folds

            aux_windows, aux_labels = (
                (None, None)
                if self._synthetic_smoke_mode
                else FullModelTrainingStageWorkflow._load_auxiliary_arrays(self.run_configuration)
            )

            def _build_fold_decoder(
                fold_train_corpus: WindowedCorpusStreamContainer,
                fold_spec: ValidationFoldStreamContainer,
            ) -> Decoder:
                fitted_decoder, _ = FullModelTrainingStageWorkflow.fit_decoder_from_corpus(
                    run_configuration=self.run_configuration,
                    proteus_corpus=fold_train_corpus,
                    auxiliary_windows=aux_windows,
                    auxiliary_labels=aux_labels,
                    synthetic_smoke_mode=self._synthetic_smoke_mode,
                    fold_seed_offset=(fold_spec.fold_index + 1) * 1000,
                )
                return fitted_decoder

            cv_evaluator = CrossValidationBenchmarkEvaluator(
                corpus_container=proteus_corpus,
                fold_containers=selected_folds,
                decoder_builder=_build_fold_decoder,
                evaluation_orders=self.run_configuration.evaluation_orders,
                output_directory=eval_out_dir,
            )
            cv_report = cv_evaluator.evaluate()

        stage_report: dict[str, object] = {
            "workflow_stage": self.workflow_stage,
            "config_identifier": self.run_configuration.config_identifier,
            "resolved_configuration_path": str(resolved_cfg_path),
            "evaluation_output_directory": str(eval_out_dir),
            "archive_verification": verification_report,
            "cross_validation": cv_report,
        }
        report_out_path = eval_out_dir / f"{self.workflow_stage}_stage_summary.json"
        report_out_path.write_text(json.dumps(stage_report, indent=2) + "\n", encoding="utf-8")
        stage_report["saved_summary_path"] = str(report_out_path)
        return stage_report
