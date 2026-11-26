"""End-to-end stage workflow orchestrating download, extract, train, package, verify, and evaluate."""

from __future__ import annotations

import json
import time

from neurips_bci.configuration.run_configurations.ExperimentRunConfiguration import (
    ExperimentRunConfiguration,
)
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator
from neurips_bci.workflow.stage_workflows.DataDownloadStageWorkflow import (
    DataDownloadStageWorkflow,
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
from neurips_bci.workflow.stage_workflows.StageWorkflow import StageWorkflow
from neurips_bci.workflow.stage_workflows.SubmissionPackagingStageWorkflow import (
    SubmissionPackagingStageWorkflow,
)


class EndToEndReproductionStageWorkflow(StageWorkflow):
    """Orchestrates the complete deterministic NeurIPS BCI reproduction pipeline."""

    def __init__(
        self,
        run_configuration: ExperimentRunConfiguration,
        synthetic_smoke_mode: bool = False,
        run_cross_validation: bool = True,
        max_cv_folds: int | None = None,
    ) -> None:
        stage_label = "smoke-test" if synthetic_smoke_mode else "reproduce"
        super().__init__(workflow_stage=stage_label, run_configuration=run_configuration)
        self._synthetic_smoke_mode = ParameterTypeValidator.require_boolean(
            synthetic_smoke_mode, "synthetic_smoke_mode"
        )
        self._run_cross_validation = ParameterTypeValidator.require_boolean(
            run_cross_validation, "run_cross_validation"
        )
        self._max_cv_folds = (
            None
            if max_cv_folds is None
            else ParameterTypeValidator.require_integer(max_cv_folds, "max_cv_folds", minimum_bound=1)
        )

    @property
    def synthetic_smoke_mode(self) -> bool:
        return self._synthetic_smoke_mode

    def execute_workflow(self) -> dict[str, object]:
        t_pipeline_start = time.perf_counter()
        resolved_cfg_path = self.persist_resolved_configuration()

        download_summary = DataDownloadStageWorkflow(
            run_configuration=self.run_configuration,
            synthetic_smoke_mode=self._synthetic_smoke_mode,
        ).execute_workflow()

        extract_summary = FeatureExtractionStageWorkflow(
            run_configuration=self.run_configuration,
            synthetic_smoke_mode=self._synthetic_smoke_mode,
        ).execute_workflow()

        train_summary = FullModelTrainingStageWorkflow(
            run_configuration=self.run_configuration,
            synthetic_smoke_mode=self._synthetic_smoke_mode,
        ).execute_workflow()

        package_summary = SubmissionPackagingStageWorkflow(
            run_configuration=self.run_configuration,
        ).execute_workflow()

        built_zip_path = str(package_summary["zip_output_path"])
        eval_summary = EvaluationVerificationStageWorkflow(
            run_configuration=self.run_configuration,
            target_archive_path=built_zip_path,
            run_archive_verification=True,
            run_cross_validation=self._run_cross_validation,
            synthetic_smoke_mode=self._synthetic_smoke_mode,
            max_cv_folds=self._max_cv_folds,
            require_all_gates_pass=True,
        ).execute_workflow()

        elapsed_seconds = float(time.perf_counter() - t_pipeline_start)
        reproduction_summary: dict[str, object] = {
            "workflow_stage": self.workflow_stage,
            "config_identifier": self.run_configuration.config_identifier,
            "master_random_seed": self.run_configuration.master_random_seed,
            "synthetic_smoke_mode": self._synthetic_smoke_mode,
            "resolved_configuration_path": str(resolved_cfg_path),
            "elapsed_seconds": elapsed_seconds,
            "stages": {
                "download": download_summary,
                "extract": extract_summary,
                "train": train_summary,
                "package": package_summary,
                "verify_and_evaluate": eval_summary,
            },
        }
        summary_out_path = (
            self.run_configuration.output_run_directory / f"{self.workflow_stage}_summary.json"
        )
        summary_out_path.write_text(
            json.dumps(reproduction_summary, indent=2) + "\n",
            encoding="utf-8",
        )
        reproduction_summary["saved_summary_path"] = str(summary_out_path)
        return reproduction_summary
