"""Command-line interface workflow dispatching `neurips-bci` subcommands."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from neurips_bci.configuration.run_configurations.ExperimentRunConfiguration import (
    ExperimentRunConfiguration,
)
from neurips_bci.workflow.cli_workflows.CliWorkflow import CliWorkflow
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


class CommandLineInterfaceWorkflow(CliWorkflow):
    """Parses `neurips-bci` CLI arguments, loads ExperimentRunConfiguration, and runs stages."""

    SUPPORTED_SUBCOMMANDS: tuple[str, ...] = (
        "download",
        "extract",
        "train",
        "package",
        "evaluate",
        "verify",
        "reproduce",
        "smoke-test",
    )

    def __init__(self, cli_arguments: Sequence[str] | None = None) -> None:
        effective_args = tuple(sys.argv[1:] if cli_arguments is None else cli_arguments)
        super().__init__(workflow_stage="cli", cli_arguments=effective_args)

    @classmethod
    def build_argument_parser(cls) -> argparse.ArgumentParser:
        root_parser = argparse.ArgumentParser(
            prog="neurips-bci",
            description="NeurIPS BCI causal streaming EEG BCI reproduction suite.",
        )
        subparsers_group = root_parser.add_subparsers(
            dest="subcommand_choice",
            required=True,
        )
        for cmd_token in cls.SUPPORTED_SUBCOMMANDS:
            cmd_parser = subparsers_group.add_parser(cmd_token)
            cmd_parser.add_argument(
                "--config",
                dest="config_file_path",
                required=True,
                help="Path to ExperimentRunConfiguration JSON file.",
            )
            cmd_parser.add_argument(
                "--output-dir",
                dest="output_dir_override",
                default=None,
                help="Optional override for output_run_directory.",
            )
            if cmd_token in ("verify", "evaluate"):
                cmd_parser.add_argument(
                    "--archive-path",
                    dest="archive_path_override",
                    default=None,
                    help="Optional path to submission folder or .zip archive.",
                )
                cmd_parser.add_argument(
                    "--reference-archive",
                    dest="reference_archive_override",
                    default=None,
                    help="Optional reference .zip archive for Gate 6 parity check.",
                )
            if cmd_token in ("evaluate", "reproduce", "smoke-test"):
                cmd_parser.add_argument(
                    "--max-folds",
                    dest="max_cv_folds",
                    type=int,
                    default=None,
                    help="Optional cap on the number of CV folds evaluated.",
                )
        return root_parser

    def execute_workflow(self) -> dict[str, object]:
        arg_parser = self.build_argument_parser()
        parsed_ns = arg_parser.parse_args(list(self.cli_arguments))

        run_cfg = ExperimentRunConfiguration.load_from_json_file(parsed_ns.config_file_path)
        if parsed_ns.output_dir_override is not None:
            cfg_dict = run_cfg.to_dictionary()
            cfg_dict["output_run_directory"] = str(parsed_ns.output_dir_override)
            run_cfg = ExperimentRunConfiguration.from_dictionary(cfg_dict)

        chosen_cmd = str(parsed_ns.subcommand_choice)
        if chosen_cmd == "download":
            stage_summary = DataDownloadStageWorkflow(run_configuration=run_cfg).execute_workflow()
        elif chosen_cmd == "extract":
            stage_summary = FeatureExtractionStageWorkflow(
                run_configuration=run_cfg
            ).execute_workflow()
        elif chosen_cmd == "train":
            stage_summary = FullModelTrainingStageWorkflow(
                run_configuration=run_cfg
            ).execute_workflow()
        elif chosen_cmd == "package":
            stage_summary = SubmissionPackagingStageWorkflow(
                run_configuration=run_cfg
            ).execute_workflow()
        elif chosen_cmd == "verify":
            stage_summary = EvaluationVerificationStageWorkflow(
                run_configuration=run_cfg,
                target_archive_path=getattr(parsed_ns, "archive_path_override", None),
                reference_archive_path=getattr(parsed_ns, "reference_archive_override", None),
                run_archive_verification=True,
                run_cross_validation=False,
                require_all_gates_pass=True,
            ).execute_workflow()
        elif chosen_cmd == "evaluate":
            stage_summary = EvaluationVerificationStageWorkflow(
                run_configuration=run_cfg,
                target_archive_path=getattr(parsed_ns, "archive_path_override", None),
                reference_archive_path=getattr(parsed_ns, "reference_archive_override", None),
                run_archive_verification=False,
                run_cross_validation=True,
                max_cv_folds=getattr(parsed_ns, "max_cv_folds", None),
            ).execute_workflow()
        elif chosen_cmd == "reproduce":
            stage_summary = EndToEndReproductionStageWorkflow(
                run_configuration=run_cfg,
                synthetic_smoke_mode=False,
                run_cross_validation=True,
                max_cv_folds=getattr(parsed_ns, "max_cv_folds", None),
            ).execute_workflow()
        elif chosen_cmd == "smoke-test":
            stage_summary = EndToEndReproductionStageWorkflow(
                run_configuration=run_cfg,
                synthetic_smoke_mode=True,
                run_cross_validation=True,
                max_cv_folds=getattr(parsed_ns, "max_cv_folds", None),
            ).execute_workflow()
        else:
            raise ValueError(f"Unsupported subcommand: {chosen_cmd}")

        return stage_summary


def run_cli_entrypoint() -> None:
    """Console script entrypoint wired in pyproject.toml (`neurips-bci`)."""
    cli_runner = CommandLineInterfaceWorkflow()
    execution_summary = cli_runner.execute_workflow()
    sys.stdout.write(json.dumps(execution_summary, indent=2) + "\n")
