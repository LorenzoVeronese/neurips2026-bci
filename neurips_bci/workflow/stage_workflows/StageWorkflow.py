"""Abstract sub-family base class for reproducible pipeline StageWorkflows."""

from __future__ import annotations

from abc import ABC
from pathlib import Path

from neurips_bci.configuration.run_configurations.ExperimentRunConfiguration import (
    ExperimentRunConfiguration,
)
from neurips_bci.workflow.Workflow import Workflow


class StageWorkflow(Workflow, ABC):
    """Base contract for pipeline stages driven by an ExperimentRunConfiguration."""

    def __init__(
        self,
        workflow_stage: str,
        run_configuration: ExperimentRunConfiguration,
    ) -> None:
        super().__init__(workflow_stage=workflow_stage)
        if not isinstance(run_configuration, ExperimentRunConfiguration):
            raise TypeError("run_configuration must be an ExperimentRunConfiguration instance")
        self._run_configuration = run_configuration

    @property
    def run_configuration(self) -> ExperimentRunConfiguration:
        return self._run_configuration

    def persist_resolved_configuration(self) -> Path:
        """Save resolved_run_configuration.json inside output_run_directory on every run."""
        destination_path = (
            self._run_configuration.output_run_directory / "resolved_run_configuration.json"
        )
        return self._run_configuration.save_to_json_file(destination_path)
