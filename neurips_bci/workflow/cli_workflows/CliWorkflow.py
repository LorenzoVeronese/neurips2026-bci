"""Abstract sub-family base class for command-line interface Workflows."""

from __future__ import annotations

from abc import ABC
from typing import Sequence

from neurips_bci.workflow.Workflow import Workflow


class CliWorkflow(Workflow, ABC):
    """Base contract for CLI argument parsing and stage dispatch."""

    def __init__(self, workflow_stage: str, cli_arguments: Sequence[str]) -> None:
        super().__init__(workflow_stage=workflow_stage)
        self._cli_arguments = tuple(str(arg_token) for arg_token in cli_arguments)

    @property
    def cli_arguments(self) -> tuple[str, ...]:
        return self._cli_arguments
