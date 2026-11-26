"""Abstract root class for end-to-end reproduction and CLI Workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Workflow(ABC):
    """Base contract for orchestrated pipeline stages (download, extract, train, package, evaluate)."""

    def __init__(self, workflow_stage: str) -> None:
        if not isinstance(workflow_stage, str) or not workflow_stage.strip():
            raise TypeError("workflow_stage must be a non-empty str")
        self._workflow_stage = workflow_stage

    @property
    def workflow_stage(self) -> str:
        return self._workflow_stage

    @abstractmethod
    def execute_workflow(self) -> dict[str, object]:
        """Execute the pipeline stage and return its structured summary report."""
