"""Abstract parent for the SectionConfiguration sub-family."""

from __future__ import annotations

from abc import ABC

from neurips_bci.configuration.Configuration import Configuration


class SectionConfiguration(Configuration, ABC):
    """Base class for modular domain sections inside an ExperimentRunConfiguration."""

    def __init__(self, configuration_scope: str) -> None:
        super().__init__(configuration_scope=configuration_scope)
