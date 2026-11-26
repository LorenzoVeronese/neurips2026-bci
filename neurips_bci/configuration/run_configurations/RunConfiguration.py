"""Abstract parent for top-level RunConfiguration objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from neurips_bci.configuration.Configuration import Configuration


class RunConfiguration(Configuration, ABC):
    """Base class for top-level run configurations governed by a single master random seed."""

    def __init__(self, config_identifier: str, master_random_seed: int) -> None:
        super().__init__(configuration_scope="experiment_run")
        if not isinstance(config_identifier, str) or not config_identifier.strip():
            raise TypeError("config_identifier must be a non-empty str")
        if isinstance(master_random_seed, bool) or not isinstance(master_random_seed, int):
            raise TypeError("master_random_seed must be int")
        self._config_identifier = config_identifier
        self._master_random_seed = master_random_seed

    @property
    def config_identifier(self) -> str:
        return self._config_identifier

    @property
    def master_random_seed(self) -> int:
        return self._master_random_seed

    @abstractmethod
    def save_to_json_file(self, destination_json_path: str | Path) -> Path:
        """Persist the complete resolved run configuration to disk."""
