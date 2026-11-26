"""Master run configuration and sole JSON configuration file reader/writer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from neurips_bci.configuration.run_configurations.RunConfiguration import RunConfiguration
from neurips_bci.configuration.section_configurations.ClassicalHeadsSectionConfiguration import (
    ClassicalHeadsSectionConfiguration,
)
from neurips_bci.configuration.section_configurations.CnnEnsembleSectionConfiguration import (
    CnnEnsembleSectionConfiguration,
)
from neurips_bci.configuration.section_configurations.DatasetSectionConfiguration import (
    DatasetSectionConfiguration,
)
from neurips_bci.configuration.section_configurations.FoundationSectionConfiguration import (
    FoundationSectionConfiguration,
)
from neurips_bci.configuration.section_configurations.FusionPaceSectionConfiguration import (
    FusionPaceSectionConfiguration,
)
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class ExperimentRunConfiguration(RunConfiguration):
    """Defines every aspect of an H10 run and serves as the sole configuration file I/O boundary."""

    def __init__(
        self,
        config_identifier: str,
        master_random_seed: int,
        output_run_directory: str | Path,
        execution_device: str,
        evaluation_orders: tuple[str, ...],
        dataset_section_configuration: DatasetSectionConfiguration,
        cnn_ensemble_section_configuration: CnnEnsembleSectionConfiguration,
        foundation_section_configuration: FoundationSectionConfiguration,
        classical_heads_section_configuration: ClassicalHeadsSectionConfiguration,
        fusion_pace_section_configuration: FusionPaceSectionConfiguration,
    ) -> None:
        super().__init__(config_identifier=config_identifier, master_random_seed=master_random_seed)
        self._output_run_directory = ParameterTypeValidator.require_path(output_run_directory, "output_run_directory")
        self._execution_device = ParameterTypeValidator.require_choice(
            execution_device, ("auto", "cuda", "cpu"), "execution_device"
        )
        if not evaluation_orders:
            raise ValueError("evaluation_orders must be non-empty")
        for order_tag in evaluation_orders:
            ParameterTypeValidator.require_choice(order_tag, ("O0", "W0", "W1", "W2", "W3", "W4"), "evaluation_orders")
        self._evaluation_orders = tuple(evaluation_orders)

        if not isinstance(dataset_section_configuration, DatasetSectionConfiguration):
            raise TypeError("dataset_section_configuration must be DatasetSectionConfiguration")
        if not isinstance(cnn_ensemble_section_configuration, CnnEnsembleSectionConfiguration):
            raise TypeError("cnn_ensemble_section_configuration must be CnnEnsembleSectionConfiguration")
        if not isinstance(foundation_section_configuration, FoundationSectionConfiguration):
            raise TypeError("foundation_section_configuration must be FoundationSectionConfiguration")
        if not isinstance(classical_heads_section_configuration, ClassicalHeadsSectionConfiguration):
            raise TypeError("classical_heads_section_configuration must be ClassicalHeadsSectionConfiguration")
        if not isinstance(fusion_pace_section_configuration, FusionPaceSectionConfiguration):
            raise TypeError("fusion_pace_section_configuration must be FusionPaceSectionConfiguration")

        self._dataset_section_configuration = dataset_section_configuration
        self._cnn_ensemble_section_configuration = cnn_ensemble_section_configuration
        self._foundation_section_configuration = foundation_section_configuration
        self._classical_heads_section_configuration = classical_heads_section_configuration
        self._fusion_pace_section_configuration = fusion_pace_section_configuration

    @property
    def output_run_directory(self) -> Path:
        return self._output_run_directory

    @property
    def execution_device(self) -> str:
        return self._execution_device

    @property
    def evaluation_orders(self) -> tuple[str, ...]:
        return self._evaluation_orders

    @property
    def dataset_section_configuration(self) -> DatasetSectionConfiguration:
        return self._dataset_section_configuration

    @property
    def cnn_ensemble_section_configuration(self) -> CnnEnsembleSectionConfiguration:
        return self._cnn_ensemble_section_configuration

    @property
    def foundation_section_configuration(self) -> FoundationSectionConfiguration:
        return self._foundation_section_configuration

    @property
    def classical_heads_section_configuration(self) -> ClassicalHeadsSectionConfiguration:
        return self._classical_heads_section_configuration

    @property
    def fusion_pace_section_configuration(self) -> FusionPaceSectionConfiguration:
        return self._fusion_pace_section_configuration

    def derive_component_seed(self, component_role_key: str, member_index: int = 0) -> int:
        ParameterTypeValidator.require_non_empty_string(component_role_key, "component_role_key")
        ParameterTypeValidator.require_integer(member_index, "member_index", minimum_bound=0)
        seed_payload = f"{self._master_random_seed}:{component_role_key}:{member_index}".encode("utf-8")
        digest_bytes = hashlib.blake2b(seed_payload, digest_size=4).digest()
        return int.from_bytes(digest_bytes, byteorder="big") & 0x7FFFFFFF

    def to_dictionary(self) -> dict[str, object]:
        return {
            "config_identifier": self._config_identifier,
            "master_random_seed": self._master_random_seed,
            "output_run_directory": str(self._output_run_directory),
            "execution_device": self._execution_device,
            "evaluation_orders": list(self._evaluation_orders),
            "dataset_section_configuration": self._dataset_section_configuration.to_dictionary(),
            "cnn_ensemble_section_configuration": self._cnn_ensemble_section_configuration.to_dictionary(),
            "foundation_section_configuration": self._foundation_section_configuration.to_dictionary(),
            "classical_heads_section_configuration": self._classical_heads_section_configuration.to_dictionary(),
            "fusion_pace_section_configuration": self._fusion_pace_section_configuration.to_dictionary(),
        }

    def save_to_json_file(self, destination_json_path: str | Path) -> Path:
        resolved_destination = ParameterTypeValidator.require_path(destination_json_path, "destination_json_path")
        resolved_destination.parent.mkdir(parents=True, exist_ok=True)
        resolved_destination.write_text(json.dumps(self.to_dictionary(), indent=2) + "\n", encoding="utf-8")
        return resolved_destination

    @classmethod
    def load_from_json_file(cls, source_json_path: str | Path) -> ExperimentRunConfiguration:
        resolved_source = ParameterTypeValidator.require_path(source_json_path, "source_json_path", must_exist=True)
        parsed_payload = json.loads(resolved_source.read_text(encoding="utf-8"))
        return cls.from_dictionary(parsed_payload)

    @classmethod
    def from_dictionary(cls, config_dictionary: dict[str, object]) -> ExperimentRunConfiguration:
        dataset_dict = dict(config_dictionary["dataset_section_configuration"])  # type: ignore[arg-type]
        cnn_dict = dict(config_dictionary.get("cnn_ensemble_section_configuration", {}))  # type: ignore[arg-type]
        if "self_train_checkpoints" in cnn_dict:
            cnn_dict["self_train_checkpoints"] = tuple(cnn_dict["self_train_checkpoints"])  # type: ignore[arg-type]
        fm_dict = dict(config_dictionary.get("foundation_section_configuration", {}))  # type: ignore[arg-type]
        if "eegpt_refit_checkpoints" in fm_dict:
            fm_dict["eegpt_refit_checkpoints"] = tuple(fm_dict["eegpt_refit_checkpoints"])  # type: ignore[arg-type]
        classical_dict = dict(config_dictionary.get("classical_heads_section_configuration", {}))  # type: ignore[arg-type]
        fusion_dict = dict(config_dictionary.get("fusion_pace_section_configuration", {}))  # type: ignore[arg-type]

        return cls(
            config_identifier=str(config_dictionary["config_identifier"]),
            master_random_seed=int(config_dictionary["master_random_seed"]),  # type: ignore[arg-type]
            output_run_directory=str(config_dictionary["output_run_directory"]),
            execution_device=str(config_dictionary.get("execution_device", "auto")),
            evaluation_orders=tuple(config_dictionary.get("evaluation_orders", ["O0", "W0", "W1", "W2", "W3", "W4"])),  # type: ignore[arg-type]
            dataset_section_configuration=DatasetSectionConfiguration(**dataset_dict),  # type: ignore[arg-type]
            cnn_ensemble_section_configuration=CnnEnsembleSectionConfiguration(**cnn_dict),  # type: ignore[arg-type]
            foundation_section_configuration=FoundationSectionConfiguration(**fm_dict),  # type: ignore[arg-type]
            classical_heads_section_configuration=ClassicalHeadsSectionConfiguration(**classical_dict),  # type: ignore[arg-type]
            fusion_pace_section_configuration=FusionPaceSectionConfiguration(**fusion_dict),  # type: ignore[arg-type]
        )
