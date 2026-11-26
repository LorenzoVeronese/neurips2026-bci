"""Typed dataset, external corpus, and checkpoint path/source configuration."""

from __future__ import annotations

from pathlib import Path

from neurips_bci.configuration.section_configurations.SectionConfiguration import SectionConfiguration
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class DatasetSectionConfiguration(SectionConfiguration):
    """Defines all dataset download URLs, staging folders, cache paths, and EEG geometry."""

    def __init__(
        self,
        raw_download_directory: str | Path,
        extracted_cache_directory: str | Path,
        pretrained_checkpoint_directory: str | Path,
        channel_count: int = 41,
        window_sample_count: int = 480,
        sampling_rate_hz: float = 120.0,
        class_count: int = 3,
        nemar_dataset_id: str = "nm000290",
        nemar_dataset_version: str = "1.0.0",
        include_dreyer2023_corpus: bool = True,
        include_shin2018_corpus: bool = True,
        include_zyma2019_corpus: bool = True,
        eegpt_hf_repo_id: str = "braindecode/eegpt-pretrained",
        codebrain_hf_repo_id: str = "braindecode/codebrain-pretrained",
        cbramod_hf_repo_id: str = "braindecode/cbramod-pretrained",
        mirepnet_hf_repo_id: str = "braindecode/mirepnet-pretrained",
    ) -> None:
        super().__init__(configuration_scope="dataset_and_checkpoints")
        self._raw_download_directory = ParameterTypeValidator.require_path(
            raw_download_directory, "raw_download_directory"
        )
        self._extracted_cache_directory = ParameterTypeValidator.require_path(
            extracted_cache_directory, "extracted_cache_directory"
        )
        self._pretrained_checkpoint_directory = ParameterTypeValidator.require_path(
            pretrained_checkpoint_directory, "pretrained_checkpoint_directory"
        )
        self._channel_count = ParameterTypeValidator.require_integer(channel_count, "channel_count", minimum_bound=1)
        self._window_sample_count = ParameterTypeValidator.require_integer(
            window_sample_count, "window_sample_count", minimum_bound=16
        )
        self._sampling_rate_hz = ParameterTypeValidator.require_float(
            sampling_rate_hz, "sampling_rate_hz", minimum_bound=1.0
        )
        self._class_count = ParameterTypeValidator.require_integer(class_count, "class_count", minimum_bound=2)
        self._nemar_dataset_id = ParameterTypeValidator.require_non_empty_string(nemar_dataset_id, "nemar_dataset_id")
        self._nemar_dataset_version = ParameterTypeValidator.require_non_empty_string(
            nemar_dataset_version, "nemar_dataset_version"
        )
        self._include_dreyer2023_corpus = ParameterTypeValidator.require_boolean(
            include_dreyer2023_corpus, "include_dreyer2023_corpus"
        )
        self._include_shin2018_corpus = ParameterTypeValidator.require_boolean(
            include_shin2018_corpus, "include_shin2018_corpus"
        )
        self._include_zyma2019_corpus = ParameterTypeValidator.require_boolean(
            include_zyma2019_corpus, "include_zyma2019_corpus"
        )
        self._eegpt_hf_repo_id = ParameterTypeValidator.require_non_empty_string(eegpt_hf_repo_id, "eegpt_hf_repo_id")
        self._codebrain_hf_repo_id = ParameterTypeValidator.require_non_empty_string(
            codebrain_hf_repo_id, "codebrain_hf_repo_id"
        )
        self._cbramod_hf_repo_id = ParameterTypeValidator.require_non_empty_string(
            cbramod_hf_repo_id, "cbramod_hf_repo_id"
        )
        self._mirepnet_hf_repo_id = ParameterTypeValidator.require_non_empty_string(
            mirepnet_hf_repo_id, "mirepnet_hf_repo_id"
        )

    @property
    def raw_download_directory(self) -> Path:
        return self._raw_download_directory

    @property
    def extracted_cache_directory(self) -> Path:
        return self._extracted_cache_directory

    @property
    def pretrained_checkpoint_directory(self) -> Path:
        return self._pretrained_checkpoint_directory

    @property
    def channel_count(self) -> int:
        return self._channel_count

    @property
    def window_sample_count(self) -> int:
        return self._window_sample_count

    @property
    def sampling_rate_hz(self) -> float:
        return self._sampling_rate_hz

    @property
    def class_count(self) -> int:
        return self._class_count

    @property
    def nemar_dataset_id(self) -> str:
        return self._nemar_dataset_id

    @property
    def nemar_dataset_version(self) -> str:
        return self._nemar_dataset_version

    @property
    def include_dreyer2023_corpus(self) -> bool:
        return self._include_dreyer2023_corpus

    @property
    def include_shin2018_corpus(self) -> bool:
        return self._include_shin2018_corpus

    @property
    def include_zyma2019_corpus(self) -> bool:
        return self._include_zyma2019_corpus

    @property
    def eegpt_hf_repo_id(self) -> str:
        return self._eegpt_hf_repo_id

    @property
    def codebrain_hf_repo_id(self) -> str:
        return self._codebrain_hf_repo_id

    @property
    def cbramod_hf_repo_id(self) -> str:
        return self._cbramod_hf_repo_id

    @property
    def mirepnet_hf_repo_id(self) -> str:
        return self._mirepnet_hf_repo_id

    def to_dictionary(self) -> dict[str, object]:
        return {
            "raw_download_directory": str(self._raw_download_directory),
            "extracted_cache_directory": str(self._extracted_cache_directory),
            "pretrained_checkpoint_directory": str(self._pretrained_checkpoint_directory),
            "channel_count": self._channel_count,
            "window_sample_count": self._window_sample_count,
            "sampling_rate_hz": self._sampling_rate_hz,
            "class_count": self._class_count,
            "nemar_dataset_id": self._nemar_dataset_id,
            "nemar_dataset_version": self._nemar_dataset_version,
            "include_dreyer2023_corpus": self._include_dreyer2023_corpus,
            "include_shin2018_corpus": self._include_shin2018_corpus,
            "include_zyma2019_corpus": self._include_zyma2019_corpus,
            "eegpt_hf_repo_id": self._eegpt_hf_repo_id,
            "codebrain_hf_repo_id": self._codebrain_hf_repo_id,
            "cbramod_hf_repo_id": self._cbramod_hf_repo_id,
            "mirepnet_hf_repo_id": self._mirepnet_hf_repo_id,
        }
