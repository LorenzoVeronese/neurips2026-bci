"""Stage workflow acquiring public BIDS/MOABB/PhysioNet corpora and HuggingFace checkpoints."""

from __future__ import annotations

from neurips_bci.acquisition.checkpoint_acquirers.HuggingFaceCheckpointAcquirer import (
    HuggingFaceCheckpointAcquirer,
)
from neurips_bci.acquisition.corpus_acquirers.CognitiveAuxiliaryCorpusAcquirer import (
    CognitiveAuxiliaryCorpusAcquirer,
)
from neurips_bci.acquisition.corpus_acquirers.DreyerZenodoCorpusAcquirer import (
    DreyerZenodoCorpusAcquirer,
)
from neurips_bci.acquisition.corpus_acquirers.ProteusNemarCorpusAcquirer import (
    ProteusNemarCorpusAcquirer,
)
from neurips_bci.configuration.run_configurations.ExperimentRunConfiguration import (
    ExperimentRunConfiguration,
)
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator
from neurips_bci.workflow.stage_workflows.StageWorkflow import StageWorkflow


class DataDownloadStageWorkflow(StageWorkflow):
    """Downloads raw/cached EEG corpora and enabled foundation-model checkpoints."""

    def __init__(
        self,
        run_configuration: ExperimentRunConfiguration,
        synthetic_smoke_mode: bool = False,
    ) -> None:
        super().__init__(workflow_stage="download", run_configuration=run_configuration)
        self._synthetic_smoke_mode = ParameterTypeValidator.require_boolean(
            synthetic_smoke_mode, "synthetic_smoke_mode"
        )

    @property
    def synthetic_smoke_mode(self) -> bool:
        return self._synthetic_smoke_mode

    def execute_workflow(self) -> dict[str, object]:
        resolved_cfg_path = self.persist_resolved_configuration()
        dataset_cfg = self.run_configuration.dataset_section_configuration
        foundation_cfg = self.run_configuration.foundation_section_configuration

        dataset_cfg.raw_download_directory.mkdir(parents=True, exist_ok=True)
        dataset_cfg.extracted_cache_directory.mkdir(parents=True, exist_ok=True)
        dataset_cfg.pretrained_checkpoint_directory.mkdir(parents=True, exist_ok=True)

        corpus_paths: list[str] = []
        proteus_cache_target = dataset_cfg.extracted_cache_directory / "proteus_windows.npz"

        if self._synthetic_smoke_mode:
            if not proteus_cache_target.is_file():
                smoke_seed = self.run_configuration.derive_component_seed("smoke_proteus_corpus", 0)
                ProteusNemarCorpusAcquirer.generate_synthetic_smoke_corpus(
                    output_npz_path=proteus_cache_target,
                    random_seed=smoke_seed,
                )
            corpus_paths.append(str(proteus_cache_target))
        else:
            proteus_acquirer = ProteusNemarCorpusAcquirer(dataset_cfg)
            corpus_paths.append(str(proteus_acquirer.acquire()))
            if dataset_cfg.include_dreyer2023_corpus:
                dreyer_acquirer = DreyerZenodoCorpusAcquirer(dataset_cfg)
                corpus_paths.append(str(dreyer_acquirer.acquire()))
            if dataset_cfg.include_shin2018_corpus or dataset_cfg.include_zyma2019_corpus:
                cognitive_acquirer = CognitiveAuxiliaryCorpusAcquirer(dataset_cfg)
                for aux_path in cognitive_acquirer.acquire():
                    corpus_paths.append(str(aux_path))

        checkpoint_paths: list[str] = []
        enabled_backbones: list[str] = []
        if foundation_cfg.enable_eegpt_bank:
            enabled_backbones.append("eegpt")
        if foundation_cfg.enable_codebrain:
            enabled_backbones.append("codebrain")
        if foundation_cfg.enable_cbramod:
            enabled_backbones.append("cbramod")
        if foundation_cfg.enable_mirepnet:
            enabled_backbones.append("mirepnet")

        if enabled_backbones and not self._synthetic_smoke_mode:
            hf_acquirer = HuggingFaceCheckpointAcquirer(dataset_cfg)
            for backbone_key in enabled_backbones:
                ckpt_path = hf_acquirer.acquire_single_backbone(backbone_key)
                checkpoint_paths.append(str(ckpt_path))

        return {
            "workflow_stage": self.workflow_stage,
            "config_identifier": self.run_configuration.config_identifier,
            "synthetic_smoke_mode": self._synthetic_smoke_mode,
            "resolved_configuration_path": str(resolved_cfg_path),
            "acquired_corpus_paths": corpus_paths,
            "acquired_checkpoint_paths": checkpoint_paths,
        }
