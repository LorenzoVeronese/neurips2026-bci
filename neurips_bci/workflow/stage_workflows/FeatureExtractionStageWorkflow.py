"""Stage workflow extracting 41-channel 120 Hz EEG windows, resting baselines, and CV splits."""

from __future__ import annotations

import json

from neurips_bci.acquisition.corpus_acquirers.CognitiveAuxiliaryCorpusAcquirer import (
    CognitiveAuxiliaryCorpusAcquirer,
)
from neurips_bci.acquisition.corpus_acquirers.DreyerZenodoCorpusAcquirer import (
    DreyerZenodoCorpusAcquirer,
)
from neurips_bci.acquisition.corpus_acquirers.ProteusNemarCorpusAcquirer import (
    ProteusNemarCorpusAcquirer,
)
from neurips_bci.acquisition.split_acquirers.ProteusFoldSplitAcquirer import (
    ProteusFoldSplitAcquirer,
)
from neurips_bci.configuration.run_configurations.ExperimentRunConfiguration import (
    ExperimentRunConfiguration,
)
from neurips_bci.container.stream_containers.WindowedCorpusStreamContainer import (
    WindowedCorpusStreamContainer,
)
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator
from neurips_bci.workflow.stage_workflows.StageWorkflow import StageWorkflow


class FeatureExtractionStageWorkflow(StageWorkflow):
    """Extracts and validates windowed .npz caches and cross-validation fold indices."""

    def __init__(
        self,
        run_configuration: ExperimentRunConfiguration,
        synthetic_smoke_mode: bool = False,
    ) -> None:
        super().__init__(workflow_stage="extract", run_configuration=run_configuration)
        self._synthetic_smoke_mode = ParameterTypeValidator.require_boolean(
            synthetic_smoke_mode, "synthetic_smoke_mode"
        )

    def execute_workflow(self) -> dict[str, object]:
        resolved_cfg_path = self.persist_resolved_configuration()
        dataset_cfg = self.run_configuration.dataset_section_configuration
        dataset_cfg.extracted_cache_directory.mkdir(parents=True, exist_ok=True)

        proteus_npz_path = dataset_cfg.extracted_cache_directory / "proteus_windows.npz"
        if self._synthetic_smoke_mode and not proteus_npz_path.is_file():
            smoke_seed = self.run_configuration.derive_component_seed("smoke_proteus_corpus", 0)
            proteus_corpus = ProteusNemarCorpusAcquirer.generate_synthetic_smoke_corpus(
                output_npz_path=proteus_npz_path,
                random_seed=smoke_seed,
            )
        else:
            proteus_acquirer = ProteusNemarCorpusAcquirer(dataset_cfg)
            proteus_corpus = proteus_acquirer.acquire_corpus_container()

        auxiliary_summaries: dict[str, dict[str, object]] = {}
        if not self._synthetic_smoke_mode:
            if dataset_cfg.include_dreyer2023_corpus:
                dreyer_npz = DreyerZenodoCorpusAcquirer(dataset_cfg).acquire()
                dreyer_corpus = WindowedCorpusStreamContainer.load_npz_archive(dreyer_npz)
                auxiliary_summaries["dreyer2023"] = {
                    "cache_path": str(dreyer_npz),
                    "window_count": dreyer_corpus.window_count(),
                }
            if dataset_cfg.include_shin2018_corpus or dataset_cfg.include_zyma2019_corpus:
                cog_acquirer = CognitiveAuxiliaryCorpusAcquirer(dataset_cfg)
                if dataset_cfg.include_shin2018_corpus:
                    shin_npz = cog_acquirer.acquire_shin2018_corpus()
                    shin_corpus = WindowedCorpusStreamContainer.load_npz_archive(shin_npz)
                    auxiliary_summaries["shin2018"] = {
                        "cache_path": str(shin_npz),
                        "window_count": shin_corpus.window_count(),
                    }
                if dataset_cfg.include_zyma2019_corpus:
                    zyma_npz = cog_acquirer.acquire_zyma2019_corpus()
                    zyma_corpus = WindowedCorpusStreamContainer.load_npz_archive(zyma_npz)
                    auxiliary_summaries["zyma2019"] = {
                        "cache_path": str(zyma_npz),
                        "window_count": zyma_corpus.window_count(),
                    }

        fold_acquirer = ProteusFoldSplitAcquirer(proteus_corpus)
        cv_folds = fold_acquirer.acquire()
        warmup_fold = fold_acquirer.build_official_warmup_fold()

        extraction_summary: dict[str, object] = {
            "workflow_stage": self.workflow_stage,
            "config_identifier": self.run_configuration.config_identifier,
            "resolved_configuration_path": str(resolved_cfg_path),
            "proteus_cache_path": str(proteus_npz_path),
            "proteus_window_count": proteus_corpus.window_count(),
            "channel_count": len(proteus_corpus.channel_labels),
            "sampling_rate_hz": proteus_corpus.sampling_rate_hz,
            "cv_fold_count": len(cv_folds),
            "cv_fold_labels": [fold_obj.fold_label for fold_obj in cv_folds],
            "warmup_fold_label": warmup_fold.fold_label,
            "auxiliary_corpora": auxiliary_summaries,
        }
        summary_out_path = self.run_configuration.output_run_directory / "extraction_summary.json"
        summary_out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_out_path.write_text(json.dumps(extraction_summary, indent=2) + "\n", encoding="utf-8")
        extraction_summary["saved_summary_path"] = str(summary_out_path)
        return extraction_summary
