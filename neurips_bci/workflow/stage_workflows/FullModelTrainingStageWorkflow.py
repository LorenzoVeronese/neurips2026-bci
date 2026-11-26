"""Stage workflow training all enabled H10 organs and adapters from extracted corpora."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import torch

from neurips_bci.acquisition.checkpoint_acquirers.HuggingFaceCheckpointAcquirer import (
    HuggingFaceCheckpointAcquirer,
)
from neurips_bci.acquisition.corpus_acquirers.ProteusNemarCorpusAcquirer import (
    ProteusNemarCorpusAcquirer,
)
from neurips_bci.adaptation.gate_adapters.SessionPaceGateAdapter import SessionPaceGateAdapter
from neurips_bci.adaptation.signal_adapters.CausalArtifactSignalAdapter import (
    CausalArtifactSignalAdapter,
)
from neurips_bci.adaptation.signal_adapters.PseudoChronoSignalAdapter import (
    PseudoChronoSignalAdapter,
)
from neurips_bci.backbone.foundation_backbones.CbramodFoundationBackbone import (
    CbramodFoundationBackbone,
)
from neurips_bci.backbone.foundation_backbones.CodebrainFoundationBackbone import (
    CodebrainFoundationBackbone,
)
from neurips_bci.backbone.foundation_backbones.MirepnetFoundationBackbone import (
    MirepnetFoundationBackbone,
)
from neurips_bci.configuration.run_configurations.ExperimentRunConfiguration import (
    ExperimentRunConfiguration,
)
from neurips_bci.container.stream_containers.WindowedCorpusStreamContainer import (
    WindowedCorpusStreamContainer,
)
from neurips_bci.decoder.ensemble_decoders.H10MultimodalEnsembleDecoder import (
    H10MultimodalEnsembleDecoder,
)
from neurips_bci.organ.cnn_organs.DualBlockCnnEnsembleCnnOrgan import (
    DualBlockCnnEnsembleCnnOrgan,
)
from neurips_bci.organ.foundation_organs.CbramodReadoutFoundationOrgan import (
    CbramodReadoutFoundationOrgan,
)
from neurips_bci.organ.foundation_organs.CodebrainLoraFoundationOrgan import (
    CodebrainLoraFoundationOrgan,
)
from neurips_bci.organ.foundation_organs.MirepnetSplineFoundationOrgan import (
    MirepnetSplineFoundationOrgan,
)
from neurips_bci.organ.foundation_organs.QuadViewEegptFoundationOrgan import (
    QuadViewEegptFoundationOrgan,
)
from neurips_bci.organ.multimodal_organs.AnchorSpecialistMultimodalOrgan import (
    AnchorSpecialistMultimodalOrgan,
)
from neurips_bci.organ.multimodal_organs.FisherAnchoredRidgeMultimodalOrgan import (
    FisherAnchoredRidgeMultimodalOrgan,
)
from neurips_bci.organ.riemannian_organs.RestSubspaceNuisanceRiemannianOrgan import (
    RestSubspaceNuisanceRiemannianOrgan,
)
from neurips_bci.organ.riemannian_organs.SubwindowGeodesicRiemannianOrgan import (
    SubwindowGeodesicRiemannianOrgan,
)
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator
from neurips_bci.workflow.stage_workflows.StageWorkflow import StageWorkflow


class FullModelTrainingStageWorkflow(StageWorkflow):
    """Trains the H10 ensemble decoder and persists flat pickle-free organ artifacts."""

    def __init__(
        self,
        run_configuration: ExperimentRunConfiguration,
        synthetic_smoke_mode: bool = False,
        trained_artifacts_directory: str | Path | None = None,
    ) -> None:
        super().__init__(workflow_stage="train", run_configuration=run_configuration)
        self._synthetic_smoke_mode = ParameterTypeValidator.require_boolean(
            synthetic_smoke_mode, "synthetic_smoke_mode"
        )
        self._trained_artifacts_directory = (
            run_configuration.output_run_directory / "trained_artifacts"
            if trained_artifacts_directory is None
            else ParameterTypeValidator.require_path(
                trained_artifacts_directory, "trained_artifacts_directory"
            )
        )

    @property
    def synthetic_smoke_mode(self) -> bool:
        return self._synthetic_smoke_mode

    @property
    def trained_artifacts_directory(self) -> Path:
        return self._trained_artifacts_directory

    @staticmethod
    def _resolve_execution_device(requested_device: str) -> str:
        if requested_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return requested_device

    @staticmethod
    def _load_auxiliary_arrays(
        run_configuration: ExperimentRunConfiguration,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        dataset_cfg = run_configuration.dataset_section_configuration
        cache_dir = dataset_cfg.extracted_cache_directory
        aux_win_parts: list[np.ndarray] = []
        aux_lbl_parts: list[np.ndarray] = []
        candidate_flags = (
            (dataset_cfg.include_dreyer2023_corpus, "dreyer2023_windows.npz"),
            (dataset_cfg.include_shin2018_corpus, "shin2018_windows.npz"),
            (dataset_cfg.include_zyma2019_corpus, "zyma2019_windows.npz"),
        )
        for is_enabled, cache_filename in candidate_flags:
            cand_npz = cache_dir / cache_filename
            if is_enabled and cand_npz.is_file():
                aux_corpus = WindowedCorpusStreamContainer.load_npz_archive(cand_npz)
                aux_win_parts.append(aux_corpus.eeg_windows)
                aux_lbl_parts.append(aux_corpus.class_labels)
        if not aux_win_parts:
            return None, None
        return np.concatenate(aux_win_parts, axis=0), np.concatenate(aux_lbl_parts, axis=0)

    @staticmethod
    def _load_codebrain_backbone(
        checkpoint_path: Path | None,
        execution_device: str,
    ) -> CodebrainFoundationBackbone:
        if checkpoint_path is not None and checkpoint_path.is_file():
            with np.load(checkpoint_path, allow_pickle=False) as archive:
                has_enc_prefix = any(entry_k.startswith("enc__") for entry_k in archive.files)
            if has_enc_prefix:
                backbone, _, _, _, _ = CodebrainFoundationBackbone.from_npz_archive(
                    checkpoint_path, execution_device=execution_device
                )
                return backbone
            backbone = CodebrainFoundationBackbone()
            with np.load(checkpoint_path, allow_pickle=False) as archive:
                raw_sd = {
                    entry_k: torch.from_numpy(archive[entry_k].astype(np.float32))
                    for entry_k in archive.files
                    if entry_k != "config_json" and np.issubdtype(archive[entry_k].dtype, np.number)
                }
            backbone.load_state_dict(raw_sd, strict=False)
            return backbone.to(torch.device(execution_device)).eval()
        return CodebrainFoundationBackbone().to(torch.device(execution_device)).eval()

    @staticmethod
    def _load_cbramod_backbone(
        checkpoint_path: Path | None,
        execution_device: str,
    ) -> CbramodFoundationBackbone:
        if checkpoint_path is not None and checkpoint_path.is_file():
            with np.load(checkpoint_path, allow_pickle=False) as archive:
                has_enc_prefix = any(entry_k.startswith("enc__") for entry_k in archive.files)
            if has_enc_prefix:
                return CbramodFoundationBackbone.from_npz_archive(
                    checkpoint_path, execution_device=execution_device
                )
            backbone = CbramodFoundationBackbone()
            with np.load(checkpoint_path, allow_pickle=False) as archive:
                raw_sd = {
                    entry_k: torch.from_numpy(archive[entry_k].astype(np.float32))
                    for entry_k in archive.files
                    if entry_k != "config_json" and np.issubdtype(archive[entry_k].dtype, np.number)
                }
            backbone.load_state_dict(raw_sd, strict=False)
            return backbone.to(torch.device(execution_device)).eval()
        return CbramodFoundationBackbone().to(torch.device(execution_device)).eval()

    @staticmethod
    def _load_mirepnet_backbone(
        checkpoint_path: Path | None,
        execution_device: str,
    ) -> MirepnetFoundationBackbone:
        if checkpoint_path is not None and checkpoint_path.is_file():
            try:
                return MirepnetFoundationBackbone.from_fp16_npz(
                    checkpoint_path, execution_device=execution_device
                )
            except Exception:
                backbone = MirepnetFoundationBackbone()
                with np.load(checkpoint_path, allow_pickle=False) as archive:
                    raw_sd = {
                        entry_k: torch.from_numpy(archive[entry_k].astype(np.float32))
                        for entry_k in archive.files
                        if entry_k != "config_json" and np.issubdtype(archive[entry_k].dtype, np.number)
                    }
                backbone.load_state_dict(raw_sd, strict=False)
                backbone = backbone.to(torch.device(execution_device)).eval()
                backbone.freeze_fused_stem()
                return backbone
        backbone = MirepnetFoundationBackbone().to(torch.device(execution_device)).eval()
        backbone.freeze_fused_stem()
        return backbone

    @classmethod
    def fit_decoder_from_corpus(
        cls,
        run_configuration: ExperimentRunConfiguration,
        proteus_corpus: WindowedCorpusStreamContainer,
        auxiliary_windows: np.ndarray | None = None,
        auxiliary_labels: np.ndarray | None = None,
        synthetic_smoke_mode: bool = False,
        fold_seed_offset: int = 0,
    ) -> tuple[H10MultimodalEnsembleDecoder, dict[str, Any]]:
        """Train an H10MultimodalEnsembleDecoder from a WindowedCorpusStreamContainer."""
        dataset_cfg = run_configuration.dataset_section_configuration
        cnn_cfg = run_configuration.cnn_ensemble_section_configuration
        fm_cfg = run_configuration.foundation_section_configuration
        classical_cfg = run_configuration.classical_heads_section_configuration
        fusion_cfg = run_configuration.fusion_pace_section_configuration
        resolved_dev = cls._resolve_execution_device(run_configuration.execution_device)

        art_prior = CausalArtifactSignalAdapter.fit_training_prior(
            proteus_corpus.eeg_windows,
            proteus_corpus.session_ids,
        )
        artifact_adapter = CausalArtifactSignalAdapter.from_prior_dictionary(
            classical_cfg, art_prior
        )
        pace_gate_adapter = SessionPaceGateAdapter(fusion_cfg)

        pc_adapter = PseudoChronoSignalAdapter.fit_from_corpus(
            corpus_container=proteus_corpus,
            subspace_dim=classical_cfg.pseudochrono_dim,
            ea_tau=classical_cfg.pseudochrono_ea_tau,
            bn_tau=classical_cfg.pseudochrono_bn_tau,
        )

        cnn_seed = run_configuration.derive_component_seed("dual_block_cnn", fold_seed_offset)
        cnn_organ = DualBlockCnnEnsembleCnnOrgan.fit_from_corpus(
            corpus_container=proteus_corpus,
            cnn_config=cnn_cfg,
            classical_config=classical_cfg,
            execution_device=resolved_dev,
            master_seed=cnn_seed,
        )
        cnn_organ._mix_metadata["art_guard"] = {"prior": art_prior}

        tsr4_organ: SubwindowGeodesicRiemannianOrgan | None = None
        if classical_cfg.enable_tsr4:
            tsr_seed = run_configuration.derive_component_seed("tsr4_organ", fold_seed_offset)
            tsr4_organ = SubwindowGeodesicRiemannianOrgan.fit_from_corpus(
                corpus_container=proteus_corpus,
                random_seed=tsr_seed,
                pc_tau=classical_cfg.tsr4_pc_tau,
            )

        rsnse_organ: RestSubspaceNuisanceRiemannianOrgan | None = None
        if classical_cfg.enable_rsnse:
            rsnse_seed = run_configuration.derive_component_seed("rsnse_organ", fold_seed_offset)
            rsnse_organ = RestSubspaceNuisanceRiemannianOrgan.fit_from_corpus(
                corpus_container=proteus_corpus,
                mlp_seeds=classical_cfg.rsnse_mlp_seeds,
                random_seed=rsnse_seed,
            )

        ckpt_dir = dataset_cfg.pretrained_checkpoint_directory
        hf_acquirer = HuggingFaceCheckpointAcquirer(dataset_cfg)

        eegpt_organ: QuadViewEegptFoundationOrgan | None = None
        if fm_cfg.enable_eegpt_bank:
            eegpt_ckpt = ckpt_dir / HuggingFaceCheckpointAcquirer.CHECKPOINT_FILENAMES["eegpt"]
            if not eegpt_ckpt.is_file() and not synthetic_smoke_mode:
                eegpt_ckpt = hf_acquirer.acquire_single_backbone("eegpt")
            if eegpt_ckpt.is_file():
                with np.load(eegpt_ckpt, allow_pickle=False) as zb_eegpt:
                    base_fp16 = {k_str: zb_eegpt[k_str].copy() for k_str in zb_eegpt.files}
                eegpt_seed = run_configuration.derive_component_seed("eegpt_organ", fold_seed_offset)
                eegpt_organ = QuadViewEegptFoundationOrgan.fit_from_corpora(
                    base_fp16_tensors=base_fp16,
                    proteus_windows=proteus_corpus.eeg_windows,
                    proteus_labels=proteus_corpus.class_labels,
                    n_lat52=fm_cfg.eegpt_lat52_seed_count,
                    n_clean4=fm_cfg.eegpt_clean4_seed_count,
                    n_rms4c=fm_cfg.eegpt_rms4c_seed_count,
                    n_rms56=fm_cfg.eegpt_rms56_seed_count,
                    auxiliary_windows=auxiliary_windows,
                    auxiliary_labels=auxiliary_labels,
                    fine_tune_epochs=max(1, fm_cfg.eegpt_lat52_epochs),
                    batch_size=cnn_cfg.batch_size,
                    learning_rate=fm_cfg.eegpt_head_lr,
                    svd_rank=fm_cfg.eegpt_svd_rank,
                    replay_per_class=cnn_cfg.replay_windows_per_class,
                    random_seed=eegpt_seed,
                    execution_device=resolved_dev,
                )

        codebrain_organ: CodebrainLoraFoundationOrgan | None = None
        if fm_cfg.enable_codebrain:
            cb_ckpt = ckpt_dir / HuggingFaceCheckpointAcquirer.CHECKPOINT_FILENAMES["codebrain"]
            if not cb_ckpt.is_file() and not synthetic_smoke_mode:
                cb_ckpt = hf_acquirer.acquire_single_backbone("codebrain")
            cb_backbone = cls._load_codebrain_backbone(
                cb_ckpt if cb_ckpt.is_file() else None, resolved_dev
            )
            cb_seed = run_configuration.derive_component_seed("codebrain_organ", fold_seed_offset)
            codebrain_organ = CodebrainLoraFoundationOrgan.fit_from_corpora(
                codebrain_backbone=cb_backbone,
                proteus_windows=proteus_corpus.eeg_windows,
                proteus_labels=proteus_corpus.class_labels,
                lora_seed_count=fm_cfg.codebrain_lora_seed_count,
                lora_rank=fm_cfg.codebrain_lora_rank,
                lora_epochs=fm_cfg.codebrain_lora_epochs,
                batch_size=cnn_cfg.batch_size,
                svd_components=fm_cfg.collapsed_svd_components,
                inverse_reg_c=fm_cfg.readout_logistic_c,
                random_seed=cb_seed,
                execution_device=resolved_dev,
            )

        cbramod_organ: CbramodReadoutFoundationOrgan | None = None
        if fm_cfg.enable_cbramod:
            cbr_ckpt = ckpt_dir / HuggingFaceCheckpointAcquirer.CHECKPOINT_FILENAMES["cbramod"]
            if not cbr_ckpt.is_file() and not synthetic_smoke_mode:
                cbr_ckpt = hf_acquirer.acquire_single_backbone("cbramod")
            cbr_backbone = cls._load_cbramod_backbone(
                cbr_ckpt if cbr_ckpt.is_file() else None, resolved_dev
            )
            cbramod_organ = CbramodReadoutFoundationOrgan.fit_from_corpus(
                cbramod_backbone=cbr_backbone,
                proteus_windows=proteus_corpus.eeg_windows,
                proteus_labels=proteus_corpus.class_labels,
                batch_size=cnn_cfg.batch_size,
                svd_components=fm_cfg.collapsed_svd_components,
                inverse_reg_c=fm_cfg.readout_logistic_c,
                execution_device=resolved_dev,
            )

        mirepnet_organ: MirepnetSplineFoundationOrgan | None = None
        if fm_cfg.enable_mirepnet:
            mr_ckpt = ckpt_dir / HuggingFaceCheckpointAcquirer.CHECKPOINT_FILENAMES["mirepnet"]
            if not mr_ckpt.is_file() and not synthetic_smoke_mode:
                mr_ckpt = hf_acquirer.acquire_single_backbone("mirepnet")
            mr_backbone = cls._load_mirepnet_backbone(
                mr_ckpt if mr_ckpt.is_file() else None, resolved_dev
            )
            mirepnet_organ = MirepnetSplineFoundationOrgan.fit_from_corpus(
                mirepnet_backbone=mr_backbone,
                proteus_windows=proteus_corpus.eeg_windows,
                proteus_labels=proteus_corpus.class_labels,
                session_ids=proteus_corpus.session_ids,
                source_41_channels=proteus_corpus.channel_labels,
                svd_components=fm_cfg.collapsed_svd_components,
                batch_size=cnn_cfg.batch_size,
                execution_device=resolved_dev,
            )

        anchor_mm_organ: AnchorSpecialistMultimodalOrgan | None = None
        if classical_cfg.enable_anchor_mm:
            amm_dev = resolved_dev if classical_cfg.anchor_mm_gpu_float64 else "cpu"
            anchor_mm_organ = AnchorSpecialistMultimodalOrgan.fit_from_features(
                windows=proteus_corpus.eeg_windows,
                class_labels=proteus_corpus.class_labels,
                subject_ids=proteus_corpus.participant_ids,
                execution_device=amm_dev,
            )

        s3d_organ: FisherAnchoredRidgeMultimodalOrgan | None = None
        if (
            classical_cfg.enable_s3d
            and anchor_mm_organ is not None
            and codebrain_organ is not None
            and cbramod_organ is not None
        ):
            win_count = proteus_corpus.window_count()
            z_sr_rows: list[np.ndarray] = []
            z_ee_rows: list[np.ndarray] = []
            cd_rows: list[np.ndarray] = []
            cb_rows: list[np.ndarray] = []
            anchor_mm_organ.reset_organ_state()
            codebrain_organ.reset_organ_state()
            cbramod_organ.reset_organ_state()
            with torch.no_grad():
                for idx_w in range(win_count):
                    win_mat = np.asarray(proteus_corpus.eeg_windows[idx_w], dtype=np.float64)
                    anchor_mm_organ.step_multimodal(
                        raw_window_f64=win_mat,
                        cleaned_window_f64=win_mat,
                        is_artifact_flagged=False,
                    )
                    z_sr_vec = (
                        anchor_mm_organ.last_z_sr
                        if anchor_mm_organ.last_z_sr is not None
                        else np.zeros(256, dtype=np.float64)
                    )
                    z_ee_vec = (
                        anchor_mm_organ.last_z_ee
                        if anchor_mm_organ.last_z_ee is not None
                        else np.zeros(64, dtype=np.float64)
                    )
                    z_sr_rows.append(z_sr_vec)
                    z_ee_rows.append(z_ee_vec)

                    prep_200, _ = CodebrainLoraFoundationOrgan.prepare_200hz_car_window(win_mat)
                    xt_200 = torch.as_tensor(prep_200[None], dtype=torch.float32, device=resolved_dev)
                    tf_cd = codebrain_organ._codebrain_backbone.forward_frozen_only(xt_200).float()
                    cd_l = tf_cd[:, :, 1:, :].mean(dim=2).reshape(-1)
                    cd_e = tf_cd[:, :, :1, :].mean(dim=2).reshape(-1)
                    cd_rows.append(
                        torch.cat([cd_l, 0.70 * (cd_l - cd_e)], dim=0).cpu().numpy().astype(np.float64)
                    )
                    tok_cb = cbramod_organ._cbramod_backbone.encode_window_features(xt_200).float()
                    cb_m = tok_cb.mean(dim=2).reshape(-1)
                    cb_l = tok_cb[:, :, 1:, :].mean(dim=2).reshape(-1)
                    cb_rows.append(
                        torch.cat([cb_m, 0.70 * cb_l], dim=0).cpu().numpy().astype(np.float64)
                    )
            s3d_organ = FisherAnchoredRidgeMultimodalOrgan.fit_from_multimodal_features(
                windows=proteus_corpus.eeg_windows,
                class_labels=proteus_corpus.class_labels,
                subject_ids=proteus_corpus.participant_ids,
                session_ids=proteus_corpus.session_ids,
                z_sr_matrix=np.stack(z_sr_rows, axis=0),
                z_ee_matrix=np.stack(z_ee_rows, axis=0),
                cd_feature_matrix=np.stack(cd_rows, axis=0),
                cb_feature_matrix=np.stack(cb_rows, axis=0),
                lam0=classical_cfg.s3d_ridge_lambda0,
            )

        decoder_instance = H10MultimodalEnsembleDecoder(
            run_configuration=run_configuration,
            cnn_organ=cnn_organ,
            artifact_adapter=artifact_adapter,
            pace_gate_adapter=pace_gate_adapter,
            pseudo_chrono_adapter=pc_adapter,
            eegpt_organ=eegpt_organ,
            tsr4_organ=tsr4_organ,
            rsnse_organ=rsnse_organ,
            anchor_mm_organ=anchor_mm_organ,
            codebrain_organ=codebrain_organ,
            cbramod_organ=cbramod_organ,
            mirepnet_organ=mirepnet_organ,
            s3d_organ=s3d_organ,
        )
        decoder_instance.reset_state()
        return decoder_instance, art_prior

    def save_decoder_artifacts(
        self,
        decoder_instance: H10MultimodalEnsembleDecoder,
        artifact_prior: dict[str, Any],
        target_directory: Path,
    ) -> list[Path]:
        target_directory.mkdir(parents=True, exist_ok=True)
        written_paths: list[Path] = []

        run_cfg_path = self.run_configuration.save_to_json_file(
            target_directory / "run_configuration.json"
        )
        written_paths.append(run_cfg_path)

        art_prior_path = target_directory / "artifact_prior.json"
        art_prior_path.write_text(json.dumps(artifact_prior, indent=2) + "\n", encoding="utf-8")
        written_paths.append(art_prior_path)

        written_paths.extend(decoder_instance.cnn_organ.save_organ_artifacts(target_directory))

        if decoder_instance._pseudo_chrono_adapter is not None:
            pc_path = decoder_instance._pseudo_chrono_adapter.save_npz(
                target_directory / "pseudochrono_nm_mt6_d6.npz"
            )
            written_paths.append(pc_path)

        for optional_organ in (
            decoder_instance._eegpt_organ,
            decoder_instance._tsr4_organ,
            decoder_instance._rsnse_organ,
            decoder_instance._anchor_mm_organ,
            decoder_instance._codebrain_organ,
            decoder_instance._cbramod_organ,
            decoder_instance._mirepnet_organ,
            decoder_instance._s3d_organ,
        ):
            if optional_organ is not None:
                written_paths.extend(optional_organ.save_organ_artifacts(target_directory))

        return written_paths

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
        elif proteus_npz_path.is_file():
            proteus_corpus = WindowedCorpusStreamContainer.load_npz_archive(proteus_npz_path)
        else:
            proteus_corpus = ProteusNemarCorpusAcquirer(dataset_cfg).acquire_corpus_container()

        aux_windows, aux_labels = (
            (None, None)
            if self._synthetic_smoke_mode
            else self._load_auxiliary_arrays(self.run_configuration)
        )

        decoder_instance, art_prior = self.fit_decoder_from_corpus(
            run_configuration=self.run_configuration,
            proteus_corpus=proteus_corpus,
            auxiliary_windows=aux_windows,
            auxiliary_labels=aux_labels,
            synthetic_smoke_mode=self._synthetic_smoke_mode,
            fold_seed_offset=0,
        )
        written_paths = self.save_decoder_artifacts(
            decoder_instance=decoder_instance,
            artifact_prior=art_prior,
            target_directory=self._trained_artifacts_directory,
        )

        enabled_organs: list[str] = ["dual_block_cnn_ensemble"]
        for organ_label, organ_ref in (
            ("pseudochrono", decoder_instance._pseudo_chrono_adapter),
            ("tsr4", decoder_instance._tsr4_organ),
            ("rsnse", decoder_instance._rsnse_organ),
            ("eegpt", decoder_instance._eegpt_organ),
            ("codebrain", decoder_instance._codebrain_organ),
            ("cbramod", decoder_instance._cbramod_organ),
            ("mirepnet", decoder_instance._mirepnet_organ),
            ("anchor_mm", decoder_instance._anchor_mm_organ),
            ("s3d", decoder_instance._s3d_organ),
        ):
            if organ_ref is not None:
                enabled_organs.append(organ_label)

        summary_payload: dict[str, object] = {
            "workflow_stage": self.workflow_stage,
            "config_identifier": self.run_configuration.config_identifier,
            "master_random_seed": self.run_configuration.master_random_seed,
            "resolved_configuration_path": str(resolved_cfg_path),
            "trained_artifacts_directory": str(self._trained_artifacts_directory),
            "proteus_training_windows": proteus_corpus.window_count(),
            "total_cnn_networks": decoder_instance.cnn_organ.total_cnn_count(),
            "enabled_components": enabled_organs,
            "saved_artifact_count": len(written_paths),
        }
        summary_out_path = self.run_configuration.output_run_directory / "training_summary.json"
        summary_out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_out_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
        summary_payload["saved_summary_path"] = str(summary_out_path)
        return summary_payload
