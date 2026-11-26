"""Top-level H10 (`r45` & `r35`) causal streaming multimodal BCI ensemble decoder."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
import zipfile
import numpy as np
import torch

from neurips_bci.adaptation.gate_adapters.AsymmetricConfidenceGateAdapter import (
    AsymmetricConfidenceGateAdapter,
)
from neurips_bci.adaptation.gate_adapters.ExcessAgreementGateAdapter import (
    ExcessAgreementGateAdapter,
)
from neurips_bci.adaptation.gate_adapters.SessionPaceGateAdapter import SessionPaceGateAdapter
from neurips_bci.adaptation.probability_adapters.KalmanPriorProbabilityAdapter import (
    KalmanPriorProbabilityAdapter,
)
from neurips_bci.adaptation.probability_adapters.PseudoChronoSinkhornProbabilityAdapter import (
    PseudoChronoSinkhornProbabilityAdapter,
)
from neurips_bci.adaptation.probability_adapters.RunningPriorProbabilityAdapter import (
    RunningPriorProbabilityAdapter,
)
from neurips_bci.adaptation.signal_adapters.CausalArtifactSignalAdapter import (
    CausalArtifactSignalAdapter,
)
from neurips_bci.adaptation.signal_adapters.PseudoChronoSignalAdapter import (
    PseudoChronoSignalAdapter,
)
from neurips_bci.configuration.run_configurations.ExperimentRunConfiguration import (
    ExperimentRunConfiguration,
)
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
from neurips_bci.decoder.ensemble_decoders.EnsembleDecoder import EnsembleDecoder
from neurips_bci.organ.cnn_organs.DualBlockCnnEnsembleCnnOrgan import (
    DualBlockCnnEnsembleCnnOrgan,
)
from neurips_bci.organ.foundation_organs.CbramodReadoutFoundationOrgan import (
    CbramodReadoutFoundationOrgan,
)
from neurips_bci.organ.foundation_organs.CodebrainLoraFoundationOrgan import (
    CodebrainLoraFoundationOrgan,
)
from neurips_bci.organ.foundation_organs.FoundationOrgan import FoundationOrgan
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
from neurips_bci.organ.riemannian_organs.SessionCspLdaRiemannianOrgan import (
    SessionCspLdaRiemannianOrgan,
)
from neurips_bci.organ.riemannian_organs.SubwindowGeodesicRiemannianOrgan import (
    SubwindowGeodesicRiemannianOrgan,
)
from neurips_bci.validator.type_validators.ArchiveSafetyTypeValidator import (
    ArchiveSafetyTypeValidator,
)
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class H10MultimodalEnsembleDecoder(EnsembleDecoder):
    """Composes all H10 causal adapters and decoding organs in exact `r45`/`r35` execution order."""

    def __init__(
        self,
        run_configuration: ExperimentRunConfiguration,
        cnn_organ: DualBlockCnnEnsembleCnnOrgan,
        artifact_adapter: CausalArtifactSignalAdapter,
        pace_gate_adapter: SessionPaceGateAdapter,
        pseudo_chrono_adapter: PseudoChronoSignalAdapter | None = None,
        eegpt_organ: QuadViewEegptFoundationOrgan | None = None,
        tsr4_organ: SubwindowGeodesicRiemannianOrgan | None = None,
        rsnse_organ: RestSubspaceNuisanceRiemannianOrgan | None = None,
        anchor_mm_organ: AnchorSpecialistMultimodalOrgan | None = None,
        codebrain_organ: CodebrainLoraFoundationOrgan | None = None,
        cbramod_organ: CbramodReadoutFoundationOrgan | None = None,
        mirepnet_organ: MirepnetSplineFoundationOrgan | None = None,
        s3d_organ: FisherAnchoredRidgeMultimodalOrgan | None = None,
    ) -> None:
        if not isinstance(run_configuration, ExperimentRunConfiguration):
            raise TypeError(
                f"run_configuration must be ExperimentRunConfiguration, got {type(run_configuration).__name__}"
            )
        if not isinstance(cnn_organ, DualBlockCnnEnsembleCnnOrgan):
            raise TypeError(f"cnn_organ must be DualBlockCnnEnsembleCnnOrgan, got {type(cnn_organ).__name__}")
        if not isinstance(artifact_adapter, CausalArtifactSignalAdapter):
            raise TypeError(
                f"artifact_adapter must be CausalArtifactSignalAdapter, got {type(artifact_adapter).__name__}"
            )
        if not isinstance(pace_gate_adapter, SessionPaceGateAdapter):
            raise TypeError(
                f"pace_gate_adapter must be SessionPaceGateAdapter, got {type(pace_gate_adapter).__name__}"
            )

        super().__init__(decoder_variant=run_configuration.config_identifier)
        self._run_configuration = run_configuration
        self._cnn_cfg = run_configuration.cnn_ensemble_section_configuration
        self._fm_cfg = run_configuration.foundation_section_configuration
        self._classical_cfg = run_configuration.classical_heads_section_configuration
        self._fusion_cfg = run_configuration.fusion_pace_section_configuration

        self._cnn_organ = cnn_organ
        self._artifact_adapter = artifact_adapter
        self._pace_gate_adapter = pace_gate_adapter
        self._pseudo_chrono_adapter = pseudo_chrono_adapter
        self._eegpt_organ = eegpt_organ
        self._tsr4_organ = tsr4_organ
        self._rsnse_organ = rsnse_organ
        self._anchor_mm_organ = anchor_mm_organ
        self._codebrain_organ = codebrain_organ
        self._cbramod_organ = cbramod_organ
        self._mirepnet_organ = mirepnet_organ
        self._s3d_organ = s3d_organ

        if self._tsr4_organ is not None and self._pseudo_chrono_adapter is not None:
            self._tsr4_organ.attach_pseudochrono_adapter(self._pseudo_chrono_adapter)
        if self._s3d_organ is not None:
            if self._codebrain_organ is not None:
                self._codebrain_organ.attach_s3d_projection(self._s3d_organ.q_cd)
            if self._cbramod_organ is not None:
                self._cbramod_organ.attach_s3d_projection(self._s3d_organ.q_cb)

        std_prior = self._fusion_cfg.balance_std_prior
        kal_n0 = self._fusion_cfg.kalman_n0
        kal_q = self._fusion_cfg.kalman_q_drift
        pc_dim = (
            self._pseudo_chrono_adapter.subspace_dim
            if self._pseudo_chrono_adapter is not None
            else self._classical_cfg.pseudochrono_dim
        )

        self._cnn40_std_balancer = RunningPriorProbabilityAdapter(prior_strength=std_prior)
        self._cnn40_kal_balancer = KalmanPriorProbabilityAdapter(initial_pseudo_count=kal_n0, drift_variance=kal_q)
        self._bal_a = RunningPriorProbabilityAdapter(prior_strength=std_prior)
        self._bal_a_s7_ref = RunningPriorProbabilityAdapter(prior_strength=std_prior)
        self._bal_a_tsr4_s7 = RunningPriorProbabilityAdapter(prior_strength=std_prior)
        self._bal_b = KalmanPriorProbabilityAdapter(initial_pseudo_count=kal_n0, drift_variance=kal_q)
        self._stoff_kal_balancer = KalmanPriorProbabilityAdapter(initial_pseudo_count=kal_n0, drift_variance=kal_q)

        self._asym_conf_gate = AsymmetricConfidenceGateAdapter()
        self._teacher_sym_gate_81 = AsymmetricConfidenceGateAdapter(
            prefix_mode="r21_biased",
            start_closed=False,
            prior_confidence=0.50,
            ema_half_life=24,
            min_prefix_windows=8,
            instant_uncertainty_mode="r21_floor15",
            instant_power=0.60,
            tau_dm=0.81,
            tau_proto=0.81,
            width_dm=0.12,
            width_proto=0.12,
        )
        self._pre_fm_sinkhorn_adapter = PseudoChronoSinkhornProbabilityAdapter.from_fusion_configuration(
            self._fusion_cfg, subspace_dim=pc_dim
        )
        self._g1_gate_codebrain = ExcessAgreementGateAdapter.from_fusion_configuration(self._fusion_cfg)
        self._g1_gate_cbramod = ExcessAgreementGateAdapter.from_fusion_configuration(self._fusion_cfg)

        self._global_run_t0: float | None = None
        self._session_t0: float | None = None
        self._session_window_index: int = 0
        self._managed_temp_directory: tempfile.TemporaryDirectory[str] | None = None
        self.reset_state()

    @property
    def run_configuration(self) -> ExperimentRunConfiguration:
        return self._run_configuration

    @property
    def cnn_organ(self) -> DualBlockCnnEnsembleCnnOrgan:
        return self._cnn_organ

    @property
    def pace_gate_adapter(self) -> SessionPaceGateAdapter:
        return self._pace_gate_adapter

    def reset_state(self) -> None:
        """Reset all per-session online state across every adapter and organ (`0` cross-session state)."""
        self._session_t0 = None
        self._session_window_index = 0
        self._pace_gate_adapter.reset_session_state()
        self._artifact_adapter.reset_session_state()
        if self._pseudo_chrono_adapter is not None:
            self._pseudo_chrono_adapter.reset_session_state()

        self._cnn_organ.reset_organ_state()
        for optional_organ in (
            self._eegpt_organ,
            self._tsr4_organ,
            self._rsnse_organ,
            self._anchor_mm_organ,
            self._codebrain_organ,
            self._cbramod_organ,
            self._mirepnet_organ,
            self._s3d_organ,
        ):
            if optional_organ is not None:
                optional_organ.reset_organ_state()

        for adapter_obj in (
            self._cnn40_std_balancer,
            self._cnn40_kal_balancer,
            self._bal_a,
            self._bal_a_s7_ref,
            self._bal_a_tsr4_s7,
            self._bal_b,
            self._stoff_kal_balancer,
            self._asym_conf_gate,
            self._teacher_sym_gate_81,
            self._pre_fm_sinkhorn_adapter,
            self._g1_gate_codebrain,
            self._g1_gate_cbramod,
        ):
            adapter_obj.reset_session_state()

    def _extract_known_covariances_from_cnn(self) -> dict[Any, np.ndarray]:
        known_covs: dict[Any, np.ndarray] = {}
        main_a = self._cnn_organ._block_a.get("main")
        if main_a is not None and main_a.get("last_covs") is not None:
            covs_arr = main_a["last_covs"]
            for idx_b, band_pair in enumerate(main_a["bands"]):
                known_covs[tuple(band_pair)] = covs_arr[idx_b]
                known_covs[(int(round(band_pair[0])), int(round(band_pair[1])))] = covs_arr[idx_b]
            known_covs[None] = covs_arr[-1]
        csp_a = self._cnn_organ._csp_a
        if csp_a is not None and csp_a.last_band_covariances is not None:
            for idx_c, bp_c in enumerate(SessionCspLdaRiemannianOrgan.SIX_CSP_BANDS_HZ):
                known_covs[tuple(bp_c)] = csp_a.last_band_covariances[idx_c]
                known_covs[(int(round(bp_c[0])), int(round(bp_c[1])))] = csp_a.last_band_covariances[idx_c]
        return known_covs

    def step_window(self, single_window: np.ndarray) -> np.ndarray:
        """Process one `(41, 480)` window causally through the full H10 ensemble pipeline."""
        step_t_start = time.perf_counter()
        if self._global_run_t0 is None:
            self._global_run_t0 = step_t_start
        if self._session_t0 is None:
            self._session_t0 = step_t_start

        win_idx = self._session_window_index
        is_ckpt_next = (win_idx + 1) in self._cnn_cfg.self_train_checkpoints
        force_det = os.environ.get("R31_FORCE_DET", "0") == "1"
        active_tier, skip_st = self._pace_gate_adapter.evaluate_window_tier(
            window_index=win_idx,
            session_elapsed_s=max(0.0, step_t_start - self._session_t0),
            run_elapsed_s=max(0.0, step_t_start - self._global_run_t0),
            is_checkpoint_next=is_ckpt_next,
            force_deterministic=force_det,
        )

        try:
            probs_out = self._step_window_at_tier(single_window, active_tier=active_tier, skip_self_training=skip_st)
        finally:
            step_elapsed = max(0.0, time.perf_counter() - step_t_start)
            if is_ckpt_next and not skip_st:
                self._pace_gate_adapter.record_self_training_duration(step_elapsed)
            else:
                self._pace_gate_adapter.record_ordinary_window_duration(step_elapsed)
            self._session_window_index += 1
        return probs_out

    def _step_window_at_tier(
        self,
        single_window: np.ndarray,
        active_tier: int,
        skip_self_training: bool,
    ) -> np.ndarray:
        raw_np = np.asarray(single_window, dtype=np.float64)
        if not np.isfinite(raw_np).all():
            raw_np = np.nan_to_num(raw_np, nan=0.0, posinf=500.0, neginf=-500.0)

        # 1. Causal Artifact Guard & PseudoChrono run-subspace coordinate extraction
        cleaned_win, dc_centered_win, is_bad_win = self._artifact_adapter.sanitize_and_gate_window(raw_np)
        cur_u: np.ndarray | None = None
        cur_phi_nm: np.ndarray | None = None
        if self._pseudo_chrono_adapter is not None and active_tier < 3:
            cur_u = self._pseudo_chrono_adapter.begin_window(raw_np)
            cur_phi_nm = self._pseudo_chrono_adapter.cur_phi_nm

        # 2. TSR4 Sub-window Geodesic Riemannian Organ
        u_tsr4: np.ndarray | None = None
        if (
            active_tier < 3
            and self._tsr4_organ is not None
            and self._classical_cfg.enable_tsr4
            and self._fusion_cfg.traj_a_tsr4_weight > 0.0
        ):
            u_tsr4 = self._tsr4_organ.step_causal_window(
                raw_np,
                is_artifact_flagged=is_bad_win,
                pseudo_chrono_adapter=self._pseudo_chrono_adapter,
            )

        # 3. Dual-Block CNN40 + Shallow-STOFF Organ & RSNSE Organ
        is_frozen_tta = bool(active_tier >= 1 and not self._fusion_cfg.tier1_keep_self_training)
        u_cnn40_a, u_shallow_stoff = self._cnn_organ.step_cnn40_and_stoff(
            cleaned_window=cleaned_win,
            dc_centered_window=dc_centered_win,
            is_bad_window=is_bad_win,
            is_frozen_tta=is_frozen_tta,
            skip_self_training=skip_self_training,
            pseudo_chrono_adapter=self._pseudo_chrono_adapter if active_tier < 3 else None,
        )
        p_cnn40_bal_a = self._cnn40_std_balancer.step(u_cnn40_a)
        if active_tier >= 3:
            return p_cnn40_bal_a

        p_cnn40_bal_b = self._cnn40_kal_balancer.step(u_cnn40_a)
        known_covs = self._extract_known_covariances_from_cnn()

        u_rsnse: np.ndarray | None = None
        if (
            active_tier < 2
            and self._rsnse_organ is not None
            and self._classical_cfg.enable_rsnse
            and self._fusion_cfg.traj_b_rsnse_weight > 0.0
        ):
            u_rsnse = self._rsnse_organ.step_causal_window(
                cleaned_win,
                is_artifact_flagged=is_bad_win,
                known_covariances=known_covs,
                pseudo_chrono_adapter=self._pseudo_chrono_adapter,
            )

        # 4. Quad-View EEGPT Foundation Organ
        use_anchor_mm = (
            active_tier < 2
            and self._anchor_mm_organ is not None
            and self._classical_cfg.enable_anchor_mm
        )
        has_eegpt = (
            active_tier < 3
            and self._eegpt_organ is not None
            and self._fm_cfg.enable_eegpt_bank
        )
        if has_eegpt and self._eegpt_organ is not None:
            pop_inv_std0 = (
                self._anchor_mm_organ.inv_std0_id
                if (use_anchor_mm and self._anchor_mm_organ is not None and self._anchor_mm_organ.has_pop_eegpt)
                else None
            )
            u_lat_a, u_rms_a = self._eegpt_organ.step_quad_views(
                cleaned_win,
                is_artifact_flagged=is_bad_win,
                pre_centered_window=dc_centered_win,
                pop_inv_std0_id=pop_inv_std0,
                pseudo_chrono_u=cur_u,
            )
            self._eegpt_organ.observe_teacher(p_cnn40_bal_a, p_cnn40_bal_b)
            u_rms_leg_a = self._eegpt_organ.last_u_rms_leg_a
            u_rms_prot_a = self._eegpt_organ.last_u_rms_prot_a
            u_lat_b = self._eegpt_organ.last_u_lat_b
            u_rms_b = self._eegpt_organ.last_u_rms_leg_b
            w_lat_a = self._fusion_cfg.traj_a_lat52_weight
            w_rms_a = self._fusion_cfg.traj_a_rms56_weight
            w_lat_b = self._fusion_cfg.traj_b_lat52_weight
            w_rms_b = self._fusion_cfg.traj_b_rms56_weight
        else:
            uniform_3 = np.full(3, 1.0 / 3.0, dtype=np.float64)
            u_lat_a = uniform_3
            u_rms_a = uniform_3
            u_rms_leg_a = uniform_3
            u_rms_prot_a = uniform_3
            u_lat_b = uniform_3
            u_rms_b = uniform_3
            w_lat_a = 0.0
            w_rms_a = 0.0
            w_lat_b = 0.0
            w_rms_b = 0.0

        # 5. Trajectory A (`BalStd`) & Trajectory B (`BalKal`) Readouts
        w_tsr_eff = self._fusion_cfg.traj_a_tsr4_weight if u_tsr4 is not None else 0.0
        w_cnn_a = self._fusion_cfg.traj_a_cnn_weight
        tot_a = max(w_cnn_a + w_lat_a + w_rms_a + w_tsr_eff, 1e-8)

        lp_a_s7_ref = (
            (w_cnn_a / tot_a) * self.centered_log_probabilities_1d(u_cnn40_a)
            + (w_lat_a / tot_a) * self.centered_log_probabilities_1d(u_lat_a)
            + (w_rms_a / tot_a) * self.centered_log_probabilities_1d(u_rms_prot_a)
        )
        if w_tsr_eff > 0.0 and u_tsr4 is not None:
            lp_a_s7_ref = lp_a_s7_ref + (w_tsr_eff / tot_a) * self.centered_log_probabilities_1d(u_tsr4)
        p_a_s7_ref = self._bal_a_s7_ref.step(self.softmax_1d(lp_a_s7_ref))
        p_a_tsr4_s7 = self._bal_a_tsr4_s7.step(self.softmax_1d(lp_a_s7_ref))

        use_s7_rms_main = (
            self._anchor_mm_organ is not None
            and self._anchor_mm_organ.ee_slice_mode in ("lat_rms_c4", "lat_c4")
            and self._fusion_cfg.traj_b_rsnse_weight > 0.0
        )
        u_rms_a_main = u_rms_leg_a if use_s7_rms_main else u_rms_a
        lp_a = (
            (w_cnn_a / tot_a) * self.centered_log_probabilities_1d(u_cnn40_a)
            + (w_lat_a / tot_a) * self.centered_log_probabilities_1d(u_lat_a)
            + (w_rms_a / tot_a) * self.centered_log_probabilities_1d(u_rms_a_main)
        )
        if w_tsr_eff > 0.0 and u_tsr4 is not None:
            lp_a = lp_a + (w_tsr_eff / tot_a) * self.centered_log_probabilities_1d(u_tsr4)
        p_a = self._bal_a.step(self.softmax_1d(lp_a))

        w_asym_a = self._fusion_cfg.trajectory_a_weight
        if w_asym_a >= 1.0:
            p_b = p_a
            p_base = p_a
            p_s7a_ref = p_a_s7_ref
            p_tsr4_s7_ref = p_a_tsr4_s7
        else:
            w_cnn_b = max(0.0, 1.0 - w_lat_b - w_rms_b)
            u_slot5_b = self.softmax_1d(
                w_cnn_b * self.centered_log_probabilities_1d(u_cnn40_a)
                + w_lat_b * self.centered_log_probabilities_1d(u_lat_b)
                + w_rms_b * self.centered_log_probabilities_1d(u_rms_b)
            )
            w_rsnse = self._fusion_cfg.traj_b_rsnse_weight
            u_tot_b = (
                self.blend_two_probabilities_log_linear(u_slot5_b, u_rsnse, w_rsnse)
                if (w_rsnse > 0.0 and u_rsnse is not None)
                else u_slot5_b
            )
            p_b = self._bal_b.step(u_tot_b)
            p_base = self.softmax_1d(
                w_asym_a * self.centered_log_probabilities_1d(p_a)
                + (1.0 - w_asym_a) * self.centered_log_probabilities_1d(p_b)
            )
            p_tsr4_s7_ref = self.softmax_1d(
                0.40 * self.centered_log_probabilities_1d(p_a_tsr4_s7)
                + 0.60 * self.centered_log_probabilities_1d(p_b)
            )
            p_s7a_ref = (
                self.softmax_1d(
                    0.40 * self.centered_log_probabilities_1d(p_a_s7_ref)
                    + 0.60 * self.centered_log_probabilities_1d(p_b)
                )
                if (w_rsnse > 0.0 and u_rsnse is not None)
                else p_base
            )

        p_out = p_base

        # 6. ANCHOR-MM Specialist Residual Injection & Multimodal Self-Training Teacher
        if (
            use_anchor_mm
            and self._anchor_mm_organ is not None
            and (not self._anchor_mm_organ.has_ee or (has_eegpt and self._eegpt_organ is not None))
        ):
            p_tchr_mm = p_s7a_ref if (self._fusion_cfg.traj_b_rsnse_weight > 0.0 and u_rsnse is not None) else p_base
            if self._eegpt_organ is not None and self._anchor_mm_organ.has_ee:
                all_f_causal = self._eegpt_organ.last_all_f_causal
                all_f_pop = self._eegpt_organ.last_all_f_pop
                n_lat_ee = self._eegpt_organ.lat_seed_count
                n_c4_ee = self._eegpt_organ.clean4_seed_count
                n_r4c_ee = self._eegpt_organ.rms4c_seed_count
            else:
                all_f_causal = None
                all_f_pop = None
                n_lat_ee = 0
                n_c4_ee = 0
                n_r4c_ee = 0

            mm_out = self._anchor_mm_organ.step_multimodal(
                raw_window_f64=raw_np,
                cleaned_window_f64=cleaned_win,
                is_artifact_flagged=is_bad_win,
                teacher_probabilities=p_tchr_mm,
                all_f_causal=all_f_causal,
                all_f_pop=all_f_pop,
                n_lat=n_lat_ee,
                n_clean4=n_c4_ee,
                n_rms4c=n_r4c_ee,
                known_covs=known_covs,
                pseudo_chrono_u=cur_u,
            )

            g_dm, g_pr = self._asym_conf_gate.step_gates(p_base, is_bad_window=False, proto_reference_probs=p_base)
            lam_dm = self._classical_cfg.anchor_mm_lam_dm
            lam_proto = self._classical_cfg.anchor_mm_lam_proto
            if lam_dm > 0.0 or lam_proto > 0.0:
                lp_damped = (
                    self.centered_log_probabilities_1d(p_base)
                    + (lam_dm * g_dm) * mm_out["res_dualmod"]
                    + (lam_proto * g_pr) * mm_out["res_proto_st"]
                )
                p_out = self.softmax_1d(lp_damped)

            g_t81, _ = self._teacher_sym_gate_81.step_gates(p_s7a_ref, is_bad_window=False)
            p_tchr_mm_st = self.softmax_1d(
                self.centered_log_probabilities_1d(p_tsr4_s7_ref)
                + (0.65 * g_t81) * mm_out["res_dm80_protost20"]
            )
            c4_direct_lam = self._classical_cfg.anchor_mm_c4_direct_lam
            if c4_direct_lam > 0.0:
                p_out = self.softmax_1d(
                    self.centered_log_probabilities_1d(p_out)
                    + c4_direct_lam * (0.5 * mm_out["res_dualmod"] + 0.5 * mm_out["res_proto_st"])
                )
            self._cnn_organ.record_multimodal_teacher_probs(p_tchr_mm_st, is_bad_window=is_bad_win)

        # 7. Shallow STOFF Anchor Blend + Pre-FM PseudoChrono Sinkhorn Cell Balancer
        w_stoff = self._cnn_cfg.shallow_stoff_anchor_weight
        if w_stoff > 0.0:
            p_stoff_kal = self._stoff_kal_balancer.step(u_shallow_stoff)
            p_out = self.blend_two_probabilities_log_linear(p_out, p_stoff_kal, w_stoff)

        if active_tier < 2:
            p_out = self._pre_fm_sinkhorn_adapter.step(
                p_out, u_coordinate=cur_u, is_bad_window=is_bad_win
            )

        # 8. Foundation Model Zoo (`CodeBrain` + `CBraMod` + `MIRepNet`) + G1 Excess Agreement Gate
        run_cd = (
            active_tier < 1
            and self._codebrain_organ is not None
            and self._fm_cfg.enable_codebrain
            and self._fm_cfg.weight_codebrain > 0.0
        )
        run_cb = (
            active_tier < 2
            and self._cbramod_organ is not None
            and self._fm_cfg.enable_cbramod
            and self._fm_cfg.weight_cbramod > 0.0
        )
        run_mr = (
            active_tier < 1
            and self._mirepnet_organ is not None
            and self._fm_cfg.enable_mirepnet
            and self._fm_cfg.weight_mirepnet > 0.0
        )

        if raw_np.shape == (41, 480) and (run_cd or run_cb or run_mr):
            p_base_fm = p_out
            lp_fm = self.centered_log_probabilities_1d(p_out)
            if run_cd or run_cb:
                x_fm_200, ext_fm = FoundationOrgan.prepare_200hz_car_window(raw_np)
            else:
                x_fm_200, ext_fm = np.zeros((41, 800), dtype=np.float32), False

            if run_cd and self._codebrain_organ is not None:
                p_cd = self._codebrain_organ.step_prepared_200hz(x_fm_200, is_extreme=ext_fm)
                s2c_replace = self._fusion_cfg.g1_clean_s2c_replacement
                if s2c_replace and self._codebrain_organ.last_p_s2c is not None:
                    p_cd = self._codebrain_organ.last_p_s2c
                g_cd = self._g1_gate_codebrain.current_reliability_weight()
                self._g1_gate_codebrain.observe_agreement(p_base_fm, p_cd, is_clean_window=True)
                w_cd = self._fm_cfg.weight_codebrain
                lp_fm = lp_fm + (w_cd * g_cd) * self.centered_log_probabilities_1d(p_cd)
                if (not s2c_replace) and self._codebrain_organ.last_p_s2c is not None:
                    p_s2c = self._codebrain_organ.last_p_s2c
                    g_s2c = g_cd if (is_bad_win or ext_fm or g_cd <= 0.30) else 1.0
                    lp_fm = lp_fm + (w_cd * g_s2c) * (
                        self.centered_log_probabilities_1d(p_s2c) - self.centered_log_probabilities_1d(p_cd)
                    )

            if run_cb and self._cbramod_organ is not None:
                p_cb = self._cbramod_organ.step_prepared_200hz(
                    x_fm_200,
                    is_extreme=ext_fm,
                    pseudo_chrono_u=cur_u,
                    is_artifact_flagged=is_bad_win,
                )
                g_cb = self._g1_gate_cbramod.current_reliability_weight()
                self._g1_gate_cbramod.observe_agreement(p_base_fm, p_cb, is_clean_window=True)
                lp_fm = lp_fm + (self._fm_cfg.weight_cbramod * g_cb) * self.centered_log_probabilities_1d(p_cb)

            if run_mr and self._mirepnet_organ is not None:
                p_mr = self._mirepnet_organ.step_score(
                    raw_np, is_artifact_flagged=is_bad_win, phi_nm_pre=cur_phi_nm
                )
                lp_fm = lp_fm + self._fm_cfg.weight_mirepnet * self.centered_log_probabilities_1d(p_mr)

            p_out = self.softmax_1d(lp_fm)

            # 9. S3D Fisher Anchored-Ridge Multimodal TTT Organ
            if (
                active_tier < 1
                and self._s3d_organ is not None
                and self._classical_cfg.enable_s3d
                and self._classical_cfg.weight_s3d > 0.0
                and self._anchor_mm_organ is not None
                and self._codebrain_organ is not None
                and self._cbramod_organ is not None
            ):
                p_s3d = self._s3d_organ.step_multimodal(
                    p_h9_t=p_out,
                    z_sr_t=self._anchor_mm_organ.last_z_sr,
                    z_ee_t=self._anchor_mm_organ.last_z_ee,
                    u16_cd_t=self._codebrain_organ.last_s3d_u16,
                    u16_cb_t=self._cbramod_organ.last_s3d_u16,
                    raw_window=raw_np,
                    is_artifact_flagged=is_bad_win,
                    phi_nm_pre=cur_phi_nm,
                )
                if p_s3d is not None:
                    p_out = self.softmax_1d(
                        self.centered_log_probabilities_1d(p_out)
                        + self._classical_cfg.weight_s3d * self.centered_log_probabilities_1d(p_s3d)
                    )

        return p_out

    def predict_proba(self, single_window_batch: np.ndarray) -> np.ndarray:
        """Return `(B, 3)` class probabilities by stepping each window causally (`B=1` online semantics)."""
        batch_np = (
            single_window_batch.detach().cpu().numpy()
            if torch.is_tensor(single_window_batch)
            else np.asarray(single_window_batch, dtype=np.float64)
        )
        if batch_np.ndim == 2:
            batch_np = batch_np[None, :, :]
        if batch_np.ndim != 3:
            raise ValueError(f"Expected 3D (B, C, T) batch, got shape {batch_np.shape}")
        if not np.isfinite(batch_np).all():
            batch_np = np.nan_to_num(batch_np, nan=0.0, posinf=500.0, neginf=-500.0)
        return np.stack([self.step_window(win_slice) for win_slice in batch_np], axis=0)

    @staticmethod
    def _synthesize_configuration_from_legacy_bundle(
        bundle_dir: Path,
        device_override: str | None = None,
    ) -> ExperimentRunConfiguration:
        merged_cfg: dict[str, Any] = {}
        for json_filename in ("eegpt_slot6.json", "r21_config.json", "r23_config.json", "r28_config.json", "r29_config.json"):
            candidate_p = bundle_dir / json_filename
            if candidate_p.is_file():
                merged_cfg.update(json.loads(candidate_p.read_text(encoding="utf-8")))

        exec_dev = device_override if device_override in ("auto", "cuda", "cpu") else "auto"
        dataset_cfg = DatasetSectionConfiguration(
            raw_download_directory=bundle_dir / "raw_downloads",
            extracted_cache_directory=bundle_dir / "extracted_caches",
            pretrained_checkpoint_directory=bundle_dir / "pretrained_checkpoints",
        )
        st_ckpts = tuple(int(c_v) for c_v in merged_cfg.get("mm_st_ckpts", [66, 126]))
        cnn_cfg = CnnEnsembleSectionConfiguration(
            self_train_checkpoints=st_ckpts,
            self_train_epoch_scale=float(merged_cfg.get("r29_st_epochs_scale", 0.5)),
            tent_buffer_length=int(merged_cfg.get("r31_tta_buffer", 64)),
            tent_freeze_window=int(merged_cfg.get("r31_tta_freeze", 130)),
            shallow_stoff_anchor_weight=float(merged_cfg.get("w_shallow_stoff", 0.06)),
        )
        w_cd = float(merged_cfg.get("r32_w_codebrain", 0.15))
        w_cb = float(merged_cfg.get("r32_w_cbramod", 0.10))
        w_mr = float(merged_cfg.get("r32_w_mirepnet", 0.10))
        fm_cfg = FoundationSectionConfiguration(
            enable_eegpt_bank=(bundle_dir / "base_eegpt_fp16.npz").is_file(),
            eegpt_lat52_seed_count=int(merged_cfg.get("n_lat52", 8)),
            eegpt_rms56_seed_count=int(merged_cfg.get("n_rms56", 3)),
            eegpt_clean4_seed_count=int(merged_cfg.get("n_clean4", 6)),
            eegpt_rms4c_seed_count=int(merged_cfg.get("n_rms4c", 2)),
            eegpt_refit_checkpoints=tuple(int(c_v) for c_v in merged_cfg.get("refit_at", [66, 126])),
            eegpt_alpha_clean4=float(merged_cfg.get("alpha_clean4", 0.35)),
            eegpt_alpha_rms4c=float(merged_cfg.get("alpha_rms4c", 0.65)),
            enable_codebrain=w_cd > 0.0 and (bundle_dir / "codebrain_41.npz").is_file(),
            weight_codebrain=w_cd,
            enable_cbramod=w_cb > 0.0 and (bundle_dir / "cbramod_41.npz").is_file(),
            weight_cbramod=w_cb,
            enable_mirepnet=w_mr > 0.0 and (bundle_dir / "mirepnet_fp16.npz").is_file(),
            weight_mirepnet=w_mr,
        )
        w_s3d = float(merged_cfg.get("r32_w_s3d", 0.12))
        gpu_heads = merged_cfg.get("r28_gpu_heads", ["anchor_mm"])
        amm_gpu = ("anchor_mm" in gpu_heads) if isinstance(gpu_heads, list) else bool(gpu_heads)
        classical_cfg = ClassicalHeadsSectionConfiguration(
            pseudochrono_ea_tau=float(merged_cfg.get("r32_pc_ea_tau", 1.3)),
            pseudochrono_bn_tau=float(merged_cfg.get("r32_pc_bn_tau", 1.3)),
            enable_tsr4=bool(merged_cfg.get("use_tsr", True)) and (bundle_dir / "tsr3task.npz").is_file(),
            tsr4_pc_tau=float(merged_cfg.get("r32_tsr_pc_tau", 1.5)),
            enable_rsnse=float(merged_cfg.get("w_rsnse", 0.08)) > 0.0 and (bundle_dir / "rsnse_t1.npz").is_file(),
            enable_anchor_mm=bool(merged_cfg.get("use_anchor_mm", True)) and (bundle_dir / "anchor_mm.npz").is_file(),
            anchor_mm_gpu_float64=amm_gpu,
            anchor_mm_lam_dm=float(merged_cfg.get("damper_lam_dm", 0.52)),
            anchor_mm_lam_proto=float(merged_cfg.get("damper_lam_proto", 0.13)),
            anchor_mm_c4_direct_lam=float(merged_cfg.get("r32_c4_direct_lam", 0.20)),
            enable_s3d=w_s3d > 0.0 and (bundle_dir / "s3d_fisher16_ancridge_2655.npz").is_file(),
            weight_s3d=w_s3d,
        )
        g_t1 = merged_cfg.get("r44_global_tier1_s", None)
        fusion_cfg = FusionPaceSectionConfiguration(
            trajectory_a_weight=float(merged_cfg.get("w_asym_a", 0.40)),
            traj_a_cnn_weight=float(merged_cfg.get("w_cnn_a", 0.75)),
            traj_a_lat52_weight=float(merged_cfg.get("w_lat52", 0.06)),
            traj_a_rms56_weight=float(merged_cfg.get("w_rms56", 0.12)),
            traj_a_tsr4_weight=float(merged_cfg.get("w_mortp", 0.07)),
            traj_b_lat52_weight=float(merged_cfg.get("w_lat52_b", 0.05)),
            traj_b_rms56_weight=float(merged_cfg.get("w_rms56_b", 0.05)),
            traj_b_rsnse_weight=float(merged_cfg.get("w_rsnse", 0.08)),
            balance_std_prior=float(merged_cfg.get("balance_prior", 10.0)),
            kalman_n0=float(merged_cfg.get("n0_kalman", 5.0)),
            kalman_q_drift=float(merged_cfg.get("q_drift", 0.002)),
            pc_sinkhorn_tau=float(merged_cfg.get("r32_pc_bal_tau", 1.3)),
            pc_sinkhorn_prior=float(merged_cfg.get("r32_final_bal_prior", 10.0)),
            pc_sinkhorn_iterations=int(merged_cfg.get("r32_pc_bal_sk_iter", 4)),
            g1_gate_enabled=bool(merged_cfg.get("r32_fm_rel_gate", True)),
            g1_gate_min=float(merged_cfg.get("r32_fm_rel_gmin", 0.25)),
            g1_gate_max=float(merged_cfg.get("r32_fm_rel_gmax", 1.00)),
            g1_clean_s2c_replacement=bool(merged_cfg.get("s7_fix1_s2c_g1", False)),
            pace_budget_per_window_s=float(merged_cfg.get("r31_budget_b_s", 0.29)),
            pace_fast_threshold_s=float(merged_cfg.get("r31_pace_hi_s", 0.30)),
            pace_credit_fast_f0_s=float(merged_cfg.get("r31_budget_F0_hi_s", 14.0)),
            pace_credit_slow_f0_s=float(merged_cfg.get("r31_budget_F0_s", 6.0)),
            pace_tier1_credit_s=float(merged_cfg.get("r31_tier05_credit_s", 1.2)),
            pace_tier2_credit_s=float(merged_cfg.get("r31_tier1_credit_s", 0.6)),
            pace_tier3_credit_s=float(merged_cfg.get("r31_tier2_credit_s", 0.2)),
            global_tier1_seconds=None if g_t1 is None else float(g_t1),
            global_tier2_seconds=float(merged_cfg.get("r32_global_tier2_s", 4300.0)),
            global_tier3_seconds=float(merged_cfg.get("r32_global_tier3_s", 4420.0)),
            tier1_keep_self_training=bool(merged_cfg.get("r44_tier1_keep_st", False)),
        )
        return ExperimentRunConfiguration(
            config_identifier=str(merged_cfg.get("version", "h10_bundle_decoder")),
            master_random_seed=20261121,
            output_run_directory=bundle_dir,
            execution_device=exec_dev,
            evaluation_orders=("O0", "W0", "W1", "W2", "W3", "W4"),
            dataset_section_configuration=dataset_cfg,
            cnn_ensemble_section_configuration=cnn_cfg,
            foundation_section_configuration=fm_cfg,
            classical_heads_section_configuration=classical_cfg,
            fusion_pace_section_configuration=fusion_cfg,
        )

    @classmethod
    def load_from_bundle_or_zip(
        cls,
        bundle_or_zip_path: str | Path,
        run_configuration: ExperimentRunConfiguration | None = None,
        device_override: str | None = None,
    ) -> H10MultimodalEnsembleDecoder:
        """Load a complete `H10MultimodalEnsembleDecoder` from an artifact directory or `.zip` bundle."""
        resolved_source = ParameterTypeValidator.require_path(
            bundle_or_zip_path, "bundle_or_zip_path", must_exist=True
        )
        temp_holder: tempfile.TemporaryDirectory[str] | None = None
        if resolved_source.is_file() and resolved_source.suffix.lower() == ".zip":
            with zipfile.ZipFile(resolved_source, "r") as zf_check:
                has_submission_py = "submission.py" in zf_check.namelist()
                for entry_str in zf_check.namelist():
                    if entry_str.endswith((".pkl", ".pickle", ".joblib")) or ".." in entry_str:
                        raise ValueError(f"Unsafe archive entry in {resolved_source}: {entry_str}")
            if has_submission_py:
                ArchiveSafetyTypeValidator(resolved_source).verify_contract()
            temp_holder = tempfile.TemporaryDirectory(prefix="neurips_bci_bundle_")
            bundle_dir = Path(temp_holder.name)
            with zipfile.ZipFile(resolved_source, "r") as zf_extract:
                zf_extract.extractall(bundle_dir)
        elif resolved_source.is_dir():
            bundle_dir = resolved_source
        else:
            raise ValueError(f"Expected directory or .zip archive, got {resolved_source}")

        if run_configuration is None:
            run_cfg_json = bundle_dir / "run_configuration.json"
            if run_cfg_json.is_file():
                run_configuration = ExperimentRunConfiguration.load_from_json_file(run_cfg_json)
            else:
                run_configuration = cls._synthesize_configuration_from_legacy_bundle(
                    bundle_dir, device_override=device_override
                )

        raw_dev = device_override if device_override is not None else run_configuration.execution_device
        resolved_dev_str = (
            ("cuda" if torch.cuda.is_available() else "cpu") if raw_dev == "auto" else str(raw_dev)
        )

        cnn_cfg = run_configuration.cnn_ensemble_section_configuration
        fm_cfg = run_configuration.foundation_section_configuration
        classical_cfg = run_configuration.classical_heads_section_configuration
        fusion_cfg = run_configuration.fusion_pace_section_configuration

        mix_json_path = bundle_dir / "mix.json"
        mix_meta = json.loads(mix_json_path.read_text(encoding="utf-8")) if mix_json_path.is_file() else {}
        art_prior = None
        if isinstance(mix_meta.get("art_guard"), dict):
            art_prior = mix_meta["art_guard"].get("prior")
        elif (bundle_dir / "artifact_prior.json").is_file():
            art_prior = json.loads((bundle_dir / "artifact_prior.json").read_text(encoding="utf-8"))

        artifact_adapter = CausalArtifactSignalAdapter.from_prior_dictionary(classical_cfg, art_prior)
        pace_gate_adapter = SessionPaceGateAdapter(fusion_cfg)

        pc_adapter: PseudoChronoSignalAdapter | None = None
        for pc_cand in ("pseudochrono_nm_mt6_d6.npz", "pseudochrono.npz"):
            if (bundle_dir / pc_cand).is_file():
                pc_adapter = PseudoChronoSignalAdapter.load_npz(
                    bundle_dir / pc_cand,
                    ea_tau=classical_cfg.pseudochrono_ea_tau,
                    bn_tau=classical_cfg.pseudochrono_bn_tau,
                )
                break

        cnn_organ = DualBlockCnnEnsembleCnnOrgan.load_from_bundle_directory(
            bundle_directory=bundle_dir,
            cnn_config=cnn_cfg,
            classical_config=classical_cfg,
            execution_device=resolved_dev_str,
        )

        eegpt_organ: QuadViewEegptFoundationOrgan | None = None
        if fm_cfg.enable_eegpt_bank and (bundle_dir / "base_eegpt_fp16.npz").is_file():
            eegpt_organ = QuadViewEegptFoundationOrgan.load_from_artifact_directory(
                artifact_directory=bundle_dir,
                n_lat52=fm_cfg.eegpt_lat52_seed_count,
                n_clean4=fm_cfg.eegpt_clean4_seed_count,
                n_rms4c=fm_cfg.eegpt_rms4c_seed_count,
                n_rms56=fm_cfg.eegpt_rms56_seed_count,
                execution_device=resolved_dev_str,
                refit_at=fm_cfg.eegpt_refit_checkpoints,
                alpha_clean4=fm_cfg.eegpt_alpha_clean4,
                alpha_rms4c=fm_cfg.eegpt_alpha_rms4c,
            )

        tsr4_organ: SubwindowGeodesicRiemannianOrgan | None = None
        if classical_cfg.enable_tsr4 and (bundle_dir / "tsr3task.npz").is_file():
            tsr4_organ = SubwindowGeodesicRiemannianOrgan.load_from_npz(
                bundle_dir / "tsr3task.npz",
                pc_tau=classical_cfg.tsr4_pc_tau,
            )

        rsnse_organ: RestSubspaceNuisanceRiemannianOrgan | None = None
        if classical_cfg.enable_rsnse and (bundle_dir / "rsnse_t1.npz").is_file():
            rsnse_organ = RestSubspaceNuisanceRiemannianOrgan.load_from_npz(bundle_dir / "rsnse_t1.npz")

        anchor_mm_organ: AnchorSpecialistMultimodalOrgan | None = None
        if classical_cfg.enable_anchor_mm and (bundle_dir / "anchor_mm.npz").is_file():
            amm_dev = resolved_dev_str if classical_cfg.anchor_mm_gpu_float64 else "cpu"
            anchor_mm_organ = AnchorSpecialistMultimodalOrgan.load_from_npz(
                bundle_dir / "anchor_mm.npz",
                execution_device=amm_dev,
            )

        codebrain_organ: CodebrainLoraFoundationOrgan | None = None
        if fm_cfg.enable_codebrain and (bundle_dir / "codebrain_41.npz").is_file():
            lora_p = bundle_dir / "s2c_codebrain_lora8_2655.npz"
            codebrain_organ = CodebrainLoraFoundationOrgan.load_from_npz(
                codebrain_npz_path=bundle_dir / "codebrain_41.npz",
                lora_npz_path=lora_p if (lora_p.is_file() and fm_cfg.codebrain_lora_seed_count > 0) else None,
                execution_device=resolved_dev_str,
            )

        cbramod_organ: CbramodReadoutFoundationOrgan | None = None
        if fm_cfg.enable_cbramod and (bundle_dir / "cbramod_41.npz").is_file():
            cbramod_organ = CbramodReadoutFoundationOrgan.load_from_npz(
                cbramod_npz_path=bundle_dir / "cbramod_41.npz",
                execution_device=resolved_dev_str,
            )

        mirepnet_organ: MirepnetSplineFoundationOrgan | None = None
        if (
            fm_cfg.enable_mirepnet
            and (bundle_dir / "mirepnet_fp16.npz").is_file()
            and (bundle_dir / "s3c_mirepnet_readouts_2655.npz").is_file()
        ):
            mirepnet_organ = MirepnetSplineFoundationOrgan.load_from_npz(
                mirepnet_fp16_npz_path=bundle_dir / "mirepnet_fp16.npz",
                readouts_npz_path=bundle_dir / "s3c_mirepnet_readouts_2655.npz",
                execution_device=resolved_dev_str,
            )

        s3d_organ: FisherAnchoredRidgeMultimodalOrgan | None = None
        if classical_cfg.enable_s3d and (bundle_dir / "s3d_fisher16_ancridge_2655.npz").is_file():
            s3d_organ = FisherAnchoredRidgeMultimodalOrgan.load_from_npz(
                bundle_dir / "s3d_fisher16_ancridge_2655.npz"
            )

        decoder_instance = cls(
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
        decoder_instance._managed_temp_directory = temp_holder
        return decoder_instance
