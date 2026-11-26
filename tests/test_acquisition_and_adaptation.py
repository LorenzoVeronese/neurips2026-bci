"""Comprehensive unit tests for Layer 4 (Acquisition) and Layer 5 (Adaptation)."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile
import numpy as np
import pytest

from neurips_bci.acquisition import (
    CheckpointAcquirer,
    CognitiveAuxiliaryCorpusAcquirer,
    CorpusAcquirer,
    DreyerZenodoCorpusAcquirer,
    HuggingFaceCheckpointAcquirer,
    ProteusFoldSplitAcquirer,
    ProteusNemarCorpusAcquirer,
)
from neurips_bci.adaptation import (
    AsymmetricConfidenceGateAdapter,
    CausalArtifactSignalAdapter,
    ExcessAgreementGateAdapter,
    KalmanPriorProbabilityAdapter,
    PseudoChronoSignalAdapter,
    PseudoChronoSinkhornProbabilityAdapter,
    RunningPriorProbabilityAdapter,
    SessionPaceGateAdapter,
)
from neurips_bci.configuration.section_configurations import (
    ClassicalHeadsSectionConfiguration,
    DatasetSectionConfiguration,
    FusionPaceSectionConfiguration,
)
from neurips_bci.container.stream_containers import WindowedCorpusStreamContainer
from neurips_bci.operator.matrix_operators import CovarianceMatrixOperator, SubspaceMatrixOperator
from neurips_bci.operator.signal_operators import SpectralSignalOperator


LOCKED_R45_ZIP_PATH = Path(
    "/home/veronese/projects/neurips-bci/dist/v2_r45_sub7_tier0_safe_guard_LOCKED_6bba611a.zip"
)
CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
SIX_MOTOR_BANDS_HZ = [[4.0, 8.0], [8.0, 12.0], [12.0, 16.0], [16.0, 24.0], [24.0, 32.0], [32.0, 40.0]]


def test_synthetic_smoke_corpus_and_fold_splits(tmp_path: Path) -> None:
    smoke_npz = tmp_path / "smoke_proteus.npz"
    smoke_corpus = ProteusNemarCorpusAcquirer.generate_synthetic_smoke_corpus(smoke_npz, random_seed=20261123)

    assert smoke_npz.is_file()
    assert smoke_corpus.window_count() == 144
    assert smoke_corpus.eeg_windows.shape == (144, 41, 480)
    assert set(smoke_corpus.class_labels.tolist()) == {0, 1, 2}
    assert set(smoke_corpus.interface_contexts.tolist()) == {"Graz", "BrainHero"}
    assert len(set(smoke_corpus.participant_ids.tolist())) == 4
    assert len(set(smoke_corpus.session_ids.tolist())) == 6
    assert smoke_corpus.resting_windows is not None
    assert smoke_corpus.resting_windows.shape == (24, 41, 480)

    # Verify strict allow_pickle=False persistence roundtrip
    with np.load(smoke_npz, allow_pickle=False) as raw_archive:
        assert "X" in raw_archive and "rest_X" in raw_archive
    reloaded = WindowedCorpusStreamContainer.load_npz_archive(smoke_npz)
    assert np.array_equal(reloaded.eeg_windows, smoke_corpus.eeg_windows)
    assert np.array_equal(reloaded.class_labels, smoke_corpus.class_labels)
    assert reloaded.resting_windows is not None
    assert np.array_equal(reloaded.resting_windows, smoke_corpus.resting_windows)

    # Verify automatic smoke CV fold adaptation + zero leakage + O0 / W0..W4 determinism
    split_acquirer = ProteusFoldSplitAcquirer(reloaded)
    smoke_folds = split_acquirer.acquire()
    assert len(smoke_folds) == 8  # 4 LOSO + 2 XS4 + 2 XS1
    assert [f.fold_regime for f in smoke_folds] == [
        "loso", "loso", "loso", "loso",
        "xsess4", "xsess4",
        "xsess1", "xsess1",
    ]
    for fold_idx, fold_obj in enumerate(smoke_folds):
        assert fold_obj.fold_index == fold_idx
        assert np.intersect1d(fold_obj.training_indices, fold_obj.testing_indices).size == 0
        for order_tag in ("O0", "W0", "W1", "W2", "W3", "W4"):
            streams_first = fold_obj.build_ordered_test_streams(reloaded, order_tag)
            streams_second = fold_obj.build_ordered_test_streams(reloaded, order_tag)
            assert len(streams_first) == len(streams_second) >= 1
            for s1, s2 in zip(streams_first, streams_second):
                assert np.array_equal(s1.ordered_windows, s2.ordered_windows)
                assert np.array_equal(s1.ordered_labels, s2.ordered_labels)
        w0_win = fold_obj.build_ordered_test_streams(reloaded, "W0")[0].ordered_windows
        w1_win = fold_obj.build_ordered_test_streams(reloaded, "W1")[0].ordered_windows
        assert not np.array_equal(w0_win, w1_win)

    warmup_fold = split_acquirer.build_official_warmup_fold()
    assert warmup_fold.fold_regime == "warmup"
    assert np.intersect1d(warmup_fold.training_indices, warmup_fold.testing_indices).size == 0
    train_subjs = set(reloaded.participant_ids[warmup_fold.training_indices].tolist())
    test_subjs = set(reloaded.participant_ids[warmup_fold.testing_indices].tolist())
    assert train_subjs.isdisjoint(test_subjs)


def test_full_16_fold_topology_on_10_participant_metadata() -> None:
    rng = np.random.default_rng(42)
    sess_plan = (
        [(f"Dreyer2026Proteus/P{i:02d}", ("1",)) for i in (1, 2, 3, 5, 6, 7, 8, 9)]
        + [("Dreyer2026Proteus/P04", ("1", "2", "3")), ("Dreyer2026Proteus/P10", ("1", "2", "3"))]
    )
    participants, sessions, interfaces, labels = [], [], [], []
    for subj_id, sess_nums in sorted(sess_plan):
        for s_num in sess_nums:
            for w_i in range(6):
                participants.append(subj_id)
                sessions.append(f"{subj_id}/{s_num}")
                interfaces.append("Graz" if w_i < 3 else "BrainHero")
                labels.append(w_i % 3)
    total_n = len(labels)
    full_mock = WindowedCorpusStreamContainer(
        corpus_identifier="MockProteus10",
        eeg_windows=rng.standard_normal((total_n, 41, 480)).astype(np.float32),
        class_labels=np.array(labels, dtype=np.int64),
        participant_ids=np.array(participants, dtype="U32"),
        session_ids=np.array(sessions, dtype="U64"),
        interface_contexts=np.array(interfaces, dtype="U32"),
        channel_labels=SubspaceMatrixOperator.PROTEUS_41_CHANNELS,
    )
    folds_16 = ProteusFoldSplitAcquirer(full_mock).acquire()
    assert len(folds_16) == 16
    assert [f.fold_regime for f in folds_16[:10]] == ["loso"] * 10
    assert [f.fold_regime for f in folds_16[10:14]] == ["xsess4"] * 4
    assert [f.fold_regime for f in folds_16[14:16]] == ["xsess1"] * 2
    assert [f.fold_label for f in folds_16[10:16]] == ["P04/2", "P04/3", "P10/2", "P10/3", "P04/3*", "P10/3*"]
    # XS1 fold 14 (P04/3*) must exclude P04/2 from training!
    p04_s2_rows = np.flatnonzero(full_mock.session_ids == "Dreyer2026Proteus/P04/2")
    assert np.intersect1d(folds_16[14].training_indices, p04_s2_rows).size == 0


def test_acquirer_preprocessing_and_checkpoint_fp16_packing(tmp_path: Path) -> None:
    ds_cfg = DatasetSectionConfiguration(
        raw_download_directory=tmp_path / "raw",
        extracted_cache_directory=tmp_path / "cache",
        pretrained_checkpoint_directory=tmp_path / "ckpts",
    )
    dreyer_acq = DreyerZenodoCorpusAcquirer(ds_cfg)
    aux_acq = CognitiveAuxiliaryCorpusAcquirer(ds_cfg)
    hf_acq = HuggingFaceCheckpointAcquirer(ds_cfg)
    assert "zenodo.org" in dreyer_acq.acquirer_source
    assert "physionet.org" in aux_acq.acquirer_source
    assert "braindecode/eegpt-pretrained" in hf_acq.acquirer_source

    rng = np.random.default_rng(101)
    src_ch = ("Fp1", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2")
    raw_250hz = rng.standard_normal((len(src_ch), 1500)) * 15.0
    mapped_120hz = CorpusAcquirer.preprocess_continuous_recording_uv(
        continuous_microvolts=raw_250hz,
        source_sampling_rate_hz=250.0,
        source_channel_labels=src_ch,
        target_channel_labels=SubspaceMatrixOperator.PROTEUS_41_CHANNELS,
        target_sampling_rate_hz=120.0,
    )
    assert mapped_120hz.shape == (41, 720)
    assert mapped_120hz.dtype == np.float32
    assert np.isfinite(mapped_120hz).all()

    # Verify CheckpointAcquirer weight_norm folding + LM head stripping + allow_pickle=False fp16 save
    g_param = np.full((8, 1, 1), 2.0, dtype=np.float32)
    v_param = np.ones((8, 4, 1), dtype=np.float32)
    mock_sd = {
        "init_conv.parametrizations.weight.original0": g_param,
        "init_conv.parametrizations.weight.original1": v_param,
        "bn.running_mean": np.zeros(8, dtype=np.float32),
        "bn.num_batches_tracked": np.array(12, dtype=np.int64),
        "final_layer.weight": np.ones((3, 8), dtype=np.float32),
        "lm_head.weight": np.ones((100, 8), dtype=np.float32),
    }
    ckpt_npz = tmp_path / "ckpts" / "codebrain_base_fp16.npz"
    CheckpointAcquirer.write_fp16_checkpoint_npz(mock_sd, ckpt_npz, config_json_text='{"patch_size": 200}')
    with np.load(ckpt_npz, allow_pickle=False) as packed:
        assert "final_layer.weight" not in packed.files
        assert "lm_head.weight" not in packed.files
        assert "init_conv.weight" in packed.files
        assert packed["init_conv.weight"].dtype == np.float16
        assert packed["bn.running_mean"].dtype == np.float32
        # ||[1, 1, 1, 1]||_2 == 2.0, so g * (v / ||v||) == 2.0 * (1.0 / 2.0) == 1.0!
        assert np.allclose(packed["init_conv.weight"].astype(np.float32), 1.0, atol=1e-3)


def test_causal_artifact_signal_adapter_clean_nan_flat_and_spike() -> None:
    rng = np.random.default_rng(2026)
    train_wins = (rng.standard_normal((48, 41, 480)) * 12.0).astype(np.float32)
    prior_dict = CausalArtifactSignalAdapter.fit_training_prior(train_wins)
    heads_cfg = ClassicalHeadsSectionConfiguration()
    guard = CausalArtifactSignalAdapter.from_prior_dictionary(heads_cfg, prior_dict)

    # 1. Clean windows
    for _ in range(5):
        clean_in = rng.standard_normal((41, 480)) * 12.0
        cleaned_w, centered_w, is_bad = guard.sanitize_and_gate_window(clean_in)
        assert not is_bad
        assert np.isfinite(cleaned_w).all()
        assert np.allclose(centered_w.mean(axis=-1), 0.0, atol=1e-10)
        assert np.allclose(cleaned_w, clean_in, atol=1e-12)

    # 2. NaN windows (both partial NaN and full-NaN packet loss)
    partial_nan = rng.standard_normal((41, 480)) * 12.0
    partial_nan[3, 100:120] = np.nan
    cleaned_nan, centered_nan, bad_nan = guard.sanitize_and_gate_window(partial_nan)
    assert bad_nan is True
    assert np.isfinite(cleaned_nan).all() and np.isfinite(centered_nan).all()
    assert np.all(cleaned_nan[3] == 0.0)

    all_nan = np.full((41, 480), np.nan)
    cleaned_all_nan, centered_all_nan, bad_all_nan = guard.sanitize_and_gate_window(all_nan)
    assert bad_all_nan is True
    assert np.all(cleaned_all_nan == 0.0) and np.all(centered_all_nan == 0.0)

    # 3. Full flat dropout window + single dead-channel latch
    flat_win = np.ones((41, 480), dtype=np.float64) * 3.5
    cleaned_flat, _, bad_flat = guard.sanitize_and_gate_window(flat_win)
    assert bad_flat is True
    assert np.all(cleaned_flat == 0.0)

    dead_ch_win = rng.standard_normal((41, 480)) * 12.0
    dead_ch_win[7, :] = 0.0
    cleaned_dead, _, _ = guard.sanitize_and_gate_window(dead_ch_win)
    assert np.all(cleaned_dead[7] == 0.0)
    assert guard.dead_channel_mask is not None and bool(guard.dead_channel_mask[7])
    # Next normal window keeps channel 7 latched dead
    follow_win = rng.standard_normal((41, 480)) * 12.0
    cleaned_follow, _, _ = guard.sanitize_and_gate_window(follow_win)
    assert np.all(cleaned_follow[7] == 0.0)

    # 4. 5000 uV isolated spike clipping & 5000 uV broadband burst gating
    guard.reset_session_state()
    for _ in range(6):
        guard.sanitize_and_gate_window(rng.standard_normal((41, 480)) * 12.0)

    spike_win = rng.standard_normal((41, 480)) * 12.0
    spike_win[2, 240] = 5000.0
    cleaned_spike, centered_spike, bad_spike = guard.sanitize_and_gate_window(spike_win)
    assert bad_spike is True
    assert float(np.max(np.abs(cleaned_spike[2]))) < 250.0  # Clipped from 5000 uV to ~8 MAD (~96 uV)
    assert np.isfinite(centered_spike).all()

    burst_win = rng.standard_normal((41, 480)) * 5000.0
    _, _, bad_burst = guard.sanitize_and_gate_window(burst_win)
    assert bad_burst is True


def test_pseudochrono_signal_adapter_fit_roundtrip_and_r45_locked_npz(tmp_path: Path) -> None:
    smoke_npz = tmp_path / "smoke_pc.npz"
    smoke_corpus = ProteusNemarCorpusAcquirer.generate_synthetic_smoke_corpus(smoke_npz, random_seed=77)
    pc_adapter = PseudoChronoSignalAdapter.fit_from_corpus(smoke_corpus, subspace_dim=6, ea_tau=1.3, bn_tau=1.3)
    assert pc_adapter.mu_phi.shape == (246,)
    assert pc_adapter.sd_phi.shape == (246,)
    assert pc_adapter.u_run_basis.shape == (246, 6)

    saved_pc = pc_adapter.save_npz(tmp_path / "pc_roundtrip.npz")
    reloaded_pc = PseudoChronoSignalAdapter.load_npz(saved_pc, ea_tau=1.3, bn_tau=1.3)
    assert np.allclose(reloaded_pc.mu_phi, pc_adapter.mu_phi)
    assert np.allclose(reloaded_pc.sd_phi, pc_adapter.sd_phi)
    assert np.allclose(reloaded_pc.u_run_basis, pc_adapter.u_run_basis)

    # Load exact shipped r45 pseudochrono_nm_mt6_d6.npz from locked submission zip
    assert LOCKED_R45_ZIP_PATH.is_file(), f"Missing locked r45 zip at {LOCKED_R45_ZIP_PATH}"
    extracted_r45_pc = tmp_path / "pseudochrono_nm_mt6_d6.npz"
    with zipfile.ZipFile(LOCKED_R45_ZIP_PATH, "r") as zf:
        extracted_r45_pc.write_bytes(zf.read("pseudochrono_nm_mt6_d6.npz"))
    r45_pc = PseudoChronoSignalAdapter.load_npz(extracted_r45_pc, ea_tau=1.3, bn_tau=1.3)
    assert r45_pc.mu_phi.shape == (246,)
    assert r45_pc.sd_phi.shape == (246,)
    assert r45_pc.u_run_basis.shape == (246, 6)
    assert r45_pc.feat_mode == "NonMotor"

    # Verify causal B=1 EA + 5166-D tangent + BN moments + retro whitening + 0.0 prefix leak!
    ref_covs = SpectralSignalOperator.compute_band_covariances(
        smoke_corpus.eeg_windows[:16], SIX_MOTOR_BANDS_HZ, sampling_rate_hz=120.0
    ).mean(axis=0)
    assert ref_covs.shape == (7, 41, 41)

    sess_wins = [smoke_corpus.eeg_windows[i].astype(np.float64) for i in range(10)]
    tangents_clean: list[np.ndarray] = []
    bn_means_clean: list[np.ndarray] = []
    r45_pc.reset_session_state()
    for idx_t, win_t in enumerate(sess_wins):
        u_t = r45_pc.begin_window(win_t)
        assert u_t.shape == (6,)
        dc_t = win_t - win_t.mean(axis=-1, keepdims=True)
        covs_t, wh_t, tan_t = r45_pc.step_window_ea_and_tangent(
            raw_window=win_t,
            dc_centered_window=dc_t,
            is_bad_window=(idx_t == 3),
            is_frozen=False,
            reference_r0=ref_covs,
            bands_hz=SIX_MOTOR_BANDS_HZ,
            scale_mode="none",
        )
        assert covs_t.shape == (7, 41, 41)
        assert wh_t.shape == (41, 480)
        assert tan_t.shape == (5166,)
        mu_bn, var_bn = r45_pc.compute_weighted_bn_moments(
            current_mean=wh_t.mean(axis=-1),
            current_var=wh_t.var(axis=-1),
            prior_mean=np.zeros(41),
            prior_var=np.ones(41),
            group_key="cnn_group_0",
            is_bad_or_frozen=(idx_t == 3),
        )
        assert mu_bn.shape == (41,) and var_bn.shape == (41,)
        tangents_clean.append(tan_t)
        bn_means_clean.append(mu_bn)

    # Prefix-sabotage test at cut=6: corrupting windows 6..9 MUST give 0.0 diff on 0..5!
    sabotaged_wins = [w.copy() for w in sess_wins]
    rng_sab = np.random.default_rng(999)
    for j in range(6, 10):
        sabotaged_wins[j] = rng_sab.standard_normal((41, 480)) * 200.0
    r45_pc.reset_session_state()
    for idx_t, win_t in enumerate(sabotaged_wins):
        r45_pc.begin_window(win_t)
        dc_t = win_t - win_t.mean(axis=-1, keepdims=True)
        _, wh_t, tan_t = r45_pc.step_window_ea_and_tangent(
            raw_window=win_t,
            dc_centered_window=dc_t,
            is_bad_window=(idx_t == 3),
            is_frozen=False,
            reference_r0=ref_covs,
            bands_hz=SIX_MOTOR_BANDS_HZ,
        )
        mu_bn, _ = r45_pc.compute_weighted_bn_moments(
            current_mean=wh_t.mean(axis=-1),
            current_var=wh_t.var(axis=-1),
            prior_mean=np.zeros(41),
            prior_var=np.ones(41),
            group_key="cnn_group_0",
            is_bad_or_frozen=(idx_t == 3),
        )
        if idx_t < 6:
            assert float(np.max(np.abs(tan_t - tangents_clean[idx_t]))) == 0.0
            assert float(np.max(np.abs(mu_bn - bn_means_clean[idx_t]))) == 0.0

    retro_out = r45_pc.retro_whiten_session_windows(
        raw_session_windows=sess_wins[:6],
        bad_flags=[i == 3 for i in range(6)],
        broadband_r0=ref_covs[-1],
        scale_mode="zstd",
    )
    assert retro_out.shape == (6, 41, 480)
    assert np.isfinite(retro_out).all()


def test_probability_and_gate_adapters_and_r45_vs_r35_pace_guard() -> None:
    r45_json = json.loads((CONFIGS_DIR / "h10_r45_submission7.json").read_text(encoding="utf-8"))
    r35_json = json.loads((CONFIGS_DIR / "h10_r35_submission6.json").read_text(encoding="utf-8"))
    r45_pace_cfg = FusionPaceSectionConfiguration(**r45_json["fusion_pace_section_configuration"])
    r35_pace_cfg = FusionPaceSectionConfiguration(**r35_json["fusion_pace_section_configuration"])

    # 1. BalStd, BalKal, PC-Sinkhorn: simplex validity, imbalance suppression, and prefix-leak == 0.0
    bal_std = RunningPriorProbabilityAdapter.from_fusion_configuration(r45_pace_cfg)
    bal_kal = KalmanPriorProbabilityAdapter.from_fusion_configuration(r45_pace_cfg)
    pc_sk = PseudoChronoSinkhornProbabilityAdapter.from_fusion_configuration(r45_pace_cfg, subspace_dim=6)

    biased_stream = np.tile(np.array([0.65, 0.20, 0.15], dtype=np.float64), (25, 1))
    rng = np.random.default_rng(123)
    u_coords = rng.standard_normal((25, 6))

    out_std, out_kal, out_sk = [], [], []
    for idx_t in range(25):
        p_s = bal_std.adapt_probabilities(biased_stream[idx_t])
        p_k = bal_kal.adapt_probabilities(biased_stream[idx_t])
        p_pc = pc_sk.adapt_probabilities(biased_stream[idx_t], u_coordinate=u_coords[idx_t], is_bad_window=False)
        for p_vec in (p_s, p_k, p_pc):
            assert p_vec.shape == (3,)
            assert np.isclose(np.sum(p_vec), 1.0)
            assert np.all(p_vec > 0.0)
        out_std.append(p_s)
        out_kal.append(p_k)
        out_sk.append(p_pc)

    # Repeating a [0.65, 0.20, 0.15] biased prediction must pull class-0 down toward 1/3 over time
    assert out_std[-1][0] < out_std[0][0] and out_std[-1][0] < 0.45
    assert out_kal[-1][0] < out_kal[0][0] and out_kal[-1][0] < 0.45
    assert out_sk[-1][0] < out_sk[0][0] and out_sk[-1][0] < 0.48

    # 2. AsymmetricConfidenceGateAdapter (LowDM_Asym)
    asym_gate = AsymmetricConfidenceGateAdapter()
    # Start closed at t=0 -> tau_dm == hist_dm => g_dm == 0.0, g_pr == 0.0!
    g_dm_0, g_pr_0 = asym_gate.step_gates(np.array([0.40, 0.35, 0.25]), is_bad_window=False)
    assert g_dm_0 == 0.0 and g_pr_0 == 0.0
    # Uncertain stream opens both gates > 0, with proto ceiling (0.80) wider than dm ceiling (0.68)
    for _ in range(10):
        g_dm, g_pr = asym_gate.step_gates(np.array([0.45, 0.30, 0.25]), is_bad_window=False)
    assert g_dm > 0.5 and g_pr > 0.5
    # Overconfident window (>= tau_inst = 0.94) instantaneously zeroes both gates
    g_dm_hi, g_pr_hi = asym_gate.step_gates(np.array([0.96, 0.02, 0.02]), is_bad_window=False)
    assert g_dm_hi == 0.0 and g_pr_hi == 0.0

    # 3. ExcessAgreementGateAdapter (G1)
    g1_agree = ExcessAgreementGateAdapter.from_fusion_configuration(r45_pace_cfg)
    g1_anti = ExcessAgreementGateAdapter.from_fusion_configuration(r45_pace_cfg)
    w_init = g1_agree.current_reliability_weight()
    assert 0.25 < w_init < 1.0
    for _ in range(15):
        g1_agree.observe_agreement(np.array([0.80, 0.10, 0.10]), np.array([0.78, 0.12, 0.10]), is_clean_window=True)
        g1_anti.observe_agreement(np.array([0.80, 0.10, 0.10]), np.array([0.10, 0.80, 0.10]), is_clean_window=True)
    assert np.isclose(g1_agree.current_reliability_weight(), 1.0)
    assert np.isclose(g1_anti.current_reliability_weight(), 0.25)

    # 4. SessionPaceGateAdapter: r45 vs r35 credit + global clock + tier1_keep_self_training
    pace_r45 = SessionPaceGateAdapter(r45_pace_cfg)
    pace_r35 = SessionPaceGateAdapter(r35_pace_cfg)
    for _ in range(45):
        pace_r45.record_ordinary_window_duration(0.22)
        pace_r35.record_ordinary_window_duration(0.22)

    # At window 45, 0.22s/w is <= 0.30s (fast in r45 -> F0=14s) but > 0.195s (slow in r35 -> F0=6s)
    t_r45_0, skip_r45_0 = pace_r45.evaluate_window_tier(45, session_elapsed_s=10.0, run_elapsed_s=500.0, is_checkpoint_next=False)
    t_r35_0, skip_r35_0 = pace_r35.evaluate_window_tier(45, session_elapsed_s=10.0, run_elapsed_s=500.0, is_checkpoint_next=False)
    assert pace_r45.hardware_speed_mode == "fast"
    assert pace_r35.hardware_speed_mode == "slow"
    assert t_r45_0 == 0 and skip_r45_0 is False
    assert t_r35_0 == 0 and skip_r35_0 is False

    # At run_elapsed_s = 3600s:
    # - r45 has global_tier1_seconds=3550s AND tier1_keep_self_training=True -> tier=1, skip_self_training=False!
    # - r35 has global_tier1_seconds=None -> stays tier=0!
    t_r45_g1, skip_r45_g1 = pace_r45.evaluate_window_tier(65, session_elapsed_s=14.0, run_elapsed_s=3600.0, is_checkpoint_next=True)
    t_r35_g1, skip_r35_g1 = pace_r35.evaluate_window_tier(65, session_elapsed_s=14.0, run_elapsed_s=3600.0, is_checkpoint_next=True)
    assert t_r45_g1 == 1 and skip_r45_g1 is False
    assert t_r35_g1 == 0 and skip_r35_g1 is False

    # When r35 hits credit-based tier 1 (credit in [0.6, 1.2)), tier1_keep_self_training=False -> skip_self_training=True!
    # For r35 at window 66: F0=6.0 + 0.205 * 67 = 19.735; elapsed=18.8 -> credit=0.935 (tier 1)
    t_r35_t1, skip_r35_t1 = pace_r35.evaluate_window_tier(66, session_elapsed_s=18.8, run_elapsed_s=600.0, is_checkpoint_next=True)
    assert t_r35_t1 == 1 and skip_r35_t1 is True

    # Monotonicity check: even if elapsed drops/catches up inside same session, active_tier never decreases until reset
    t_r35_later, _ = pace_r35.evaluate_window_tier(80, session_elapsed_s=18.9, run_elapsed_s=601.0, is_checkpoint_next=False)
    assert t_r35_later == 1
    pace_r35.reset_session_state()
    t_r35_reset, _ = pace_r35.evaluate_window_tier(0, session_elapsed_s=0.05, run_elapsed_s=602.0, is_checkpoint_next=False)
    assert t_r35_reset == 0


def test_adaptation_modules_pass_archive_safety_source_audit() -> None:
    import ast

    forbidden_imports = {"urllib", "requests", "httpx", "aiohttp", "ftplib", "telnetlib", "pickle"}
    adaptation_dir = Path(__file__).resolve().parent.parent / "neurips_bci" / "adaptation"
    py_files = sorted(adaptation_dir.rglob("*.py"))
    assert len(py_files) >= 14
    for py_path in py_files:
        syntax_tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        for syntax_node in ast.walk(syntax_tree):
            if isinstance(syntax_node, ast.Import):
                for alias_item in syntax_node.names:
                    assert alias_item.name.split(".")[0] not in forbidden_imports
            elif isinstance(syntax_node, ast.ImportFrom) and syntax_node.module:
                assert syntax_node.module.split(".")[0] not in forbidden_imports
            elif isinstance(syntax_node, ast.keyword):
                if syntax_node.arg == "allow_pickle" and isinstance(syntax_node.value, ast.Constant):
                    assert syntax_node.value.value is not True
                if syntax_node.arg == "weights_only" and isinstance(syntax_node.value, ast.Constant):
                    assert syntax_node.value.value is not False

