"""Unit, numerical parity (`r45/submission.py`), session-reset, and B=1 causality tests for NeurIPS BCI."""

from __future__ import annotations

import gc
import importlib.util
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from pathlib import Path
import sys
import zipfile
import numpy as np
import pytest
import torch

torch.set_num_threads(1)

from neurips_bci.backbone.foundation_backbones.CudaGraphFoundationBackbone import (
    CudaGraphFoundationBackbone,
)
from neurips_bci.backbone.foundation_backbones.MirepnetFoundationBackbone import (
    MirepnetFoundationBackbone,
)
from neurips_bci.configuration.run_configurations.ExperimentRunConfiguration import (
    ExperimentRunConfiguration,
)
from neurips_bci.decoder.ensemble_decoders.H10MultimodalEnsembleDecoder import (
    H10MultimodalEnsembleDecoder,
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


LOCKED_R45_ZIP_PATH = Path(
    "/home/veronese/projects/neurips-bci/dist/v2_r45_sub7_tier0_safe_guard_LOCKED_6bba611a.zip"
)
SMOKE_CONFIG_PATH = Path("/home/veronese/projects/neurips-bci/deliverables/configs/h10_smoke_test.json")


@pytest.fixture(scope="module")
def extracted_r45_bundle_and_module(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, object]:
    if not LOCKED_R45_ZIP_PATH.is_file():
        pytest.skip(f"Locked r45 bundle zip not found at {LOCKED_R45_ZIP_PATH}")
    os.environ["R31_FORCE_DET"] = "1"
    bundle_dir = tmp_path_factory.mktemp("r45_locked_bundle")
    with zipfile.ZipFile(LOCKED_R45_ZIP_PATH, "r") as zf:
        zf.extractall(bundle_dir)

    sub_py = bundle_dir / "submission.py"
    spec = importlib.util.spec_from_file_location("_r45_ref_submission", sub_py)
    assert spec is not None and spec.loader is not None
    ref_mod = importlib.util.module_from_spec(spec)
    sys.modules["_r45_ref_submission"] = ref_mod
    spec.loader.exec_module(ref_mod)
    return bundle_dir, ref_mod


@pytest.fixture(scope="module")
def synthetic_eeg_session() -> np.ndarray:
    rng = np.random.default_rng(20261124)
    t_grid = np.linspace(0.0, 4.0, 480, endpoint=False, dtype=np.float64)
    windows = []
    for w_idx in range(8):
        alpha_wave = 12.0 * np.sin(2.0 * np.pi * (9.0 + 0.5 * w_idx) * t_grid)[None, :]
        beta_wave = 6.0 * np.cos(2.0 * np.pi * (18.0 + 0.3 * w_idx) * t_grid)[None, :]
        spatial_gain = rng.standard_normal((41, 1))
        noise = 8.0 * rng.standard_normal((41, 480))
        win_arr = spatial_gain * alpha_wave + 0.5 * spatial_gain * beta_wave + noise
        windows.append(win_arr.astype(np.float64))
    return np.stack(windows, axis=0)


def test_quad_view_eegpt_organ_cpu_parity(
    extracted_r45_bundle_and_module: tuple[Path, object],
    synthetic_eeg_session: np.ndarray,
) -> None:
    bundle_dir, ref_mod = extracted_r45_bundle_and_module
    dev = torch.device("cpu")

    organ = QuadViewEegptFoundationOrgan.load_from_artifact_directory(
        artifact_directory=bundle_dir,
        n_lat52=1,
        n_clean4=1,
        n_rms4c=1,
        n_rms56=1,
        execution_device=dev,
        alpha_clean4=0.35,
        alpha_rms4c=0.65,
        eegpt_mode="ee7_pcearms_pcrec_t58",
    )

    with np.load(bundle_dir / "base_eegpt_fp16.npz", allow_pickle=False) as zb:
        base_fp16 = {k_s: zb[k_s].copy() for k_s in zb.files}
    with np.load(bundle_dir / "eegpt_align_priors.npz", allow_pickle=False) as zp:
        r0_52 = zp["R0_52_broad"].copy()
        r0_var = zp["r0_var"].copy()

    lat_m, lat_mu, lat_rep = ref_mod._load_slot6_eegpt_seed(bundle_dir / "lat52_s0.npz", base_fp16, "lat52", dev)
    c4_m, c4_mu, c4_rep = ref_mod._load_slot6_eegpt_seed(bundle_dir / "eegpt_clean4_s0.npz", base_fp16, "clean4", dev)
    r4c_m, r4c_mu, r4c_rep = ref_mod._load_slot6_eegpt_seed(bundle_dir / "eegpt_4corp_nodreyer_s0.npz", base_fp16, "rms4c", dev)
    rms_m, rms_mu, rms_rep = ref_mod._load_slot6_eegpt_seed(bundle_dir / "rms56_s0.npz", base_fp16, "rms56", dev)

    ref_bank = ref_mod.EEGPTStreamBank(
        [lat_m], [lat_mu], [lat_rep],
        [rms_m], [rms_mu], [rms_rep],
        R0_52_broad=r0_52, r0_var=r0_var,
        device=dev,
        clean4_models=[c4_m], clean4_mus=[c4_mu], clean4_reps=[c4_rep], alpha_clean4=0.35,
        rms4c_models=[r4c_m], rms4c_mus=[r4c_mu], rms4c_reps=[r4c_rep], alpha_rms4c=0.65,
    )
    with np.load(bundle_dir / "pseudochrono_nm_mt6_d6.npz", allow_pickle=False) as zpc:
        pc_spec = {
            "mu_phi": zpc["mu_phi"].astype(np.float64),
            "sd_phi": zpc["sd_phi"].astype(np.float64),
            "U_run": zpc["U_run"].astype(np.float64),
            "feat_mode": str(zpc["feat_mode"].item()),
            "center_u": bool(int(zpc["center_u"].item())),
        }
    ref_pc = ref_mod._R32PseudoChronoSession(pc_spec, ea_tau=1.3, bn_tau=1.3)
    ref_bank._r32_eegpt_mode = "ee7_pcearms_pcrec_t58"
    ref_bank._r32_pc_ref = ref_pc

    with np.load(bundle_dir / "anchor_mm.npz", allow_pickle=False) as zamm:
        pop_inv_std0 = zamm["inv_std0_id"].astype(np.float64)

    for w_idx in range(4):
        win = synthetic_eeg_session[w_idx]
        bad_flag = w_idx == 2
        cur_u = ref_pc.begin_window(win)
        ref_lat, ref_rms = ref_bank.step(win, flagged=bad_flag, pop_inv_std0_id=pop_inv_std0)
        our_lat, our_rms = organ.step_quad_views(
            win, is_artifact_flagged=bad_flag, pop_inv_std0_id=pop_inv_std0, pseudo_chrono_u=cur_u
        )
        assert np.max(np.abs(our_lat - ref_lat)) < 1e-6
        assert np.max(np.abs(our_rms - ref_rms)) < 1e-6
        assert np.max(np.abs(organ.last_u_rms_leg_a - ref_bank.last_u_rms_leg_a)) < 1e-6
        assert organ.last_all_f_causal is not None and organ.last_all_f_pop is not None
        assert np.max(np.abs(organ.last_all_f_causal - ref_bank.last_all_f_causal)) < 1e-5
        assert np.max(np.abs(organ.last_all_f_pop - ref_bank.last_all_f_pop)) < 1e-5


def test_codebrain_cbramod_mirepnet_cpu_parity(
    extracted_r45_bundle_and_module: tuple[Path, object],
    synthetic_eeg_session: np.ndarray,
) -> None:
    bundle_dir, ref_mod = extracted_r45_bundle_and_module
    dev = torch.device("cpu")

    s3d_organ = FisherAnchoredRidgeMultimodalOrgan.load_from_npz(bundle_dir / "s3d_fisher16_ancridge_2655.npz")
    cd_organ = CodebrainLoraFoundationOrgan.load_from_npz(
        bundle_dir / "codebrain_41.npz",
        lora_npz_path=bundle_dir / "s2c_codebrain_lora8_2655.npz",
        execution_device=dev,
    )
    cd_organ.attach_s3d_projection(s3d_organ.q_cd)

    cb_organ = CbramodReadoutFoundationOrgan.load_from_npz(
        bundle_dir / "cbramod_41.npz",
        execution_device=dev,
    )
    cb_organ.attach_s3d_projection(s3d_organ.q_cb)

    mr_organ = MirepnetSplineFoundationOrgan.load_from_npz(
        bundle_dir / "mirepnet_fp16.npz",
        bundle_dir / "s3c_mirepnet_readouts_2655.npz",
        execution_device=dev,
    )

    ref_cd = ref_mod._R32FrozenFMOnlineHead.load_codebrain_npz(
        bundle_dir / "codebrain_41.npz", dev, s2c_lora8_path=bundle_dir / "s2c_codebrain_lora8_2655.npz"
    )
    ref_cd.attach_s3d_proj(s3d_organ.q_cd, "cd")

    ref_cb = ref_mod._R32FrozenFMOnlineHead.load_cbramod_npz(bundle_dir / "cbramod_41.npz", dev)
    ref_cb.attach_s3d_proj(s3d_organ.q_cb, "cb")

    ref_mr = ref_mod._R32MIRepNetOnlineHead.load_npz(
        bundle_dir / "mirepnet_fp16.npz", bundle_dir / "s3c_mirepnet_readouts_2655.npz", dev
    )

    for w_idx in range(4):
        win = synthetic_eeg_session[w_idx].copy()
        if w_idx == 3:
            win[0, 0] = 6500.0  # exercise extreme-amplitude gating

        x200_ref, ext_ref = ref_mod._R32FrozenFMOnlineHead.prep_input_200hz(win)
        p_cd_ref = ref_cd.step_prepped(x200_ref, ext_ref)
        p_cb_ref = ref_cb.step_prepped(x200_ref, ext_ref)
        p_mr_ref = ref_mr.step_score(win, bad_bool=(w_idx == 2))

        p_cd_our = cd_organ.step_raw_window(win)
        p_cb_our = cb_organ.step_raw_window(win)
        p_mr_our = mr_organ.step_score(win, is_artifact_flagged=(w_idx == 2))

        assert np.max(np.abs(p_cd_our - p_cd_ref)) < 1e-5
        assert cd_organ.last_p_s2c is not None and ref_cd.last_p_s2c is not None
        assert np.max(np.abs(cd_organ.last_p_s2c - ref_cd.last_p_s2c)) < 1e-5
        assert np.max(np.abs(p_cb_our - p_cb_ref)) < 1e-5
        assert np.max(np.abs(p_mr_our - p_mr_ref)) < 1e-5

        if w_idx < 3:
            assert cd_organ.last_s3d_u16 is not None and ref_cd.last_s3d_u16 is not None
            assert cb_organ.last_s3d_u16 is not None and ref_cb.last_s3d_u16 is not None
            assert np.max(np.abs(cd_organ.last_s3d_u16 - ref_cd.last_s3d_u16)) < 1e-5
            assert np.max(np.abs(cb_organ.last_s3d_u16 - ref_cb.last_s3d_u16)) < 1e-5
        else:
            assert cd_organ.last_s3d_u16 is None and ref_cd.last_s3d_u16 is None
            assert cb_organ.last_s3d_u16 is None and ref_cb.last_s3d_u16 is None

    # Verify CudaGraphFoundationBackbone CPU eager fallback wrapper
    mr_bb = MirepnetFoundationBackbone.from_fp16_npz(bundle_dir / "mirepnet_fp16.npz", execution_device="cpu")
    cg_bb = CudaGraphFoundationBackbone(mr_bb)
    probe_x = torch.randn(1, 45, 1000, dtype=torch.float32)
    with torch.no_grad():
        tok_eager, mean_eager = mr_bb.encode_window_features(probe_x)
        tok_cg, mean_cg = cg_bb.encode_window_features(probe_x)  # type: ignore[misc]
    assert torch.max(torch.abs(tok_eager - tok_cg)).item() == 0.0
    assert torch.max(torch.abs(mean_eager - mean_cg)).item() == 0.0


def test_anchor_mm_and_s3d_cpu_parity(
    extracted_r45_bundle_and_module: tuple[Path, object],
    synthetic_eeg_session: np.ndarray,
) -> None:
    bundle_dir, ref_mod = extracted_r45_bundle_and_module

    amm_organ = AnchorSpecialistMultimodalOrgan.load_from_npz(
        bundle_dir / "anchor_mm.npz", execution_device="cpu"
    )
    ref_amm = ref_mod.AnchorMMOnlineHead.load(bundle_dir / "anchor_mm.npz")
    with np.load(bundle_dir / "pseudochrono_nm_mt6_d6.npz", allow_pickle=False) as zpc:
        pc_spec = {
            "mu_phi": zpc["mu_phi"].astype(np.float64),
            "sd_phi": zpc["sd_phi"].astype(np.float64),
            "U_run": zpc["U_run"].astype(np.float64),
            "feat_mode": str(zpc["feat_mode"].item()),
            "center_u": bool(int(zpc["center_u"].item())),
        }
    ref_pc = ref_mod._R32PseudoChronoSession(pc_spec, ea_tau=1.3, bn_tau=1.3)
    ref_amm._r32_pc_ref = ref_pc
    ref_amm._r32_zsr_pc_tau = 1.5
    for br_obj in ref_amm.branches.values():
        br_obj._r32_pc_ref = ref_pc
        br_obj._r32_rec_mode = "pc"
        br_obj._r32_pc_tau = 1.5
        br_obj._r32_n0_glob = 0.0
        br_obj._r32_n0_prior = 8.0
        br_obj._r32_bal_mode = "raw_u"
        br_obj._r32_min_router_n = 4
        br_obj._r32_ramp_n0 = 6.0
    ref_amm.reset()

    s3d_organ = FisherAnchoredRidgeMultimodalOrgan.load_from_npz(bundle_dir / "s3d_fisher16_ancridge_2655.npz")
    ref_s3d = ref_mod._R32S3DFisherAncRidgeHead.load_npz(bundle_dir / "s3d_fisher16_ancridge_2655.npz")

    rng = np.random.default_rng(90210)
    for w_idx in range(6):
        win = synthetic_eeg_session[w_idx]
        bad_flag = w_idx == 2
        cur_u = ref_pc.begin_window(win)

        f_causal = rng.standard_normal((19, 512)).astype(np.float64) * 0.4
        f_pop = rng.standard_normal((19, 512)).astype(np.float64) * 0.4
        tchr_p = rng.dirichlet([2.0, 1.5, 1.0]).astype(np.float64)
        u16_cd = rng.standard_normal(16).astype(np.float64) * 0.25
        u16_cb = rng.standard_normal(16).astype(np.float64) * 0.25

        ref_out = ref_amm.step(
            x_raw_f64=win,
            x_clean_f64=win,
            is_bad_bool=bad_flag,
            p_tchr=tchr_p,
            all_f_causal=f_causal,
            all_f_pop=f_pop,
            n_lat=8,
            n_clean4=6,
            n_rms4c=2,
        )
        our_out = amm_organ.step_multimodal(
            raw_window_f64=win,
            cleaned_window_f64=win,
            is_artifact_flagged=bad_flag,
            teacher_probabilities=tchr_p,
            all_f_causal=f_causal,
            all_f_pop=f_pop,
            n_lat=8,
            n_clean4=6,
            n_rms4c=2,
            pseudo_chrono_u=cur_u,
        )
        assert np.max(np.abs(our_out["res_dualmod"] - ref_out["res_dualmod"])) < 1e-9
        assert np.max(np.abs(our_out["res_proto_st"] - ref_out["res_proto_st"])) < 1e-9
        assert np.max(np.abs(our_out["res_dm80_protost20"] - ref_out["res_dm80_protost20"])) < 1e-9
        assert amm_organ.last_z_sr is not None and amm_organ.last_z_ee is not None
        assert np.max(np.abs(amm_organ.last_z_sr - ref_amm._r32_last_z_sr)) < 1e-9
        assert np.max(np.abs(amm_organ.last_z_ee - ref_amm._r32_last_z_ee)) < 1e-9

        p_s3d_ref = ref_s3d.step(
            tchr_p,
            ref_amm._r32_last_z_sr,
            ref_amm._r32_last_z_ee,
            u16_cd,
            u16_cb,
            win,
            bad_bool=bad_flag,
        )
        p_s3d_our = s3d_organ.step_multimodal(
            p_h9_t=tchr_p,
            z_sr_t=amm_organ.last_z_sr,
            z_ee_t=amm_organ.last_z_ee,
            u16_cd_t=u16_cd,
            u16_cb_t=u16_cb,
            raw_window=win,
            is_artifact_flagged=bad_flag,
        )
        assert p_s3d_our is not None and p_s3d_ref is not None
        assert np.max(np.abs(p_s3d_our - p_s3d_ref)) < 1e-10


def test_h10_decoder_reset_invariance_causality_and_r45_parity(
    extracted_r45_bundle_and_module: tuple[Path, object],
    synthetic_eeg_session: np.ndarray,
    tmp_path: Path,
) -> None:
    import json

    bundle_dir, ref_mod = extracted_r45_bundle_and_module
    os.environ["R31_FORCE_DET"] = "1"

    # Build a symlinked 4-seed EEGPT view of the r45 bundle so CPU login-node cgroups (<8 GiB RAM)
    # exercise all 13 organs and all 4 EEGPT stream views without allocating 19x2 float32 transformers.
    mini_bundle_dir = tmp_path / "r45_4seed_bundle"
    mini_bundle_dir.mkdir(parents=True, exist_ok=True)
    for asset_path in bundle_dir.iterdir():
        if asset_path.name == "eegpt_slot6.json":
            slot6_cfg = json.loads(asset_path.read_text(encoding="utf-8"))
            slot6_cfg.update({"n_lat52": 1, "n_clean4": 1, "n_rms4c": 1, "n_rms56": 1})
            (mini_bundle_dir / "eegpt_slot6.json").write_text(json.dumps(slot6_cfg), encoding="utf-8")
        else:
            (mini_bundle_dir / asset_path.name).symlink_to(asset_path)

    # 1. Reduced smoke-configuration load, reset_state() session invariance, and B=1 strict causality
    smoke_cfg = ExperimentRunConfiguration.load_from_json_file(SMOKE_CONFIG_PATH)
    smoke_decoder = H10MultimodalEnsembleDecoder.load_from_bundle_or_zip(
        bundle_or_zip_path=mini_bundle_dir,
        run_configuration=smoke_cfg,
        device_override="cpu",
    )

    sess_a = synthetic_eeg_session[:2]
    sess_b = synthetic_eeg_session[2:4] * 1.35 + 4.0

    probs_a_first = smoke_decoder.predict_proba(sess_a)
    assert probs_a_first.shape == (2, 3)
    assert np.all(np.isfinite(probs_a_first))
    assert np.allclose(probs_a_first.sum(axis=-1), 1.0, atol=1e-7)

    _ = smoke_decoder.predict_proba(sess_b)
    smoke_decoder.reset_state()
    probs_a_after_reset = smoke_decoder.predict_proba(sess_a)
    assert float(np.max(np.abs(probs_a_first - probs_a_after_reset))) == 0.0

    # Strict B=1 causality test (future_leak == 0.0 when suffix windows 2..3 are sabotaged)
    smoke_decoder.reset_state()
    clean_stream = synthetic_eeg_session[:4].copy()
    sabotaged_stream = synthetic_eeg_session[:4].copy()
    sabotaged_stream[2:] = -9500.0 * sabotaged_stream[2:] + 4200.0

    probs_clean = np.vstack([smoke_decoder.predict_proba(clean_stream[i : i + 1]) for i in range(4)])
    smoke_decoder.reset_state()
    probs_sabotaged = np.vstack([smoke_decoder.predict_proba(sabotaged_stream[i : i + 1]) for i in range(4)])
    future_leak = float(np.max(np.abs(probs_clean[:2] - probs_sabotaged[:2])))
    assert future_leak == 0.0
    del smoke_decoder
    gc.collect()

    # 2. Full-stack end-to-end parity against r45 Slot6DualEEGPTMix on real bundle weights
    full_decoder = H10MultimodalEnsembleDecoder.load_from_bundle_or_zip(
        bundle_or_zip_path=mini_bundle_dir,
        device_override="cpu",
    )
    full_decoder.reset_state()
    our_full_probs = full_decoder.predict_proba(synthetic_eeg_session[:2])
    del full_decoder
    gc.collect()

    ref_full = ref_mod.Slot6DualEEGPTMix.load(mini_bundle_dir, device="cpu")
    ref_full.cfg["sess_pace_guard"] = False
    ref_full.reset_state()
    ref_full_probs = ref_full.predict_proba(synthetic_eeg_session[:2])
    del ref_full
    gc.collect()

    assert our_full_probs.shape == (2, 3)
    assert np.allclose(our_full_probs.sum(axis=-1), 1.0, atol=1e-7)
    assert float(np.max(np.abs(our_full_probs - ref_full_probs))) < 1e-4
