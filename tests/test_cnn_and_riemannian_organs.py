"""Comprehensive parity, causality, reset-invariance, and fit->save->load roundtrip tests for Layer 6 & 7 organs."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import importlib.util
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pytest
import torch

torch.set_num_threads(1)

from neurips_bci.backbone.convolutional_backbones import (
    AdaptiveBatchNormConvolutionalBackbone,
    ConvolutionalBackbone,
    FilterBankConvolutionalBackbone,
    GroupedEnsembleConvolutionalBackbone,
    ShallowTemporalConvolutionalBackbone,
)
from neurips_bci.configuration.section_configurations.ClassicalHeadsSectionConfiguration import (
    ClassicalHeadsSectionConfiguration,
)
from neurips_bci.configuration.section_configurations.CnnEnsembleSectionConfiguration import (
    CnnEnsembleSectionConfiguration,
)
from neurips_bci.container.stream_containers.WindowedCorpusStreamContainer import WindowedCorpusStreamContainer
from neurips_bci.operator.matrix_operators.SubspaceMatrixOperator import SubspaceMatrixOperator
from neurips_bci.organ.cnn_organs import CnnOrgan, DualBlockCnnEnsembleCnnOrgan
from neurips_bci.organ.riemannian_organs import (
    ParticipantSpecialistRiemannianOrgan,
    RestSubspaceNuisanceRiemannianOrgan,
    RiemannianOrgan,
    SessionCspLdaRiemannianOrgan,
    SubwindowGeodesicRiemannianOrgan,
)


LOCKED_R45_ZIP = Path("/home/veronese/projects/neurips-bci/dist/v2_r45_sub7_tier0_safe_guard_LOCKED_6bba611a.zip")
R45_SUBMISSION_PY = Path(
    "/home/veronese/.gemini/antigravity-cli/brain/1171b0da-9f2d-412f-8132-fe04e8433f86/scratch/r45_ref/submission.py"
)


@pytest.fixture(scope="module")
def r45_ref_module():
    if not R45_SUBMISSION_PY.exists():
        pytest.skip(f"Missing reference r45 submission.py at {R45_SUBMISSION_PY}")
    spec = importlib.util.spec_from_file_location("r45_ref_submission", R45_SUBMISSION_PY)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def synthetic_test_windows() -> np.ndarray:
    rng = np.random.default_rng(20261124)
    t_axis = np.linspace(0.0, 4.0, 480, endpoint=False)
    windows = []
    for w_i in range(72):
        cls_k = w_i % 3
        freq_hz = (10.0, 18.0, 26.0)[cls_k]
        spatial_pat = np.sin(np.linspace(0.2, 3.0, 41) * (cls_k + 1))[:, None]
        carrier = np.cos(2.0 * np.pi * freq_hz * t_axis + 0.3 * w_i)[None, :]
        win = 6.0 * spatial_pat * carrier + 2.5 * rng.standard_normal((41, 480))
        windows.append(win.astype(np.float32))
    return np.stack(windows, axis=0)


def _make_synthetic_corpus(n_per_session: int = 18) -> WindowedCorpusStreamContainer:
    rng = np.random.default_rng(4242)
    t_axis = np.linspace(0.0, 4.0, 480, endpoint=False)
    all_x, all_y, all_part, all_sess, all_ctx = [], [], [], [], []
    session_specs = [
        ("P01", "P01/S1", "Graz"),
        ("P01", "P01/S2", "Neurow"),
        ("P02", "P02/S1", "Graz"),
        ("P03", "P03/S1", "Neurow"),
    ]
    for p_idx, (part_id, sess_id, ctx_str) in enumerate(session_specs):
        gain = 1.0 + 0.15 * p_idx
        for i_w in range(n_per_session):
            cls_k = i_w % 3
            freq_hz = (9.5, 16.0, 24.0)[cls_k]
            spat = np.cos(np.linspace(0.1, 2.8, 41) * (cls_k + 1) + 0.2 * p_idx)[:, None]
            sig = gain * 5.0 * spat * np.sin(2.0 * np.pi * freq_hz * t_axis + 0.1 * i_w)[None, :]
            win = (sig + 2.0 * rng.standard_normal((41, 480))).astype(np.float32)
            all_x.append(win)
            all_y.append(cls_k)
            all_part.append(part_id)
            all_sess.append(sess_id)
            all_ctx.append(ctx_str)

    rest_x = (2.0 * rng.standard_normal((16, 41, 480))).astype(np.float32)
    rest_sess = np.array([session_specs[i % 4][1] for i in range(16)], dtype="U64")
    return WindowedCorpusStreamContainer(
        corpus_identifier="synthetic_unit_test_corpus",
        eeg_windows=np.stack(all_x, axis=0),
        class_labels=np.array(all_y, dtype=np.int64),
        participant_ids=np.array(all_part, dtype="U32"),
        session_ids=np.array(all_sess, dtype="U64"),
        interface_contexts=np.array(all_ctx, dtype="U32"),
        channel_labels=SubspaceMatrixOperator.PROTEUS_41_CHANNELS,
        sampling_rate_hz=120.0,
        resting_windows=rest_x,
        resting_session_ids=rest_sess,
    )


def test_convolutional_backbones_strict_load_and_parity(r45_ref_module, synthetic_test_windows) -> None:
    assert issubclass(AdaptiveBatchNormConvolutionalBackbone, ConvolutionalBackbone)
    assert issubclass(ShallowTemporalConvolutionalBackbone, ConvolutionalBackbone)
    assert issubclass(FilterBankConvolutionalBackbone, ConvolutionalBackbone)
    assert issubclass(GroupedEnsembleConvolutionalBackbone, ConvolutionalBackbone)

    nine_bands = [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]]
    xb = torch.as_tensor(synthetic_test_windows[:4], dtype=torch.float32)

    with zipfile.ZipFile(LOCKED_R45_ZIP, "r") as zf:
        sd_sh25 = torch.load(io.BytesIO(zf.read("cnn_0.pt")), map_location="cpu", weights_only=True)
        sd_sh25_1 = torch.load(io.BytesIO(zf.read("cnn_1.pt")), map_location="cpu", weights_only=True)
        sd_fb9 = torch.load(io.BytesIO(zf.read("cnn2_cnn_0.pt")), map_location="cpu", weights_only=True)
        sd_fb9_1 = torch.load(io.BytesIO(zf.read("cnn2_cnn_1.pt")), map_location="cpu", weights_only=True)
        sd_fbm = torch.load(io.BytesIO(zf.read("cnn3_cnn_0.pt")), map_location="cpu", weights_only=True)
        sd_fbm_1 = torch.load(io.BytesIO(zf.read("cnn3_cnn_1.pt")), map_location="cpu", weights_only=True)
        sd_sh49 = torch.load(io.BytesIO(zf.read("cnn4_cnn_0.pt")), map_location="cpu", weights_only=True)
        sd_sh49_1 = torch.load(io.BytesIO(zf.read("cnn4_cnn_1.pt")), map_location="cpu", weights_only=True)

    our_sh25_0 = ShallowTemporalConvolutionalBackbone(41, 480, 3, kernel=25).eval()
    our_sh25_1 = ShallowTemporalConvolutionalBackbone(41, 480, 3, kernel=25).eval()
    our_sh25_0.load_state_dict(sd_sh25, strict=True)
    our_sh25_1.load_state_dict(sd_sh25_1, strict=True)

    ref_sh25_0 = r45_ref_module.ShallowNet(41, 480, 3, kernel=25).eval()
    ref_sh25_1 = r45_ref_module.ShallowNet(41, 480, 3, kernel=25).eval()
    ref_sh25_0.load_state_dict(sd_sh25, strict=True)
    ref_sh25_1.load_state_dict(sd_sh25_1, strict=True)

    our_sh49_0 = ShallowTemporalConvolutionalBackbone(41, 480, 3, kernel=49).eval()
    our_sh49_1 = ShallowTemporalConvolutionalBackbone(41, 480, 3, kernel=49).eval()
    our_sh49_0.load_state_dict(sd_sh49, strict=True)
    our_sh49_1.load_state_dict(sd_sh49_1, strict=True)
    ref_sh49_0 = r45_ref_module.ShallowNet(41, 480, 3, kernel=49).eval()
    ref_sh49_0.load_state_dict(sd_sh49, strict=True)

    our_fb9_0 = FilterBankConvolutionalBackbone(41, 480, 3, nine_bands, sub4=False, modpool=False).eval()
    our_fb9_1 = FilterBankConvolutionalBackbone(41, 480, 3, nine_bands, sub4=False, modpool=False).eval()
    our_fb9_0.load_state_dict(sd_fb9, strict=True)
    our_fb9_1.load_state_dict(sd_fb9_1, strict=True)
    ref_fb9_0 = r45_ref_module.FBCNet(41, 480, 3, nine_bands, sub4=False, modpool=False).eval()
    ref_fb9_1 = r45_ref_module.FBCNet(41, 480, 3, nine_bands, sub4=False, modpool=False).eval()
    ref_fb9_0.load_state_dict(sd_fb9, strict=True)
    ref_fb9_1.load_state_dict(sd_fb9_1, strict=True)

    our_fbm_0 = FilterBankConvolutionalBackbone(41, 480, 3, nine_bands, sub4=True, modpool=True).eval()
    our_fbm_1 = FilterBankConvolutionalBackbone(41, 480, 3, nine_bands, sub4=True, modpool=True).eval()
    our_fbm_0.load_state_dict(sd_fbm, strict=True)
    our_fbm_1.load_state_dict(sd_fbm_1, strict=True)
    ref_fbm_0 = r45_ref_module.FBCNet(41, 480, 3, nine_bands, sub4=True, modpool=True).eval()
    ref_fbm_1 = r45_ref_module.FBCNet(41, 480, 3, nine_bands, sub4=True, modpool=True).eval()
    ref_fbm_0.load_state_dict(sd_fbm, strict=True)
    ref_fbm_1.load_state_dict(sd_fbm_1, strict=True)

    with torch.no_grad():
        assert torch.max(torch.abs(our_sh25_0(xb) - ref_sh25_0(xb))).item() < 1e-6
        assert torch.max(torch.abs(our_sh49_0(xb) - ref_sh49_0(xb))).item() < 1e-6
        assert torch.max(torch.abs(our_fb9_0(xb) - ref_fb9_0(xb))).item() < 1e-6
        assert torch.max(torch.abs(our_fbm_0(xb) - ref_fbm_0(xb))).item() < 1e-6

        our_grp_sh = GroupedEnsembleConvolutionalBackbone([our_sh25_0, our_sh25_1])
        ref_grp_sh = r45_ref_module.GroupedNets([ref_sh25_0, ref_sh25_1])
        assert torch.max(torch.abs(our_grp_sh(xb, fused=True) - ref_grp_sh(xb, fused=True))).item() < 1e-5

        our_grp_fbm = GroupedEnsembleConvolutionalBackbone([our_fbm_0, our_fbm_1])
        ref_grp_fbm = r45_ref_module.GroupedNets([ref_fbm_0, ref_fbm_1])
        assert torch.max(torch.abs(our_grp_fbm(xb) - ref_grp_fbm(xb))).item() < 1e-6
        xf_b = our_grp_fbm.filter_bank(xb)
        assert torch.max(torch.abs(our_grp_fbm.band_features(xf_b) - ref_grp_fbm.band_features(xf_b))).item() < 1e-5


def test_riemannian_organs_r45_parity_causality_and_reset(tmp_path: Path, r45_ref_module, synthetic_test_windows) -> None:
    assert issubclass(SessionCspLdaRiemannianOrgan, RiemannianOrgan)
    assert issubclass(ParticipantSpecialistRiemannianOrgan, RiemannianOrgan)
    assert issubclass(SubwindowGeodesicRiemannianOrgan, RiemannianOrgan)
    assert issubclass(RestSubspaceNuisanceRiemannianOrgan, RiemannianOrgan)

    with zipfile.ZipFile(LOCKED_R45_ZIP, "r") as zf:
        for fn in ("tsr3task.npz", "rsnse_t1.npz", "csp.npz", "specialists.npz", "pseudochrono_nm_mt6_d6.npz"):
            (tmp_path / fn).write_bytes(zf.read(fn))

    with np.load(tmp_path / "pseudochrono_nm_mt6_d6.npz", allow_pickle=False) as zpc:
        pc_spec = {
            "mu_phi": zpc["mu_phi"].astype(np.float64),
            "sd_phi": zpc["sd_phi"].astype(np.float64),
            "U_run": zpc["U_run"].astype(np.float64),
            "feat_mode": str(zpc["feat_mode"].item()),
            "center_u": bool(int(zpc["center_u"].item())),
        }

    # 1. SubwindowGeodesicRiemannianOrgan (TSR4 pc_le + shipped)
    our_tsr = SubwindowGeodesicRiemannianOrgan.load_from_npz(tmp_path / "tsr3task.npz", pc_tau=1.5)
    ref_tsr = r45_ref_module.TSR3TaskOnlineHead.load(tmp_path / "tsr3task.npz", arm="TSR4_SubWin_Vel_sLDA")
    ref_tsr._r32_tsr_mode = "pc_le"
    ref_tsr._r32_tsr_pc_tau = 1.5
    pc_our = r45_ref_module._R32PseudoChronoSession(pc_spec, ea_tau=1.3, bn_tau=1.3, pc_after_freeze=True)
    pc_ref = r45_ref_module._R32PseudoChronoSession(pc_spec, ea_tau=1.3, bn_tau=1.3, pc_after_freeze=True)
    ref_tsr._r32_pc_ref = pc_ref

    max_err_tsr = 0.0
    tsr_first_run = []
    for idx_w in range(16):
        win = synthetic_test_windows[idx_w].astype(np.float64)
        bad_f = idx_w == 5
        pc_our.begin_window(win)
        pc_ref.begin_window(win)
        p_our = our_tsr.step_causal_window(win, bad_f, pseudo_chrono_adapter=pc_our)
        p_ref = ref_tsr.step(win, is_bad_bool=bad_f, x_clean_f64=win)
        tsr_first_run.append(p_our.copy())
        max_err_tsr = max(max_err_tsr, float(np.max(np.abs(p_our - p_ref))))
    assert max_err_tsr < 1e-10, f"TSR4 parity error {max_err_tsr}"

    # Reset invariance + causality check on TSR4
    our_tsr.reset_organ_state()
    pc_our_2 = r45_ref_module._R32PseudoChronoSession(pc_spec, ea_tau=1.3, bn_tau=1.3, pc_after_freeze=True)
    mutated_stream = synthetic_test_windows[:16].astype(np.float64).copy()
    mutated_stream[10:] = -99.0 * mutated_stream[10:]
    for idx_w in range(10):
        pc_our_2.begin_window(mutated_stream[idx_w])
        p_rep = our_tsr.step_causal_window(mutated_stream[idx_w], idx_w == 5, pseudo_chrono_adapter=pc_our_2)
        assert float(np.max(np.abs(p_rep - tsr_first_run[idx_w]))) == 0.0

    # 2. RestSubspaceNuisanceRiemannianOrgan (RSNSE)
    our_rsnse = RestSubspaceNuisanceRiemannianOrgan.load_from_npz(tmp_path / "rsnse_t1.npz")
    ref_rsnse = r45_ref_module.RSNSET1OnlineHead.load(tmp_path / "rsnse_t1.npz")
    max_err_rsnse = 0.0
    rsnse_first = []
    for idx_w in range(16):
        win = synthetic_test_windows[idx_w].astype(np.float64)
        bad_f = idx_w == 4
        p_our = our_rsnse.step_causal_window(win, bad_f)
        p_ref = ref_rsnse.step(win, is_bad_bool=bad_f)
        rsnse_first.append(p_our.copy())
        max_err_rsnse = max(max_err_rsnse, float(np.max(np.abs(p_our - p_ref))))
    assert max_err_rsnse < 1e-10, f"RSNSE parity error {max_err_rsnse}"
    assert our_rsnse.extract_standardized_661d_features(synthetic_test_windows[0]).shape == (661,)

    our_rsnse.reset_organ_state()
    for idx_w in range(8):
        win = synthetic_test_windows[idx_w].astype(np.float64)
        assert float(np.max(np.abs(our_rsnse.step_causal_window(win, idx_w == 4) - rsnse_first[idx_w]))) == 0.0

    # 3. SessionCspLdaRiemannianOrgan (including refit at t=66)
    our_csp = SessionCspLdaRiemannianOrgan.load_from_npz(tmp_path / "csp.npz", csp_weight=0.10, adaptive_shrinkage=10.0)
    with np.load(tmp_path / "csp.npz", allow_pickle=False) as zc:
        ref_csp = r45_ref_module.SessionCSP(
            {
                "csp_w": 0.10,
                "csp_weighted": True,
                "csp_bands": [[4, 8], [8, 13], [13, 20], [20, 30], [30, 45], [8, 30]],
                "csp_shrink_adaptive": 10.0,
                "csp_soft_cov": True,
            },
            zc["csp_train_ref"],
            zc["csp_train_class"],
        )
    rng_tchr = np.random.default_rng(77)
    max_err_csp = 0.0
    for idx_w in range(70):
        win = synthetic_test_windows[idx_w].astype(np.float64)
        bad_f = idx_w == 7
        p_our = our_csp.step_causal_window(win, bad_f)
        p_ref = ref_csp.step(win, None, None, skip=bad_f)
        term_err = float(np.max(np.abs(our_csp.additive_log_term() - ref_csp.term())))
        max_err_csp = max(max_err_csp, float(np.max(np.abs(p_our - p_ref))), term_err)
        fake_p = rng_tchr.dirichlet([2.0 if (idx_w % 3) == k else 0.5 for k in range(3)])
        our_csp.observe_teacher_probabilities(fake_p)
        ref_csp.observe(fake_p)
    assert max_err_csp < 1e-10, f"CSP parity error {max_err_csp}"

    # 4. ParticipantSpecialistRiemannianOrgan
    our_spec = ParticipantSpecialistRiemannianOrgan.load_from_npz(tmp_path / "specialists.npz")
    ref_spec = r45_ref_module.Specialists.load(
        tmp_path,
        {"spec_w": 0.2, "spec_kappa": 30.0, "spec_delta": 0.05, "spec_n0": 20.0, "spec_refit": 0.0, "cnn_spec_w": 0.0},
    )
    max_err_spec = 0.0
    for idx_w in range(16):
        win = synthetic_test_windows[idx_w].astype(np.float64)
        bad_f = idx_w == 3
        for m_s in ref_spec.models:
            m_s._skip = bad_f
        p_our = our_spec.step_causal_window(win, bad_f)
        p_ref = ref_spec.step(win, None)
        base_lp = np.log(np.array([0.5, 0.3, 0.2]))
        t_our = our_spec.compute_routed_term_and_update(base_lp, balance_prior=10.0)
        t_ref = ref_spec.term(base_lp, 10.0)
        max_err_spec = max(max_err_spec, float(np.max(np.abs(p_our - p_ref))), float(np.max(np.abs(t_our - t_ref))))
    assert max_err_spec < 1e-10, f"Specialists parity error {max_err_spec}"


def test_dual_block_cnn_organ_r45_bundle_load_and_stream_parity(
    tmp_path: Path,
    r45_ref_module,
    synthetic_test_windows,
) -> None:
    assert issubclass(DualBlockCnnEnsembleCnnOrgan, CnnOrgan)
    bundle_dir = tmp_path / "r45_cnn_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(LOCKED_R45_ZIP, "r") as zf:
        for member in zf.namelist():
            if (
                member.endswith(".pt")
                or member.endswith("weights.npz")
                or member in ("csp.npz", "specialists.npz", "mix.json", "pseudochrono_nm_mt6_d6.npz")
            ):
                (bundle_dir / member).write_bytes(zf.read(member))

    cnn_cfg = CnnEnsembleSectionConfiguration()
    cls_cfg = ClassicalHeadsSectionConfiguration()
    our_organ = DualBlockCnnEnsembleCnnOrgan.load_from_bundle_directory(
        bundle_dir, cnn_cfg, cls_cfg, execution_device="cpu"
    )
    assert our_organ.total_cnn_count() == 160

    # Build exact r45 reference Block A + Block B + Shallow_STOFF with identical r29/r31/r32 overrides
    mix_a = r45_ref_module.Track2Mix.load(bundle_dir, device="cpu")
    mix_meta = json.loads((bundle_dir / "mix.json").read_text(encoding="utf-8"))
    with np.load(bundle_dir / "csp.npz", allow_pickle=False) as zc:
        mix_meta["csp_train"] = (zc["csp_train_ref"], zc["csp_train_class"])
    mix_meta["specialists"] = r45_ref_module.Specialists.load(bundle_dir, mix_meta["specialists"])
    b_main = r45_ref_module.Track2Model.load(bundle_dir, "cpu", prefix="blkB_")
    b_sec = r45_ref_module.Track2Model.load(bundle_dir, "cpu", prefix="blkB_cnn2_")
    b_trd = r45_ref_module.Track2Model.load(bundle_dir, "cpu", prefix="blkB_cnn3_")
    b_fth = r45_ref_module.Track2Model.load(bundle_dir, "cpu", prefix="blkB_cnn4_")
    b_main.set_replay(*mix_a.main.replay)
    b_sec.set_replay(*mix_a.second.replay)
    mix_b = r45_ref_module.Track2Mix(b_main, b_sec, third=b_trd, fourth=b_fth, **mix_meta)
    sh_stoff_a = r45_ref_module._clone_shallow_stoff_model(mix_a.main)
    sh_stoff_b = r45_ref_module._clone_shallow_stoff_model(mix_b.main)

    for blk in (mix_a, mix_b):
        blk.main.cfg["self_train_at"] = [66, 126]
        blk.second.cfg["self_train_at"] = [66, 126]
        blk.csp.cfg["csp_at"] = [66, 126]
        blk.main.cfg["r29_st_step_scale"] = 0.5
        blk.second.cfg["r29_st_step_scale"] = 0.5
        blk.main._r23_early_gentle = True
        blk.second._r23_early_gentle = True
    for exp in (
        mix_a.main, mix_a.second, mix_a.third, mix_a.fourth,
        mix_b.main, mix_b.second, mix_b.third, mix_b.fourth,
        sh_stoff_a, sh_stoff_b,
    ):
        exp.cfg = dict(exp.cfg)
        exp.cfg["tta_lr"] = float(exp.cfg.get("tta_lr", 0.0)) * 0.5
        exp.cfg["tta_buffer"] = 64
        exp.cfg["tta_freeze"] = 130
        exp.cfg["adapt_bn"] = 0.10
        exp.cfg["_r31_cum_bn_n0"] = 0.0
    mix_a.reset_state()
    mix_b.reset_state()
    sh_stoff_a.reset_state()
    sh_stoff_b.reset_state()

    with np.load(bundle_dir / "pseudochrono_nm_mt6_d6.npz", allow_pickle=False) as zpc:
        pc_spec = {
            "mu_phi": zpc["mu_phi"].astype(np.float64),
            "sd_phi": zpc["sd_phi"].astype(np.float64),
            "U_run": zpc["U_run"].astype(np.float64),
            "feat_mode": str(zpc["feat_mode"].item()),
            "center_u": bool(int(zpc["center_u"].item())),
        }
    pc_our = r45_ref_module._R32PseudoChronoSession(pc_spec, ea_tau=1.3, bn_tau=1.3, pc_after_freeze=True)
    pc_ref = r45_ref_module._R32PseudoChronoSession(pc_spec, ea_tau=1.3, bn_tau=1.3, pc_after_freeze=True)
    for exp in (
        mix_a.main, mix_a.second, mix_a.third, mix_a.fourth,
        mix_b.main, mix_b.second, mix_b.third, mix_b.fourth,
        sh_stoff_a, sh_stoff_b,
    ):
        exp._set_bn_adapt(0.10)
        exp._group._r32_pc_bn = pc_ref

    prev_ro_a = np.zeros(3, dtype=np.float64)
    prev_ro_b = np.zeros(3, dtype=np.float64)
    max_cnn40_err = 0.0
    max_stoff_err = 0.0
    rng_tchr = np.random.default_rng(909)

    # Stream 10 windows (verifying Tent, PseudoChrono EA+BN, CSP, Specialists, and STOFF)
    for idx_w in range(10):
        x_np = synthetic_test_windows[idx_w].astype(np.float64)
        x_cent = x_np - x_np.mean(axis=-1, keepdims=True)
        bad_a = idx_w == 3

        pc_our.begin_window(x_np)
        u_cnn40_our, u_stoff_our = our_organ.step_cnn40_and_stoff(
            cleaned_window=x_np,
            dc_centered_window=x_cent,
            is_bad_window=bad_a,
            pseudo_chrono_adapter=pc_our,
        )

        pc_ref.begin_window(x_np)
        for m_exp in (*mix_a._cnn_experts(), *mix_a.spec.models):
            m_exp._skip = bad_a
        cov_pc, w_pc, z_pc = pc_ref.step_window_ea_and_bn(x_np, x_cent, bad_a, False, mix_a.main)
        known_a = {tuple(b): cov_pc[i] for i, b in enumerate(mix_a.main.cfg["bands"])}
        known_a[None] = cov_pc[-1]
        pre_csp = mix_a.csp.step(x_np, None, known_a, skip=bad_a)
        for bi_c, b_c in enumerate(mix_a.csp.cfg["csp_bands"]):
            known_a[tuple(b_c)] = mix_a.csp._last_csp_covs[bi_c]
        comps_a = mix_a.main.components(x_np, covs=cov_pc, _pre_W=w_pc, _pre_z=z_pc, _pre_x_cent=x_cent)
        cov_last_a = mix_a.main.last_covs[-1:]
        w_last_a = mix_a.main._last_w_raw[-1:]
        comps_a["cnn2"] = mix_a.second.components(x_np, covs=cov_last_a, _pre_W=w_last_a, _pre_x_cent=x_cent)["cnn"]
        comps_a["cnn3"] = mix_a.third.components(x_np, covs=cov_last_a, _pre_W=w_last_a, _pre_x_cent=x_cent)["cnn"]
        comps_a["cnn4"] = mix_a.fourth.components(x_np, covs=cov_last_a, _pre_W=w_last_a, _pre_x_cent=x_cent)["cnn"]
        comps_a["csp"] = pre_csp
        comps_a["spec"] = mix_a.spec.step(x_np, known_a)

        for m_exp in (mix_b.main, mix_b.second, *mix_b.spec.models):
            m_exp._skip = bad_a
        mix_b.third._skip = bad_a
        mix_b.fourth._skip = bad_a
        comps_b = mix_b.main.components(
            x_np,
            covs=mix_a.main.last_covs,
            _pre_W=mix_a.main._last_w_raw,
            _pre_z=mix_a.main.last_z,
            _pre_x_cent=x_cent,
        )
        comps_b["cnn2"] = mix_b.second.components(x_np, covs=cov_last_a, _pre_W=w_last_a, _pre_x_cent=x_cent)["cnn"]
        comps_b["cnn3"] = mix_b.third.components(x_np, covs=cov_last_a, _pre_W=w_last_a, _pre_x_cent=x_cent)["cnn"]
        comps_b["cnn4"] = mix_b.fourth.components(x_np, covs=cov_last_a, _pre_W=w_last_a, _pre_x_cent=x_cent)["cnn"]
        comps_b["csp"] = r45_ref_module.Slot6DualEEGPTMix._step_csp_reusing_a(mix_b.csp, mix_a.csp, None, skip=bad_a)
        comps_b["spec"] = r45_ref_module.Slot6DualEEGPTMix._step_spec_reusing_a(mix_b.spec, mix_a.spec)

        _ = mix_a.combine(comps_a)
        u_a_ref = mix_a._ro_sum - prev_ro_a
        prev_ro_a = mix_a._ro_sum.copy()
        _ = mix_b.combine(comps_b)
        u_b_ref = mix_b._ro_sum - prev_ro_b
        prev_ro_b = mix_b._ro_sum.copy()
        norm_lp = r45_ref_module.Slot6DualEEGPTMix._norm_lp
        u_cnn40_ref = r45_ref_module.softmax((0.5 * norm_lp(u_a_ref) + 0.5 * norm_lp(u_b_ref))[None])[0]

        sh_stoff_a._skip = bad_a
        sh_stoff_b._skip = bad_a
        uk25_a = sh_stoff_a.components(x_np, covs=cov_last_a, _pre_W=w_last_a, _pre_x_cent=x_cent)["cnn"]
        uk25_b = sh_stoff_b.components(x_np, covs=cov_last_a, _pre_W=w_last_a, _pre_x_cent=x_cent)["cnn"]
        uk25 = r45_ref_module.softmax((0.5 * norm_lp(uk25_a) + 0.5 * norm_lp(uk25_b))[None])[0]
        uk49 = r45_ref_module.softmax((0.5 * norm_lp(comps_a["cnn4"]) + 0.5 * norm_lp(comps_b["cnn4"]))[None])[0]
        u_stoff_ref = r45_ref_module.softmax((0.5 * norm_lp(uk25) + 0.5 * norm_lp(uk49))[None])[0]

        max_cnn40_err = max(max_cnn40_err, float(np.max(np.abs(u_cnn40_our - u_cnn40_ref))))
        max_stoff_err = max(max_stoff_err, float(np.max(np.abs(u_stoff_our - u_stoff_ref))))

        tchr_p = rng_tchr.dirichlet([3.0 if (idx_w % 3) == k else 0.4 for k in range(3)])
        our_organ.record_multimodal_teacher_probs(tchr_p, is_bad_window=bad_a)
        for exp_s in (mix_a.main, mix_a.second, mix_b.main, mix_b.second):
            exp_s._pseudo[-1] = tchr_p.copy()

    assert max_cnn40_err < 1e-6, f"CNN40 stream parity error {max_cnn40_err}"
    assert max_stoff_err < 1e-6, f"Shallow_STOFF stream parity error {max_stoff_err}"


def test_end_to_end_fit_save_load_roundtrip_all_organs(tmp_path: Path) -> None:
    corpus = _make_synthetic_corpus(n_per_session=15)
    cnn_cfg = CnnEnsembleSectionConfiguration(
        seeds_per_family=2,
        block_a_seed_offset=0,
        block_b_seed_offset=50,
        shallow25_epochs=2,
        fbcnet_epochs=2,
        fbmodnet_distill_epochs=2,
        shallow49_distill_epochs=2,
        batch_size=16,
        distill_bn_groups=2,
        swa_tail_fraction=0.5,
        replay_windows_per_class=4,
        self_train_checkpoints=(10,),
        self_train_epochs_shallow=1,
        self_train_epochs_fbcnet=1,
        self_train_epoch_scale=1.0,
        self_train_teacher_gate_threshold=0.35,
        ag3_epochs=1,
    )
    cls_cfg = ClassicalHeadsSectionConfiguration(rsnse_mlp_seeds=2)

    # 1. Fit & roundtrip all 4 Riemannian organs
    riem_dir = tmp_path / "riemannian_artifacts"
    csp_fit = SessionCspLdaRiemannianOrgan.fit_from_corpus(corpus)
    csp_paths = csp_fit.save_organ_artifacts(riem_dir)
    csp_loaded = SessionCspLdaRiemannianOrgan.load_from_npz(csp_paths[0], refit_checkpoints=(10,))

    spec_fit = ParticipantSpecialistRiemannianOrgan.fit_from_corpus(corpus)
    spec_paths = spec_fit.save_organ_artifacts(riem_dir)
    spec_loaded = ParticipantSpecialistRiemannianOrgan.load_from_npz(spec_paths[0])

    tsr_fit = SubwindowGeodesicRiemannianOrgan.fit_from_corpus(corpus, random_seed=0)
    tsr_paths = tsr_fit.save_organ_artifacts(riem_dir)
    tsr_loaded = SubwindowGeodesicRiemannianOrgan.load_from_npz(tsr_paths[0])

    rsnse_fit = RestSubspaceNuisanceRiemannianOrgan.fit_from_corpus(corpus, mlp_seeds=2, random_seed=0)
    rsnse_paths = rsnse_fit.save_organ_artifacts(riem_dir)
    rsnse_loaded = RestSubspaceNuisanceRiemannianOrgan.load_from_npz(rsnse_paths[0])

    for npz_p in (*csp_paths, *spec_paths, *tsr_paths, *rsnse_paths):
        with np.load(npz_p, allow_pickle=False) as chk:
            assert len(chk.files) > 0

    for idx_w in range(12):
        win = corpus.eeg_windows[idx_w]
        assert np.allclose(csp_fit.step_causal_window(win, False), csp_loaded.step_causal_window(win, False))
        csp_fit.observe_teacher_probabilities(np.eye(3)[idx_w % 3])
        csp_loaded.observe_teacher_probabilities(np.eye(3)[idx_w % 3])
        assert np.allclose(spec_fit.step_causal_window(win, False), spec_loaded.step_causal_window(win, False))
        assert np.allclose(tsr_fit.step_causal_window(win, False), tsr_loaded.step_causal_window(win, False))
        assert np.allclose(rsnse_fit.step_causal_window(win, False), rsnse_loaded.step_causal_window(win, False))

    # 2. Fit & roundtrip DualBlockCnnEnsembleCnnOrgan (including self-training + AG3 trigger at window 10!)
    cnn_dir = tmp_path / "cnn_organ_bundle"
    cnn_fitted = DualBlockCnnEnsembleCnnOrgan.fit_from_corpus(
        corpus, cnn_cfg, cls_cfg, execution_device="cpu", master_seed=1234
    )
    assert cnn_fitted.total_cnn_count() == 16
    saved_paths = cnn_fitted.save_organ_artifacts(cnn_dir)
    expected_files = {
        "weights.npz", "cnn_0.pt", "cnn_1.pt",
        "cnn2_weights.npz", "cnn2_cnn_0.pt", "cnn2_cnn_1.pt",
        "cnn3_weights.npz", "cnn3_cnn_0.pt", "cnn3_cnn_1.pt",
        "cnn4_weights.npz", "cnn4_cnn_0.pt", "cnn4_cnn_1.pt",
        "blkB_weights.npz", "blkB_cnn_0.pt", "blkB_cnn_1.pt",
        "blkB_cnn2_weights.npz", "blkB_cnn2_cnn_0.pt", "blkB_cnn2_cnn_1.pt",
        "blkB_cnn3_weights.npz", "blkB_cnn3_cnn_0.pt", "blkB_cnn3_cnn_1.pt",
        "blkB_cnn4_weights.npz", "blkB_cnn4_cnn_0.pt", "blkB_cnn4_cnn_1.pt",
        "csp.npz", "specialists.npz", "mix.json",
    }
    assert expected_files.issubset({p_item.name for p_item in saved_paths})
    for p_item in saved_paths:
        if p_item.suffix == ".npz":
            with np.load(p_item, allow_pickle=False) as chk:
                assert len(chk.files) > 0

    cnn_loaded = DualBlockCnnEnsembleCnnOrgan.load_from_bundle_directory(
        cnn_dir, cnn_cfg, cls_cfg, execution_device="cpu"
    )
    cnn_fitted.reset_organ_state()
    for idx_w in range(12):
        win = corpus.eeg_windows[idx_w]
        u_fit, st_fit = cnn_fitted.step_cnn40_and_stoff(win, is_bad_window=(idx_w == 2))
        u_ld, st_ld = cnn_loaded.step_cnn40_and_stoff(win, is_bad_window=(idx_w == 2))
        assert u_fit.shape == (3,) and st_fit.shape == (3,)
        assert np.allclose(u_fit, u_ld, atol=1e-6)
        assert np.allclose(st_fit, st_ld, atol=1e-6)
        pseudo_tchr = np.array([0.80, 0.10, 0.10]) if idx_w % 3 == 0 else (
            np.array([0.10, 0.80, 0.10]) if idx_w % 3 == 1 else np.array([0.10, 0.10, 0.80])
        )
        cnn_fitted.record_multimodal_teacher_probs(pseudo_tchr)
        cnn_loaded.record_multimodal_teacher_probs(pseudo_tchr)
