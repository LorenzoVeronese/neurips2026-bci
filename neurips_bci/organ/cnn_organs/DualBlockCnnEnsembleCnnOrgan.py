"""Dual-block 160-CNN + TS-LR + AG3 + CSP + Specialists + Shallow-STOFF streaming ensemble organ."""

from __future__ import annotations

import contextlib
import copy
from itertools import permutations
import json
from pathlib import Path
from typing import Any
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn.functional as F
from torch import nn

from neurips_bci.backbone.convolutional_backbones.AdaptiveBatchNormConvolutionalBackbone import (
    AdaptiveBatchNormConvolutionalBackbone,
)
from neurips_bci.backbone.convolutional_backbones.FilterBankConvolutionalBackbone import (
    FilterBankConvolutionalBackbone,
)
from neurips_bci.backbone.convolutional_backbones.GroupedEnsembleConvolutionalBackbone import (
    GroupedEnsembleConvolutionalBackbone,
)
from neurips_bci.backbone.convolutional_backbones.ShallowTemporalConvolutionalBackbone import (
    ShallowTemporalConvolutionalBackbone,
)
from neurips_bci.configuration.section_configurations.ClassicalHeadsSectionConfiguration import (
    ClassicalHeadsSectionConfiguration,
)
from neurips_bci.configuration.section_configurations.CnnEnsembleSectionConfiguration import (
    CnnEnsembleSectionConfiguration,
)
from neurips_bci.container.stream_containers.WindowedCorpusStreamContainer import WindowedCorpusStreamContainer
from neurips_bci.operator.matrix_operators.CovarianceMatrixOperator import CovarianceMatrixOperator
from neurips_bci.operator.signal_operators.SpectralSignalOperator import SpectralSignalOperator
from neurips_bci.organ.cnn_organs.CnnOrgan import CnnOrgan
from neurips_bci.organ.riemannian_organs.ParticipantSpecialistRiemannianOrgan import (
    ParticipantSpecialistRiemannianOrgan,
)
from neurips_bci.organ.riemannian_organs.SessionCspLdaRiemannianOrgan import SessionCspLdaRiemannianOrgan
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class DualBlockCnnEnsembleCnnOrgan(CnnOrgan):
    """Encapsulates Block A and Block B (8 x N CNNs), TS-LR, DistanceGate, AG3, CSP, Specialists, and STOFF."""

    SIX_MAIN_BANDS_HZ: tuple[tuple[int, int], ...] = (
        (4, 8),
        (8, 13),
        (13, 20),
        (20, 30),
        (30, 45),
        (1, 45),
    )
    NINE_FBC_BANDS_HZ: tuple[tuple[int, int], ...] = (
        (4, 8),
        (8, 12),
        (12, 16),
        (16, 20),
        (20, 24),
        (24, 28),
        (28, 32),
        (32, 36),
        (36, 40),
    )

    def __init__(
        self,
        block_a_bundle: dict[str, Any],
        block_b_bundle: dict[str, Any],
        csp_organ_a: SessionCspLdaRiemannianOrgan | None,
        csp_organ_b: SessionCspLdaRiemannianOrgan | None,
        specialist_organ_a: ParticipantSpecialistRiemannianOrgan | None,
        specialist_organ_b: ParticipantSpecialistRiemannianOrgan | None,
        mix_metadata: dict[str, Any],
        cnn_config: CnnEnsembleSectionConfiguration,
        classical_config: ClassicalHeadsSectionConfiguration,
        execution_device: str = "cpu",
    ) -> None:
        super().__init__(organ_identifier="dual_block_cnn_ensemble")
        self._cnn_config = cnn_config
        self._classical_config = classical_config
        resolved_dev = "cuda" if (execution_device == "auto" and torch.cuda.is_available()) else (
            "cpu" if execution_device == "auto" else execution_device
        )
        self._device = torch.device(resolved_dev)
        self._mix_metadata = copy.deepcopy(mix_metadata)
        self._csp_a = csp_organ_a
        self._csp_b = csp_organ_b
        self._spec_a = specialist_organ_a
        self._spec_b = specialist_organ_b

        self._block_a = self._prepare_block_runtime(block_a_bundle, "A")
        self._block_b = self._prepare_block_runtime(block_b_bundle, "B")
        self._sh_stoff_a = self._clone_shallow_stoff_expert(self._block_a["main"])
        self._sh_stoff_b = self._clone_shallow_stoff_expert(self._block_b["main"])

        main_a = self._block_a["main"]
        self._sampling_rate_hz = float(main_a["cfg"].get("sfreq", 120.0))
        self.fir = None
        self.sos = SpectralSignalOperator.build_butterworth_sos_bank(
            [list(band_pair) for band_pair in main_a["bands"]],
            sampling_rate_hz=self._sampling_rate_hz,
        )
        self.cfg = {"ea_cov": str(main_a["cfg"].get("ea_cov", "lw")), "ea_scale": str(main_a["cfg"].get("ea_scale", "none"))}
        self.R0 = main_a["r0"]
        self.scaler_mean = main_a["scaler_mean"]
        self.scaler_scale = main_a["scaler_scale"]

        self._cnn_p_sum = np.zeros(3, dtype=np.float64)
        self._cnn_p_t = 0
        self._multimodal_teacher_hist: list[np.ndarray] = []
        self._cnn40_bal_hist: list[np.ndarray] = []
        self._has_external_teacher = False
        self._builtin_pc_ea_state: dict[str, Any] = {}
        self.reset_organ_state()

    @property
    def execution_device(self) -> torch.device:
        return self._device

    def total_cnn_count(self) -> int:
        total_nets = 0
        for blk_rt in (self._block_a, self._block_b):
            for fam_key in ("main", "second", "third", "fourth"):
                if blk_rt.get(fam_key) is not None:
                    total_nets += len(blk_rt[fam_key]["cnns"])
        return total_nets

    def _prepare_block_runtime(self, raw_bundle: dict[str, Any], block_tag: str) -> dict[str, Any]:
        prepared: dict[str, Any] = {"tag": block_tag}
        checkpoints = tuple(int(ckpt) for ckpt in self._cnn_config.self_train_checkpoints)
        for fam_key in ("main", "second", "third", "fourth"):
            spec = raw_bundle.get(fam_key)
            if spec is None:
                prepared[fam_key] = None
                continue
            fam_cfg = copy.deepcopy(spec["cfg"])
            cnns: list[nn.Module] = []
            for net_mod in spec["cnns"]:
                net_mod = net_mod.to(self._device).eval()
                for sub_m in net_mod.modules():
                    if isinstance(sub_m, AdaptiveBatchNormConvolutionalBackbone):
                        sub_m.adapt = float(fam_cfg.get("adapt_bn", 0.10))
                cnns.append(net_mod)

            bn_train = [self._capture_bn_state(net_mod) for net_mod in cnns]
            train_states = [{k_s: t_v.detach().clone() for k_s, t_v in net_mod.state_dict().items()} for net_mod in cnns]

            is_main = fam_key == "main"
            is_st_family = fam_key in ("main", "second")
            tta_lr = float(self._cnn_config.tent_learning_rate) if is_main else 0.0
            st_at = checkpoints if is_st_family else ()
            bands_list = [tuple(map(float, bp)) for bp in fam_cfg.get("bands", [])]
            grouped_net = GroupedEnsembleConvolutionalBackbone(cnns).to(self._device)
            grouped_net.configure_tent_gradients(enable_tent=(tta_lr > 0.0))

            prepared[fam_key] = {
                "role": fam_key,
                "cfg": fam_cfg,
                "saved_cfg": copy.deepcopy(spec["cfg"]),
                "bands": tuple(bands_list),
                "r0": np.asarray(spec["r0"], dtype=np.float64).copy(),
                "scaler_mean": None if spec.get("scaler_mean") is None else np.asarray(spec["scaler_mean"], dtype=np.float64).copy(),
                "scaler_scale": None if spec.get("scaler_scale") is None else np.asarray(spec["scaler_scale"], dtype=np.float64).copy(),
                "coef": None if spec.get("coef") is None else np.asarray(spec["coef"], dtype=np.float64).copy(),
                "intercept": None if spec.get("intercept") is None else np.asarray(spec["intercept"], dtype=np.float64).copy(),
                "replay": None if spec.get("replay") is None else (
                    np.asarray(spec["replay"][0], dtype=np.float32).copy(),
                    np.asarray(spec["replay"][1], dtype=np.int64).copy(),
                ),
                "gate": copy.deepcopy(spec.get("gate")),
                "cnns": cnns,
                "bn_train": bn_train,
                "train_states": train_states,
                "group": grouped_net,
                "tta_lr": tta_lr,
                "tta_buffer": int(self._cnn_config.tent_buffer_length),
                "tta_freeze": int(self._cnn_config.tent_freeze_window),
                "tta_min_buf": int(fam_cfg.get("tta_min_buf") or 1),
                "tta_div": float(fam_cfg.get("tta_div", 1.0)),
                "tta_warmup": int(fam_cfg.get("tta_warmup", 0)),
                "self_train_at": st_at,
                "self_train_epochs": int(
                    fam_cfg.get(
                        "self_train_epochs",
                        self._cnn_config.self_train_epochs_shallow
                        if is_main
                        else self._cnn_config.self_train_epochs_fbcnet,
                    )
                ),
                "self_train_lr": float(fam_cfg.get("self_train_lr", 1e-3)),
                "self_train_bs": int(fam_cfg.get("self_train_bs", 32)),
                "self_train_conf": float(fam_cfg.get("self_train_conf", 0.0)),
                "self_train_min": int(fam_cfg.get("self_train_min", 8)),
                "self_train_im_w": float(fam_cfg.get("self_train_im_w", 1.0 if is_main else 0.0)),
                "self_train_params": str(fam_cfg.get("self_train_params", "spatial_head" if is_main else "all")),
                "self_train_replay": int(fam_cfg.get("self_train_replay", 100 if is_st_family else 0)),
                "self_train_gate": fam_cfg.get("self_train_gate", "near" if is_st_family else ""),
                "self_train_seed": int(fam_cfg.get("self_train_seed", 0)),
                "label_smoothing": float(fam_cfg.get("label_smoothing", self._cnn_config.label_smoothing)),
                "ts_refit": float(fam_cfg.get("ts_refit") or 0.0) if is_main else 0.0,
                "ts_refit_at": tuple(int(v) for v in (fam_cfg.get("ts_refit_at") or (66, 96, 126, 156))),
                "lr_c": float(fam_cfg.get("lr_C", 0.1)),
                "ea_prior": float(fam_cfg.get("ea_prior", 10.0)),
                "ea_scale": str(fam_cfg.get("ea_scale", "none")),
                "ea_freeze": int(fam_cfg.get("ea_freeze", 0)),
                "bn_freeze": int(fam_cfg.get("bn_freeze", 0)),
                "use_ts": bool(fam_cfg.get("use_ts", False)) and (spec.get("coef") is not None),
            }

        shared_gate = next(
            (prepared[fk]["gate"] for fk in ("main", "second", "third", "fourth") if prepared[fk] and prepared[fk]["gate"] is not None),
            None,
        )
        if shared_gate is not None:
            for fk in ("main", "second", "third", "fourth"):
                if prepared[fk] is not None and prepared[fk]["gate"] is None:
                    prepared[fk]["gate"] = copy.deepcopy(shared_gate)

        chan_count = int(prepared["main"]["r0"].shape[-1])
        ag3_raw = self._mix_metadata.get("ag3")
        if ag3_raw is not None:
            w_ts = float(self._cnn_config.weight_ts_lr)
            w_fb2 = float(self._cnn_config.weight_fbcnet)
            w_sh = 1.0 - w_ts - w_fb2
            prepared["ag3"] = {
                "mode": str(ag3_raw.get("mode", "row_l2_gated_wtst")),
                "epochs": int(ag3_raw.get("epochs", self._cnn_config.ag3_epochs)),
                "lr": float(ag3_raw.get("lr", self._cnn_config.ag3_learning_rate)),
                "decay": float(ag3_raw.get("decay", self._cnn_config.ag3_weight_decay)),
                "w_sh": float(w_sh),
                "w_fb2": float(w_fb2),
                "w_fb3": 0.0,
                "adapter_a": np.eye(chan_count, dtype=np.float64),
                "fitted_at": -1,
                "last_third": 1,
            }
        else:
            prepared["ag3"] = None
        prepared["p_sum"] = np.zeros(3, dtype=np.float64)
        prepared["ro_sum"] = np.zeros(3, dtype=np.float64)
        prepared["p_t"] = 0
        return prepared

    def _clone_shallow_stoff_expert(self, src_main: dict[str, Any]) -> dict[str, Any]:
        cloned_cnns = [copy.deepcopy(net_m).to(self._device).eval() for net_m in src_main["cnns"]]
        bn_train = [self._capture_bn_state(net_m) for net_m in cloned_cnns]
        train_states = [{k_s: t_v.detach().clone() for k_s, t_v in net_m.state_dict().items()} for net_m in cloned_cnns]
        grouped_net = GroupedEnsembleConvolutionalBackbone(cloned_cnns).to(self._device)
        grouped_net.configure_tent_gradients(enable_tent=(src_main["tta_lr"] > 0.0))
        return {
            "role": "stoff",
            "cfg": copy.deepcopy(src_main["cfg"]),
            "bands": (),
            "r0": src_main["r0"][-1:].copy(),
            "scaler_mean": None,
            "scaler_scale": None,
            "coef": None,
            "intercept": None,
            "replay": None,
            "gate": None,
            "cnns": cloned_cnns,
            "bn_train": bn_train,
            "train_states": train_states,
            "group": grouped_net,
            "tta_lr": float(src_main["tta_lr"]),
            "tta_buffer": int(src_main["tta_buffer"]),
            "tta_freeze": int(src_main["tta_freeze"]),
            "tta_min_buf": int(src_main["tta_min_buf"]),
            "tta_div": float(src_main["tta_div"]),
            "tta_warmup": int(src_main["tta_warmup"]),
            "self_train_at": (),
            "self_train_epochs": 0,
            "self_train_lr": 0.0,
            "self_train_bs": 32,
            "self_train_conf": 0.0,
            "self_train_min": 8,
            "self_train_im_w": 0.0,
            "self_train_params": "spatial_head",
            "self_train_replay": 0,
            "self_train_gate": "",
            "self_train_seed": 0,
            "label_smoothing": float(src_main["label_smoothing"]),
            "ts_refit": 0.0,
            "ts_refit_at": (),
            "lr_c": 0.1,
            "ea_prior": float(src_main["ea_prior"]),
            "ea_scale": str(src_main["ea_scale"]),
            "ea_freeze": int(src_main["ea_freeze"]),
            "bn_freeze": int(src_main["bn_freeze"]),
            "use_ts": False,
        }

    @staticmethod
    def _capture_bn_state(net_module: nn.Module) -> dict[str, torch.Tensor]:
        bn_keys = {
            f"{mod_path}.{stat_key}" if mod_path else stat_key
            for mod_path, sub_mod in net_module.named_modules()
            if isinstance(sub_mod, nn.BatchNorm2d)
            for stat_key in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")
        }
        return {k_str: v_t.detach().clone() for k_str, v_t in net_module.state_dict().items() if k_str in bn_keys}

    @staticmethod
    def _reset_single_expert(exp_dict: dict[str, Any]) -> None:
        for net_m, bn_sd, full_sd in zip(exp_dict["cnns"], exp_dict["bn_train"], exp_dict["train_states"]):
            if exp_dict["self_train_at"]:
                net_m.load_state_dict(full_sd)
            net_m.load_state_dict(bn_sd, strict=False)
            for sub_m in net_m.modules():
                if isinstance(sub_m, AdaptiveBatchNormConvolutionalBackbone):
                    sub_m.adapt = float(exp_dict["cfg"].get("adapt_bn", 0.10))
        exp_dict["group"].read_from(exp_dict["cnns"])
        exp_dict["group"].configure_tent_gradients(enable_tent=(exp_dict["tta_lr"] > 0.0))
        exp_dict["cov_sum"] = np.zeros_like(exp_dict["r0"])
        exp_dict["t"] = 0
        exp_dict["n_ref"] = 0
        exp_dict["gate_r"] = None
        exp_dict["bn_frozen"] = False
        exp_dict["ok"] = []
        exp_dict["raw"] = []
        exp_dict["pseudo"] = []
        exp_dict["ts_coef"] = exp_dict["coef"]
        exp_dict["ts_intercept"] = exp_dict["intercept"]
        exp_dict["ts_z"] = []
        exp_dict["st_done"] = set()
        exp_dict["buf"] = []
        exp_dict["buf_raw_w"] = []
        exp_dict["hbuf"] = []
        exp_dict["hbuf_ver"] = None
        exp_dict["group_opt"] = None
        exp_dict["local_pc_bn"] = {}
        exp_dict["last_covs"] = None
        exp_dict["last_w_raw"] = None
        exp_dict["last_z"] = None
        exp_dict["ag3_w_raw_last"] = None
        exp_dict["skip_st_core"] = False

    def reset_organ_state(self) -> None:
        for blk_rt in (self._block_a, self._block_b):
            for fam_key in ("main", "second", "third", "fourth"):
                if blk_rt.get(fam_key) is not None:
                    self._reset_single_expert(blk_rt[fam_key])
            if blk_rt.get("ag3") is not None:
                chan_count = int(blk_rt["main"]["r0"].shape[-1])
                blk_rt["ag3"]["adapter_a"] = np.eye(chan_count, dtype=np.float64)
                blk_rt["ag3"]["fitted_at"] = -1
                blk_rt["ag3"]["last_third"] = 1
            blk_rt["p_sum"] = np.zeros(3, dtype=np.float64)
            blk_rt["ro_sum"] = np.zeros(3, dtype=np.float64)
            blk_rt["p_t"] = 0
        self._reset_single_expert(self._sh_stoff_a)
        self._reset_single_expert(self._sh_stoff_b)

        checkpoints = tuple(int(ckpt) for ckpt in self._cnn_config.self_train_checkpoints)
        if self._csp_a is not None:
            self._csp_a.set_refit_checkpoints(checkpoints)
            self._csp_a.reset_organ_state()
        if self._csp_b is not None:
            self._csp_b.set_refit_checkpoints(checkpoints)
            self._csp_b.reset_organ_state()
        if self._spec_a is not None:
            self._spec_a.reset_organ_state()
        if self._spec_b is not None:
            self._spec_b.reset_organ_state()

        self._cnn_p_sum = np.zeros(3, dtype=np.float64)
        self._cnn_p_t = 0
        self._multimodal_teacher_hist = []
        self._cnn40_bal_hist = []
        self._has_external_teacher = False
        self._builtin_pc_ea_state = {}

    @staticmethod
    def _gate_parameters(exp_dict: dict[str, Any]) -> tuple[str, float, float, float | None, int]:
        raw_gate = exp_dict.get("self_train_gate", "")
        lo_val = float(exp_dict["cfg"].get("self_train_gate_lo", 0.5))
        hi_val = float(exp_dict["cfg"].get("self_train_gate_hi", 2.0))
        max_mult = exp_dict["cfg"].get("self_train_gate_max", None)
        gate_n = int(exp_dict["cfg"].get("self_train_gate_n", 36))
        if not raw_gate:
            return "", lo_val, hi_val, max_mult, gate_n
        if isinstance(raw_gate, dict):
            mode_str = str(raw_gate.get("mode", "near" if raw_gate.get("max_mult") == 1.0 else "epochs")).lower()
            lo_val = float(raw_gate.get("lo", raw_gate.get("min_mult", lo_val)))
            if "hi" in raw_gate:
                hi_val = float(raw_gate["hi"])
            elif mode_str in ("near", "ng", "near_epochs", "near_im", "near_both") and hi_val == 2.0:
                hi_val = 1.0
            if "max_mult" in raw_gate:
                max_mult = raw_gate["max_mult"]
            elif mode_str in ("near", "ng", "near_epochs", "near_im", "near_both") and max_mult is None:
                max_mult = 1.0
            if "n" in raw_gate or "gate_n" in raw_gate:
                gate_n = int(raw_gate.get("n", raw_gate.get("gate_n", gate_n)))
        else:
            mode_str = str(raw_gate).lower()
            if mode_str in ("near", "ng", "near_epochs", "near_im", "near_both"):
                if hi_val == 2.0:
                    hi_val = 1.0
                if max_mult is None:
                    max_mult = 1.0
        if max_mult is not None:
            max_mult = float(max_mult)
        mapped = {
            "epochs": "epochs",
            "im": "im",
            "both": "both",
            "near": "epochs",
            "ng": "epochs",
            "near_epochs": "epochs",
            "near_im": "im",
            "near_both": "both",
        }[mode_str]
        return mapped, lo_val, hi_val, max_mult, gate_n

    @classmethod
    def _evaluate_gate_state(cls, exp_dict: dict[str, Any]) -> tuple[float, int, float] | None:
        gate_obj = exp_dict.get("gate")
        gate_r = exp_dict.get("gate_r")
        if gate_obj is None or gate_r is None:
            return None
        _, lo_m, hi_m, max_mult, _ = cls._gate_parameters(exp_dict)
        tang_v = CovarianceMatrixOperator.project_to_tangent_vector(gate_obj["isqrt"] @ gate_r @ gate_obj["isqrt"])
        dist_val = float(np.linalg.norm(gate_obj["vectors"] - tang_v, axis=1).min() / gate_obj["scale"])
        third_idx = int(dist_val >= gate_obj["lo"]) + int(dist_val >= gate_obj["hi"])
        if gate_obj["hi"] <= gate_obj["lo"]:
            mult_val = float(lo_m if dist_val < gate_obj["lo"] else hi_m)
        else:
            mult_val = float(
                lo_m + (hi_m - lo_m) * np.clip((dist_val - gate_obj["lo"]) / (gate_obj["hi"] - gate_obj["lo"]), 0.0, 1.0)
            )
        if max_mult is not None:
            mult_val = min(mult_val, float(max_mult))
        return dist_val, third_idx, mult_val

    @staticmethod
    def _refresh_group_pre_bn_cache(exp_dict: dict[str, Any]) -> None:
        grp = exp_dict["group"]
        if not exp_dict["buf"]:
            exp_dict["hbuf"] = []
        else:
            cached_h = []
            with torch.no_grad():
                for idx0 in range(0, len(exp_dict["buf"]), 4):
                    cat_x = torch.cat(exp_dict["buf"][idx0 : idx0 + 4], dim=0)
                    cat_h = grp.pre_bn(cat_x)
                    cached_h.extend(u_slice.clone() for u_slice in cat_h.split(1, dim=0))
            exp_dict["hbuf"] = cached_h
        exp_dict["hbuf_ver"] = grp.version

    @classmethod
    def _refresh_tent_buffer_with_ag3(
        cls,
        exp_dict: dict[str, Any],
        adapter_a: np.ndarray,
        target_device: torch.device,
        skip_cache_refresh: bool = False,
    ) -> None:
        if not exp_dict["buf_raw_w"]:
            return
        eye_c = np.eye(adapter_a.shape[0], dtype=adapter_a.dtype)
        if np.array_equal(adapter_a, eye_c):
            exp_dict["buf"] = [u_raw.clone() for u_raw in exp_dict["buf_raw_w"]]
        else:
            at_dev = torch.as_tensor(adapter_a, dtype=torch.float32, device=target_device)
            raw_cat = torch.cat(exp_dict["buf_raw_w"], dim=0)
            adapted_cat = torch.matmul(at_dev, raw_cat)
            exp_dict["buf"] = [u_ad.clone() for u_ad in adapted_cat.split(1, dim=0)]
        exp_dict["group"].version += 1
        exp_dict["group_opt"] = None
        if not skip_cache_refresh:
            cls._refresh_group_pre_bn_cache(exp_dict)

    def _run_expert_self_training(
        self,
        exp_dict: dict[str, Any],
        blk_rt: dict[str, Any],
        whiten_mats: np.ndarray,
    ) -> None:
        ag3_st = blk_rt.get("ag3")
        step_scale = float(self._cnn_config.self_train_epoch_scale)
        if ag3_st is not None and exp_dict["role"] in ("main", "second"):
            w_raw_last = exp_dict["ag3_w_raw_last"]
            if w_raw_last is None:
                w_raw_last = np.linalg.solve(ag3_st["adapter_a"], whiten_mats[-1])
            mode_ag3 = ag3_st["mode"]
            if exp_dict["role"] == "main" and ag3_st["fitted_at"] != exp_dict["t"]:
                gate_st = self._evaluate_gate_state(exp_dict) if exp_dict.get("gate") is not None else None
                d_cur = 1.0 if gate_st is None else float(gate_st[0])
                third_cur = 1 if gate_st is None else int(gate_st[1])
                mult_cur = 1.0 if gate_st is None else float(gate_st[2])
                _, _, _, max_mult_g, _ = self._gate_parameters(exp_dict)
                if max_mult_g is not None:
                    mult_cur = min(mult_cur, float(max_mult_g))
                ag3_st["last_third"] = third_cur
                will_run_core = mode_ag3 in (
                    "w_then_st",
                    "gated_wtst",
                    "row_l2_gated_wtst",
                    "row_l2_wtst",
                    "median_gate",
                ) or (mode_ag3 == "near_skip_st" and third_cur != 0)

                if mode_ag3 == "median_gate" and d_cur < 1.0:
                    ag3_st["fitted_at"] = exp_dict["t"]
                else:
                    past_n = min(len(exp_dict["raw"]) - 1, len(exp_dict["pseudo"]))
                    if past_n >= 1:
                        past_p = np.stack(exp_dict["pseudo"][:past_n], axis=0)
                        conf_v, cls_v = past_p.max(axis=1), past_p.argmax(axis=1)
                        usable_m = conf_v >= exp_dict["self_train_conf"]
                        if not all(exp_dict["ok"][:past_n]):
                            usable_m &= np.asarray(exp_dict["ok"][:past_n], dtype=bool)
                        keep_idx = np.flatnonzero(usable_m)
                        if len(keep_idx) >= exp_dict["self_train_min"]:
                            y_keep = cls_v[keep_idx]
                            if len(set(y_keep.tolist())) >= 3:
                                raw_keep = np.stack([exp_dict["raw"][idx_k] for idx_k in keep_idx], axis=0)
                                with torch.no_grad():
                                    raw_k_dev = torch.as_tensor(raw_keep, dtype=torch.float64, device=self._device)
                                    w_raw_dev = torch.as_tensor(w_raw_last, dtype=torch.float64, device=self._device)
                                    x_tgt_dev = torch.matmul(w_raw_dev, raw_k_dev).to(torch.float32)
                                cls_counts = np.bincount(y_keep, minlength=3)
                                w_keep = 1.0 / cls_counts[y_keep]
                                w_keep = w_keep / w_keep.mean()
                                im_w_fit = float(exp_dict["self_train_im_w"])
                                g_mode, _, _, _, _ = self._gate_parameters(exp_dict)
                                if g_mode in ("im", "both"):
                                    im_w_fit *= mult_cur
                                epochs_fit = int(ag3_st["epochs"])
                                decay_fit = float(ag3_st["decay"])
                                if mode_ag3 in ("gated_wtst", "row_l2_gated_wtst"):
                                    epochs_fit = max(1, int(np.floor(epochs_fit * mult_cur + 0.5)))
                                    decay_fit = decay_fit / max(mult_cur * mult_cur, 0.25)
                                new_adapter = GroupedEnsembleConvolutionalBackbone.fit_ag3_spatial_adapter(
                                    shallow_group=exp_dict["group"],
                                    fbcnet_group=blk_rt["second"]["group"],
                                    fbmod_group=None,
                                    whitened_targets=x_tgt_dev,
                                    pseudo_labels=y_keep,
                                    sample_weights=w_keep,
                                    initial_adapter=ag3_st["adapter_a"],
                                    epochs=epochs_fit,
                                    learning_rate=float(ag3_st["lr"]),
                                    weight_decay=decay_fit,
                                    im_weight=im_w_fit,
                                    label_smoothing=float(exp_dict["label_smoothing"]),
                                    batch_size=int(exp_dict["self_train_bs"]),
                                    random_seed=4096 * int(exp_dict["t"]) + int(exp_dict["self_train_seed"]),
                                    target_device=self._device,
                                    weight_shallow=float(ag3_st["w_sh"]),
                                    weight_fbcnet=float(ag3_st["w_fb2"]),
                                    weight_fbmod=0.0,
                                    row_l2_normalize=(mode_ag3 in ("row_l2_gated_wtst", "row_l2_wtst")),
                                    step_scale=step_scale,
                                )
                                changed_flag = not np.array_equal(new_adapter, ag3_st["adapter_a"])
                                ag3_st["adapter_a"] = new_adapter
                                ag3_st["fitted_at"] = exp_dict["t"]
                                if changed_flag:
                                    self._refresh_tent_buffer_with_ag3(
                                        exp_dict,
                                        new_adapter,
                                        self._device,
                                        skip_cache_refresh=will_run_core,
                                    )
                                    self._refresh_tent_buffer_with_ag3(
                                        blk_rt["second"],
                                        new_adapter,
                                        self._device,
                                        skip_cache_refresh=will_run_core,
                                    )
            whiten_mats[-1] = ag3_st["adapter_a"] @ w_raw_last
            if mode_ag3 == "w_only" or (mode_ag3 == "near_skip_st" and ag3_st.get("last_third", 1) == 0):
                exp_dict["group"].configure_tent_gradients(enable_tent=(exp_dict["tta_lr"] > 0.0))
                return

        past_n = min(len(exp_dict["raw"]) - 1, len(exp_dict["pseudo"]))
        if past_n < 1:
            return
        past_p = np.stack(exp_dict["pseudo"][:past_n], axis=0)
        conf_v, cls_v = past_p.max(axis=1), past_p.argmax(axis=1)
        usable_m = conf_v >= exp_dict["self_train_conf"]
        if not all(exp_dict["ok"][:past_n]):
            usable_m &= np.asarray(exp_dict["ok"][:past_n], dtype=bool)
        keep_idx = np.flatnonzero(usable_m)
        if len(keep_idx) < exp_dict["self_train_min"]:
            return
        y_keep = cls_v[keep_idx]
        if len(set(y_keep.tolist())) < 3:
            return
        cls_counts = np.bincount(y_keep, minlength=3)
        w_keep = 1.0 / cls_counts[y_keep]
        w_keep = w_keep / w_keep.mean()

        raw_keep = np.stack([exp_dict["raw"][idx_k] for idx_k in keep_idx], axis=0)
        with torch.no_grad():
            raw_dev = torch.as_tensor(raw_keep, dtype=torch.float64, device=self._device)
            w_dev = torch.as_tensor(whiten_mats[-1], dtype=torch.float64, device=self._device)
            xt_sess = torch.matmul(w_dev, raw_dev).to(torch.float32)
            if exp_dict["self_train_replay"] and exp_dict["replay"] is not None:
                rx_arr, ry_arr = exp_dict["replay"]
                rx_dev = torch.as_tensor(rx_arr, dtype=torch.float32, device=self._device)
                xt_all = torch.cat([xt_sess, rx_dev], dim=0)
                y_all = np.concatenate([y_keep, ry_arr])
                w_all = np.concatenate([w_keep, np.ones(len(ry_arr), dtype=np.float64)])
            else:
                xt_all = xt_sess
                y_all = y_keep
                w_all = w_keep

        epochs_st = int(exp_dict["self_train_epochs"])
        im_w_st = float(exp_dict["self_train_im_w"])
        g_mode, _, _, max_mult_g, _ = self._gate_parameters(exp_dict)
        if g_mode:
            gate_st = self._evaluate_gate_state(exp_dict)
            mult_st = 1.0 if gate_st is None else float(gate_st[2])
            if max_mult_g is not None:
                mult_st = min(mult_st, float(max_mult_g))
            if g_mode in ("epochs", "both"):
                epochs_st = max(1, int(np.floor(epochs_st * mult_st + 0.5)))
            if g_mode in ("im", "both"):
                im_w_st = im_w_st * mult_st
        if int(exp_dict["t"]) < 66:
            epochs_st = max(1, min(int(epochs_st), 3))

        sess_mask = (np.arange(len(y_all)) < len(keep_idx)) if im_w_st > 0.0 else None
        st_seed = 4096 * (int(exp_dict["t"]) + 1000 * int(exp_dict["self_train_seed"]))
        exp_dict["group"].run_grouped_replay_self_training(
            combined_windows=xt_all,
            combined_labels=y_all,
            combined_weights=w_all,
            session_mask=sess_mask,
            epochs=epochs_st,
            learning_rate=float(exp_dict["self_train_lr"]),
            batch_size=int(exp_dict["self_train_bs"]),
            label_smoothing=float(exp_dict["label_smoothing"]),
            train_params_mode=str(exp_dict["self_train_params"]),
            im_weight=im_w_st,
            random_seed=st_seed,
            step_scale=step_scale,
            allow_tf32=True,
            enable_tent_after=(exp_dict["tta_lr"] > 0.0),
        )
        exp_dict["group_opt"] = None
        self._refresh_group_pre_bn_cache(exp_dict)

    def _step_single_expert(
        self,
        exp_dict: dict[str, Any],
        blk_rt: dict[str, Any] | None,
        x_cent_f64: np.ndarray,
        skip_flag: bool,
        skip_self_training: bool,
        pseudo_chrono_adapter: Any,
        pc_bn_active: bool,
        pc_bn_tau: float,
        pc_after_freeze: bool,
        pre_covs: np.ndarray | None = None,
        pre_whiten: np.ndarray | None = None,
        pre_z: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if exp_dict["self_train_at"] or exp_dict["ts_refit"] > 0.0:
            exp_dict["ok"].append(not skip_flag)
        if exp_dict["self_train_at"]:
            exp_dict["raw"].append(x_cent_f64.astype(np.float32))

        if pre_covs is not None:
            covs_cur = pre_covs
        elif exp_dict["bands"]:
            covs_cur = SpectralSignalOperator.compute_band_covariances(
                x_cent_f64[None],
                [list(bp) for bp in exp_dict["bands"]],
                sampling_rate_hz=self._sampling_rate_hz,
            )[0]
        else:
            covs_cur = CovarianceMatrixOperator.ledoit_wolf_covariance(x_cent_f64[None])
        exp_dict["last_covs"] = covs_cur

        keep_stat = not skip_flag
        if keep_stat and not (exp_dict["ea_freeze"] and exp_dict["t"] >= exp_dict["ea_freeze"]):
            exp_dict["cov_sum"] = exp_dict["cov_sum"] + covs_cur
            exp_dict["n_ref"] += 1
        exp_dict["t"] += 1

        ref_cov = (exp_dict["ea_prior"] * exp_dict["r0"] + exp_dict["cov_sum"]) / (
            exp_dict["ea_prior"] + float(exp_dict["n_ref"])
        )
        whiten_raw = pre_whiten if pre_whiten is not None else self.cnn_whitener(ref_cov, exp_dict["ea_scale"], True)
        exp_dict["last_w_raw"] = whiten_raw

        ag3_st = None if blk_rt is None else blk_rt.get("ag3")
        w_raw_last = None
        whiten_eff = whiten_raw
        if ag3_st is not None and exp_dict["role"] in ("main", "second"):
            w_raw_last = whiten_raw[-1].copy()
            exp_dict["ag3_w_raw_last"] = w_raw_last
            whiten_eff = whiten_raw.copy()
            whiten_eff[-1] = ag3_st["adapter_a"] @ w_raw_last

        if exp_dict["self_train_gate"] and exp_dict["self_train_at"]:
            _, _, _, _, gate_n = self._gate_parameters(exp_dict)
            if exp_dict["n_ref"] <= gate_n:
                exp_dict["gate_r"] = np.array(ref_cov[-1], dtype=np.float64)

        if (
            exp_dict["t"] in exp_dict["self_train_at"]
            and exp_dict["t"] not in exp_dict["st_done"]
            and not skip_self_training
            and blk_rt is not None
        ):
            exp_dict["st_done"].add(exp_dict["t"])
            self._run_expert_self_training(exp_dict, blk_rt, whiten_eff)

        out_comps: dict[str, np.ndarray] = {}
        if exp_dict["use_ts"]:
            if pre_z is not None:
                z_std = pre_z
            else:
                wb_ts = whiten_eff[:-1]
                tang_raw = CovarianceMatrixOperator.project_to_tangent_vector(wb_ts @ covs_cur[:-1] @ wb_ts).reshape(-1)
                z_std = (tang_raw - exp_dict["scaler_mean"]) / exp_dict["scaler_scale"]
            exp_dict["last_z"] = z_std
            if exp_dict["ts_refit"] > 0.0 and exp_dict["coef"] is not None:
                past_t0 = exp_dict["t"] - 1
                if (
                    exp_dict["t"] in exp_dict["ts_refit_at"]
                    and len(exp_dict["pseudo"]) == past_t0
                    and len(exp_dict["ts_z"]) == past_t0
                ):
                    ok_m = np.asarray(exp_dict["ok"][:past_t0], dtype=bool)
                    if int(ok_m.sum()) >= 2:
                        p_stack = np.stack(exp_dict["pseudo"][:past_t0], axis=0)[ok_m]
                        y_ps = p_stack.argmax(axis=1)
                        wt_ps = p_stack[np.arange(len(y_ps)), y_ps]
                        if len(np.unique(y_ps)) >= 2:
                            exp_dict["ts_coef"], exp_dict["ts_intercept"] = self.refit_logistic_regression_toward(
                                exp_dict["coef"],
                                exp_dict["intercept"],
                                np.stack(exp_dict["ts_z"][:past_t0], axis=0)[ok_m],
                                y_ps,
                                wt_ps,
                                lr_c=exp_dict["lr_c"],
                                pull_strength=exp_dict["ts_refit"],
                            )
                exp_dict["ts_z"].append(z_std)
            out_comps["ts"] = CovarianceMatrixOperator.stable_softmax(
                z_std @ exp_dict["ts_coef"].T + exp_dict["ts_intercept"]
            )

        xw_np = (whiten_eff[-1] @ x_cent_f64)[None]
        xw_dev = torch.as_tensor(np.ascontiguousarray(xw_np, dtype=np.float32), dtype=torch.float32, device=self._device)
        grp = exp_dict["group"]

        if not pc_bn_active and not exp_dict["bn_frozen"]:
            grp.adapt = 0.0 if not keep_stat else (1.0 / (15.0 + float(exp_dict["n_ref"])))
        elif pc_bn_active and not exp_dict["bn_frozen"]:
            grp.adapt = 0.10
        if exp_dict["bn_freeze"] and not exp_dict["bn_frozen"] and exp_dict["t"] >= exp_dict["bn_freeze"]:
            grp.adapt = 0.0
            exp_dict["bn_frozen"] = True

        with torch.no_grad():
            h_cur = grp.pre_bn(xw_dev)

        tta_stop = exp_dict["tta_freeze"]
        if exp_dict["tta_lr"] > 0.0 and keep_stat and not (tta_stop and exp_dict["t"] >= tta_stop):
            exp_dict["buf"] = (exp_dict["buf"] + [xw_dev])[-exp_dict["tta_buffer"] :]
            if not exp_dict["hbuf"]:
                exp_dict["hbuf_ver"] = grp.version
            exp_dict["hbuf"] = (exp_dict["hbuf"] + [h_cur])[-exp_dict["tta_buffer"] :]
            if exp_dict["t"] >= exp_dict["tta_warmup"]:
                exp_dict["group_opt"] = grp.step_tent_entropy_minimization(
                    pre_bn_buffer=torch.cat(exp_dict["hbuf"], dim=0),
                    tent_optimizer=exp_dict["group_opt"],
                    tent_lr=float(exp_dict["tta_lr"]),
                    diversity_weight=float(exp_dict["tta_div"]),
                    min_buffer_windows=int(exp_dict["tta_min_buf"]),
                )

        with torch.no_grad():
            seed_probs = grp.predict_seed_probabilities(
                xw_batch=xw_dev,
                pre_bn_act=h_cur,
                skip_update=skip_flag,
                bn_frozen=bool(exp_dict["bn_frozen"]),
                pseudo_chrono_adapter=pseudo_chrono_adapter if pc_bn_active else None,
                local_pc_bn_cache=exp_dict["local_pc_bn"],
                pc_bn_tau=pc_bn_tau,
                pc_after_freeze=pc_after_freeze,
            )
            out_comps["cnn"] = seed_probs.mean(dim=0).cpu().numpy().astype(np.float64)

        if (
            ag3_st is not None
            and w_raw_last is not None
            and exp_dict["tta_lr"] > 0.0
            and keep_stat
            and not (tta_stop and exp_dict["t"] >= tta_stop)
        ):
            xw_raw_dev = torch.as_tensor(
                (w_raw_last @ x_cent_f64)[None],
                dtype=torch.float32,
                device=self._device,
            )
            exp_dict["buf_raw_w"] = (exp_dict["buf_raw_w"] + [xw_raw_dev])[-exp_dict["tta_buffer"] :]
        return out_comps

    def _step_builtin_pseudochrono_ea(
        self,
        x_cent_f64: np.ndarray,
        is_bad_window: bool,
        is_frozen_tta: bool,
        cur_u: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        main_a = self._block_a["main"]
        covs_t = SpectralSignalOperator.compute_band_covariances(
            x_cent_f64[None],
            [list(bp) for bp in main_a["bands"]],
            sampling_rate_hz=self._sampling_rate_hz,
        )[0]
        ea_st = self._builtin_pc_ea_state
        if not ea_st:
            ea_st.update({"u_ok": [], "cov_ok": [], "cov_cum": None, "n_ok": 0})
        if not is_bad_window and not is_frozen_tta:
            ea_st["cov_cum"] = covs_t.copy() if ea_st["cov_cum"] is None else (ea_st["cov_cum"] + covs_t)
            ea_st["n_ok"] += 1
            ea_st["u_ok"].append(np.asarray(cur_u, dtype=np.float64).copy())
            ea_st["cov_ok"].append(covs_t.copy())
            if len(ea_st["u_ok"]) > 512:
                ea_st["u_ok"].pop(0)
                ea_st["cov_ok"].pop(0)
        ea_tau = float(self._classical_config.pseudochrono_ea_tau)
        if ea_tau <= 0.0:
            return covs_t, None, None
        r0_mat = main_a["r0"]
        if ea_st["u_ok"]:
            u_mat = np.stack(ea_st["u_ok"], axis=0)
            sq_d = np.sum((u_mat - np.asarray(cur_u, dtype=np.float64)[None, :]) ** 2, axis=1)
            w_ea = np.exp(-0.5 * sq_d / (float(u_mat.shape[1]) * (ea_tau * ea_tau)))
            w_sum = float(w_ea.sum())
            c_ker = np.tensordot(w_ea, np.stack(ea_st["cov_ok"], axis=0), axes=(0, 0))
        else:
            w_sum = 0.0
            c_ker = np.zeros_like(r0_mat)
        c_glob = (ea_st["cov_cum"] / float(ea_st["n_ok"])) if (ea_st["n_ok"] > 0 and ea_st["cov_cum"] is not None) else r0_mat
        r_t = (6.0 * r0_mat + 2.0 * c_glob + c_ker) / (6.0 + 2.0 + w_sum)
        w_pc = self.cnn_whitener(r_t, main_a["ea_scale"], True)
        tang_z = CovarianceMatrixOperator.project_to_tangent_vector(w_pc[:-1] @ covs_t[:-1] @ w_pc[:-1]).reshape(-1)
        z_pc = (tang_z - main_a["scaler_mean"]) / main_a["scaler_scale"]
        return covs_t, w_pc, z_pc

    def _combine_block_readout(
        self,
        blk_rt: dict[str, Any],
        comps: dict[str, np.ndarray],
        csp_organ: SessionCspLdaRiemannianOrgan | None,
        spec_organ: ParticipantSpecialistRiemannianOrgan | None,
    ) -> np.ndarray:
        w_ts = float(self._cnn_config.weight_ts_lr)
        w_cnn2 = float(self._cnn_config.weight_fbcnet)
        w_main = 1.0 - w_ts - w_cnn2
        ts_term = w_ts * np.log(comps["ts"] + 1e-12) if (w_ts > 0.0 and "ts" in comps) else 0.0
        logp = ts_term + w_cnn2 * np.log(comps["cnn2"] + 1e-12) + w_main * np.log(comps["cnn"] + 1e-12)

        spec_term = (
            spec_organ.compute_routed_term_and_update(logp, balance_prior=10.0)
            if spec_organ is not None
            else np.zeros(3, dtype=np.float64)
        )
        if csp_organ is not None:
            logp = logp + csp_organ.additive_log_term()
        if spec_organ is not None:
            logp = logp + spec_term

        p_u = CovarianceMatrixOperator.stable_softmax(logp)
        blk_rt["p_sum"] = blk_rt["p_sum"] + p_u
        blk_rt["p_t"] += 1
        p_bal = self.balance_probability_step(p_u, blk_rt["p_sum"], blk_rt["p_t"], prior_pseudo_count=10.0)

        if csp_organ is not None:
            csp_organ.observe_teacher_probabilities(p_bal)
        for fk in ("main", "second"):
            exp_d = blk_rt.get(fk)
            if exp_d is not None and (exp_d["self_train_at"] or exp_d["ts_refit"] > 0.0):
                exp_d["pseudo"].append(p_bal.copy())

        ro_w3 = float(self._cnn_config.readout_weight_fbmodnet) if "cnn3" in comps else 0.0
        ro_w4 = float(self._cnn_config.readout_weight_shallow49) if "cnn4" in comps else 0.0
        if ro_w3 > 0.0 or ro_w4 > 0.0:
            w_u = 1.0 - ro_w3 - ro_w4
            lp_ro = w_u * np.log(np.clip(p_u, 1e-12, 1.0))
            if ro_w3 > 0.0:
                lp_ro = lp_ro + ro_w3 * np.log(np.clip(comps["cnn3"], 1e-12, 1.0))
            if ro_w4 > 0.0:
                lp_ro = lp_ro + ro_w4 * np.log(np.clip(comps["cnn4"], 1e-12, 1.0))
            q_ro = CovarianceMatrixOperator.stable_softmax(lp_ro)
            blk_rt["ro_sum"] = blk_rt["ro_sum"] + q_ro
            return q_ro
        return p_u

    def step_cnn40_and_stoff(
        self,
        cleaned_window: np.ndarray,
        dc_centered_window: np.ndarray | None = None,
        is_bad_window: bool = False,
        is_frozen_tta: bool = False,
        skip_self_training: bool = False,
        pseudo_chrono_adapter: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_clean_f64 = np.asarray(cleaned_window, dtype=np.float64)
        if not np.isfinite(x_clean_f64).all():
            x_clean_f64 = np.nan_to_num(x_clean_f64, nan=0.0, posinf=500.0, neginf=-500.0)
        x_cent_f64 = (
            (x_clean_f64 - x_clean_f64.mean(axis=-1, keepdims=True))
            if dc_centered_window is None
            else np.asarray(dc_centered_window, dtype=np.float64)
        )

        bad_flag = bool(is_bad_window)
        frz_flag = bool(is_frozen_tta)
        skip_main = bad_flag or frz_flag
        next_t_1based = int(self._block_a["main"]["t"]) + 1
        gate_thresh = float(self._cnn_config.self_train_teacher_gate_threshold)
        if gate_thresh > 0.0 and next_t_1based in self._cnn_config.self_train_checkpoints:
            src_hist = self._multimodal_teacher_hist if self._has_external_teacher else self._cnn40_bal_hist
            st_gate_ok = len(src_hist) >= 8 and float(np.mean(np.stack(src_hist, axis=0).max(axis=1))) >= gate_thresh
            for blk_rt in (self._block_a, self._block_b):
                for fk in ("main", "second"):
                    if blk_rt.get(fk) is not None:
                        blk_rt[fk]["skip_st_core"] = not st_gate_ok

        pc_bn_tau = (
            float(getattr(pseudo_chrono_adapter, "bn_tau", self._classical_config.pseudochrono_bn_tau))
            if pseudo_chrono_adapter is not None
            else 0.0
        )
        pc_bn_active = pseudo_chrono_adapter is not None and pc_bn_tau > 0.0
        pc_after_freeze = bool(getattr(pseudo_chrono_adapter, "pc_after_freeze", True))

        cov_pc = None
        w_pc = None
        z_pc = None
        pre_csp_a = None
        known_a: dict | None = None
        if pseudo_chrono_adapter is not None:
            if hasattr(pseudo_chrono_adapter, "step_window_ea_and_bn"):
                cov_pc, w_pc, z_pc = pseudo_chrono_adapter.step_window_ea_and_bn(
                    x_clean_f64, x_cent_f64, bad_flag, frz_flag, self
                )
            else:
                cur_u = getattr(pseudo_chrono_adapter, "cur_u", getattr(pseudo_chrono_adapter, "current_embedding", None))
                if cur_u is not None:
                    cov_pc, w_pc, z_pc = self._step_builtin_pseudochrono_ea(x_cent_f64, bad_flag, frz_flag, cur_u)

            if cov_pc is not None:
                known_a = {tuple(bp): cov_pc[idx_b] for idx_b, bp in enumerate(self._block_a["main"]["bands"])}
                known_a[None] = cov_pc[-1]
                if self._csp_a is not None:
                    pre_csp_a = self._csp_a.step_causal_window(x_clean_f64, bad_flag, known_covariances=known_a)
                    if self._csp_a.last_band_covariances is not None:
                        for idx_c, bp_c in enumerate(SessionCspLdaRiemannianOrgan.SIX_CSP_BANDS_HZ):
                            known_a[tuple(bp_c)] = self._csp_a.last_band_covariances[idx_c]

        comps_a = self._step_single_expert(
            self._block_a["main"],
            self._block_a,
            x_cent_f64,
            skip_main,
            skip_self_training,
            pseudo_chrono_adapter,
            pc_bn_active,
            pc_bn_tau,
            pc_after_freeze,
            pre_covs=cov_pc,
            pre_whiten=w_pc,
            pre_z=z_pc,
        )
        cov_last_a = self._block_a["main"]["last_covs"][-1:]
        w_last_a = self._block_a["main"]["last_w_raw"][-1:]
        comps_a["cnn2"] = self._step_single_expert(
            self._block_a["second"],
            self._block_a,
            x_cent_f64,
            skip_main,
            skip_self_training,
            pseudo_chrono_adapter,
            pc_bn_active,
            pc_bn_tau,
            pc_after_freeze,
            pre_covs=cov_last_a,
            pre_whiten=w_last_a,
        )["cnn"]
        if self._block_a["third"] is not None:
            comps_a["cnn3"] = self._step_single_expert(
                self._block_a["third"],
                self._block_a,
                x_cent_f64,
                skip_main,
                skip_self_training,
                pseudo_chrono_adapter,
                pc_bn_active,
                pc_bn_tau,
                pc_after_freeze,
                pre_covs=cov_last_a,
                pre_whiten=w_last_a,
            )["cnn"]
        if self._block_a["fourth"] is not None:
            comps_a["cnn4"] = self._step_single_expert(
                self._block_a["fourth"],
                self._block_a,
                x_cent_f64,
                skip_main,
                skip_self_training,
                pseudo_chrono_adapter,
                pc_bn_active,
                pc_bn_tau,
                pc_after_freeze,
                pre_covs=cov_last_a,
                pre_whiten=w_last_a,
            )["cnn"]

        if known_a is None:
            covs_ma = self._block_a["main"]["last_covs"]
            known_a = {tuple(bp): covs_ma[idx_b] for idx_b, bp in enumerate(self._block_a["main"]["bands"])}
            known_a[None] = covs_ma[-1]
        if self._csp_a is not None:
            comps_a["csp"] = (
                pre_csp_a
                if pre_csp_a is not None
                else self._csp_a.step_causal_window(x_clean_f64, bad_flag, known_covariances=known_a)
            )
        if self._spec_a is not None:
            comps_a["spec"] = self._spec_a.step_causal_window(x_clean_f64, skip_main, known_covariances=known_a)

        covs_full_a = self._block_a["main"]["last_covs"]
        w_raw_full_a = self._block_a["main"]["last_w_raw"]
        z_raw_a = self._block_a["main"]["last_z"]
        comps_b = self._step_single_expert(
            self._block_b["main"],
            self._block_b,
            x_cent_f64,
            skip_main,
            skip_self_training,
            pseudo_chrono_adapter,
            pc_bn_active,
            pc_bn_tau,
            pc_after_freeze,
            pre_covs=covs_full_a,
            pre_whiten=w_raw_full_a,
            pre_z=z_raw_a,
        )
        comps_b["cnn2"] = self._step_single_expert(
            self._block_b["second"],
            self._block_b,
            x_cent_f64,
            skip_main,
            skip_self_training,
            pseudo_chrono_adapter,
            pc_bn_active,
            pc_bn_tau,
            pc_after_freeze,
            pre_covs=cov_last_a,
            pre_whiten=w_last_a,
        )["cnn"]
        if self._block_b["third"] is not None:
            comps_b["cnn3"] = self._step_single_expert(
                self._block_b["third"],
                self._block_b,
                x_cent_f64,
                bad_flag,
                skip_self_training,
                pseudo_chrono_adapter,
                pc_bn_active,
                pc_bn_tau,
                pc_after_freeze,
                pre_covs=cov_last_a,
                pre_whiten=w_last_a,
            )["cnn"]
        if self._block_b["fourth"] is not None:
            comps_b["cnn4"] = self._step_single_expert(
                self._block_b["fourth"],
                self._block_b,
                x_cent_f64,
                bad_flag,
                skip_self_training,
                pseudo_chrono_adapter,
                pc_bn_active,
                pc_bn_tau,
                pc_after_freeze,
                pre_covs=cov_last_a,
                pre_whiten=w_last_a,
            )["cnn"]
        if self._csp_b is not None and self._csp_a is not None:
            comps_b["csp"] = self._csp_b.step_reusing_peer_covariances(self._csp_a, bad_flag)
        if self._spec_b is not None and self._spec_a is not None:
            comps_b["spec"] = self._spec_b.step_reusing_peer_scores(self._spec_a)

        u_a = self._combine_block_readout(self._block_a, comps_a, self._csp_a, self._spec_a)
        u_b = self._combine_block_readout(self._block_b, comps_b, self._csp_b, self._spec_b)
        u_cnn40_probs = CovarianceMatrixOperator.stable_softmax(
            0.5 * self.centered_log_probabilities(u_a) + 0.5 * self.centered_log_probabilities(u_b)
        )
        self._cnn_p_t += 1
        self._cnn_p_sum = self._cnn_p_sum + u_cnn40_probs
        p_cnn40_bal = self.balance_probability_step(u_cnn40_probs, self._cnn_p_sum, self._cnn_p_t, prior_pseudo_count=10.0)
        self._cnn40_bal_hist.append(p_cnn40_bal.copy())

        u_k25_stoff_a = self._step_single_expert(
            self._sh_stoff_a,
            None,
            x_cent_f64,
            skip_main,
            True,
            pseudo_chrono_adapter,
            pc_bn_active,
            pc_bn_tau,
            pc_after_freeze,
            pre_covs=cov_last_a,
            pre_whiten=w_last_a,
        )["cnn"]
        u_k25_stoff_b = self._step_single_expert(
            self._sh_stoff_b,
            None,
            x_cent_f64,
            skip_main,
            True,
            pseudo_chrono_adapter,
            pc_bn_active,
            pc_bn_tau,
            pc_after_freeze,
            pre_covs=cov_last_a,
            pre_whiten=w_last_a,
        )["cnn"]
        u_k25_stoff = CovarianceMatrixOperator.stable_softmax(
            0.5 * self.centered_log_probabilities(u_k25_stoff_a) + 0.5 * self.centered_log_probabilities(u_k25_stoff_b)
        )
        u_k49_a = comps_a.get("cnn4", u_k25_stoff_a)
        u_k49_b = comps_b.get("cnn4", u_k25_stoff_b)
        u_k49_stoff = CovarianceMatrixOperator.stable_softmax(
            0.5 * self.centered_log_probabilities(u_k49_a) + 0.5 * self.centered_log_probabilities(u_k49_b)
        )
        u_shallow_stoff_raw = CovarianceMatrixOperator.stable_softmax(
            0.5 * self.centered_log_probabilities(u_k25_stoff) + 0.5 * self.centered_log_probabilities(u_k49_stoff)
        )
        return u_cnn40_probs, u_shallow_stoff_raw

    def record_multimodal_teacher_probs(self, teacher_probs: np.ndarray, is_bad_window: bool = False) -> None:
        _ = is_bad_window
        tchr_copy = np.asarray(teacher_probs, dtype=np.float64).copy()
        self._has_external_teacher = True
        self._multimodal_teacher_hist.append(tchr_copy)
        for blk_rt in (self._block_a, self._block_b):
            for fk in ("main", "second"):
                exp_d = blk_rt.get(fk)
                if exp_d is not None and exp_d["pseudo"]:
                    exp_d["pseudo"][-1] = tchr_copy.copy()

    def step_causal_window(self, cleaned_window: np.ndarray, is_artifact_flagged: bool) -> np.ndarray:
        u_cnn40, u_stoff = self.step_cnn40_and_stoff(
            cleaned_window=cleaned_window,
            is_bad_window=bool(is_artifact_flagged),
        )
        p_cnn40_bal = self._cnn40_bal_hist[-1]
        w_stoff = float(self._cnn_config.shallow_stoff_anchor_weight)
        if w_stoff <= 0.0:
            return p_cnn40_bal
        return CovarianceMatrixOperator.stable_softmax(
            (1.0 - w_stoff) * self.centered_log_probabilities(p_cnn40_bal)
            + w_stoff * self.centered_log_probabilities(u_stoff)
        )

    @staticmethod
    def _build_cnn_from_cfg_dict(parsed_cfg: dict[str, Any], chan_count: int, time_count: int, class_count: int) -> nn.Module:
        arch_str = str(parsed_cfg["cnn_arch"])
        drop_p = float(parsed_cfg.get("dropout", 0.5))
        if arch_str == "shallow":
            sh_opts = parsed_cfg.get("shallow", {})
            return ShallowTemporalConvolutionalBackbone(
                n_chans=chan_count,
                n_times=time_count,
                n_classes=class_count,
                n_filters=int(sh_opts.get("n_filters", 40)),
                kernel=int(sh_opts.get("kernel", 25)),
                pool=int(sh_opts.get("pool", 35)),
                stride=int(sh_opts.get("stride", 7)),
                dropout=drop_p,
            )
        if arch_str == "fbcnet":
            fb_opts = parsed_cfg.get("fbcnet", {})
            fb_bands = parsed_cfg.get("fb_bands", DualBlockCnnEnsembleCnnOrgan.NINE_FBC_BANDS_HZ)
            return FilterBankConvolutionalBackbone(
                n_chans=chan_count,
                n_times=time_count,
                n_classes=class_count,
                bands=fb_bands,
                m=int(parsed_cfg.get("fb_m", 32)),
                taps=int(fb_opts.get("taps", 121)),
                n_segments=int(fb_opts.get("n_segments", 4)),
                dropout=drop_p,
                sampling_rate_hz=float(parsed_cfg.get("sfreq", 120.0)),
                sub4=bool(fb_opts.get("sub4", False)),
                modpool=bool(fb_opts.get("modpool", False)),
            )
        raise ValueError(f"Unsupported CNN architecture in bundle: {arch_str}")

    @classmethod
    def _load_expert_spec_from_prefix(cls, bundle_dir: Path, prefix: str) -> dict[str, Any] | None:
        weights_file = bundle_dir / f"{prefix}weights.npz"
        if not weights_file.exists():
            return None
        with np.load(weights_file, allow_pickle=False) as archive:
            parsed_cfg = json.loads(str(archive["cfg"]))
            r0_arr = archive["R0"].copy()
            has_ts = "coef" in archive.files
            sm_arr = archive["scaler_mean"].copy() if has_ts else None
            ss_arr = archive["scaler_scale"].copy() if has_ts else None
            coef_arr = archive["coef"].copy() if has_ts else None
            inter_arr = archive["intercept"].copy() if has_ts else None
            class_count = len(coef_arr) if coef_arr is not None else 3
            if "n_nets" in archive.files:
                net_count = int(archive["n_nets"])
            elif parsed_cfg.get("n_nets"):
                net_count = int(parsed_cfg["n_nets"])
            else:
                net_count = int(parsed_cfg.get("n_seeds", 20)) * max(1, int(parsed_cfg.get("snapshot_cycles", 1)))
            gate_dict = None
            if "gate_vectors" in archive.files:
                g_stats = archive["gate_stats"]
                gate_dict = {
                    "isqrt": archive["gate_isqrt"].copy(),
                    "vectors": archive["gate_vectors"].copy(),
                    "scale": float(g_stats[0]),
                    "lo": float(g_stats[1]),
                    "hi": float(g_stats[2]),
                }
            replay_tup = (
                (archive["replay_x"].copy(), archive["replay_y"].copy()) if "replay_x" in archive.files else None
            )

        chan_count = int(r0_arr.shape[-1])
        time_count = int(parsed_cfg.get("n_times", 480))
        cnns: list[nn.Module] = []
        for idx_k in range(net_count if parsed_cfg.get("use_cnn", True) else 0):
            net_mod = cls._build_cnn_from_cfg_dict(parsed_cfg, chan_count, time_count, class_count)
            pt_state = torch.load(bundle_dir / f"{prefix}cnn_{idx_k}.pt", map_location="cpu", weights_only=True)
            net_mod.load_state_dict(pt_state, strict=True)
            cnns.append(net_mod)
        return {
            "cfg": parsed_cfg,
            "r0": r0_arr,
            "scaler_mean": sm_arr,
            "scaler_scale": ss_arr,
            "coef": coef_arr,
            "intercept": inter_arr,
            "replay": replay_tup,
            "gate": gate_dict,
            "cnns": cnns,
        }

    @classmethod
    def load_from_bundle_directory(
        cls,
        bundle_directory: Path,
        cnn_config: CnnEnsembleSectionConfiguration,
        classical_config: ClassicalHeadsSectionConfiguration,
        execution_device: str = "cpu",
    ) -> DualBlockCnnEnsembleCnnOrgan:
        resolved_dir = ParameterTypeValidator.require_path(bundle_directory, "bundle_directory", must_exist=True)
        mix_meta = json.loads((resolved_dir / "mix.json").read_text(encoding="utf-8"))

        block_a_bundle = {
            "main": cls._load_expert_spec_from_prefix(resolved_dir, ""),
            "second": cls._load_expert_spec_from_prefix(resolved_dir, "cnn2_"),
            "third": cls._load_expert_spec_from_prefix(resolved_dir, "cnn3_"),
            "fourth": cls._load_expert_spec_from_prefix(resolved_dir, "cnn4_"),
        }
        block_b_bundle = {
            "main": cls._load_expert_spec_from_prefix(resolved_dir, "blkB_"),
            "second": cls._load_expert_spec_from_prefix(resolved_dir, "blkB_cnn2_"),
            "third": cls._load_expert_spec_from_prefix(resolved_dir, "blkB_cnn3_"),
            "fourth": cls._load_expert_spec_from_prefix(resolved_dir, "blkB_cnn4_"),
        }
        if block_a_bundle["main"] is None or block_a_bundle["second"] is None:
            raise FileNotFoundError(f"Missing Block A CNN weights in {resolved_dir}")
        if block_b_bundle["main"] is None or block_b_bundle["second"] is None:
            raise FileNotFoundError(f"Missing Block B CNN weights in {resolved_dir}")

        if block_b_bundle["main"]["replay"] is None and block_a_bundle["main"]["replay"] is not None:
            block_b_bundle["main"]["replay"] = block_a_bundle["main"]["replay"]
        if block_b_bundle["second"]["replay"] is None and block_a_bundle["second"]["replay"] is not None:
            block_b_bundle["second"]["replay"] = block_a_bundle["second"]["replay"]

        checkpoints = tuple(int(ckpt) for ckpt in cnn_config.self_train_checkpoints)
        csp_a = csp_b = None
        if mix_meta.get("csp") is not None and (resolved_dir / "csp.npz").exists():
            csp_a = SessionCspLdaRiemannianOrgan.load_from_npz(
                resolved_dir / "csp.npz",
                csp_weight=classical_config.session_csp_weight,
                adaptive_shrinkage=classical_config.session_csp_shrinkage_n0,
                refit_checkpoints=checkpoints,
            )
            csp_b = SessionCspLdaRiemannianOrgan.load_from_npz(
                resolved_dir / "csp.npz",
                csp_weight=classical_config.session_csp_weight,
                adaptive_shrinkage=classical_config.session_csp_shrinkage_n0,
                refit_checkpoints=checkpoints,
            )

        spec_a = spec_b = None
        if mix_meta.get("specialists") is not None and (resolved_dir / "specialists.npz").exists():
            spec_a = ParticipantSpecialistRiemannianOrgan.load_from_npz(
                resolved_dir / "specialists.npz",
                specialist_weight=classical_config.specialist_weight,
                kappa=classical_config.specialist_kappa,
                gap_delta=classical_config.specialist_gap_delta,
                prior_n0=classical_config.specialist_prior_n0,
            )
            spec_b = ParticipantSpecialistRiemannianOrgan.load_from_npz(
                resolved_dir / "specialists.npz",
                specialist_weight=classical_config.specialist_weight,
                kappa=classical_config.specialist_kappa,
                gap_delta=classical_config.specialist_gap_delta,
                prior_n0=classical_config.specialist_prior_n0,
            )

        return cls(
            block_a_bundle=block_a_bundle,
            block_b_bundle=block_b_bundle,
            csp_organ_a=csp_a,
            csp_organ_b=csp_b,
            specialist_organ_a=spec_a,
            specialist_organ_b=spec_b,
            mix_metadata=mix_meta,
            cnn_config=cnn_config,
            classical_config=classical_config,
            execution_device=execution_device,
        )

    @staticmethod
    def _save_expert_with_prefix(target_dir: Path, prefix: str, exp_dict: dict[str, Any], include_replay: bool) -> list[Path]:
        written_files: list[Path] = []
        npz_payload: dict[str, np.ndarray] = {
            "cfg": np.array(json.dumps(exp_dict["saved_cfg"])),
            "R0": exp_dict["r0"].astype(np.float64),
            "n_nets": np.array(len(exp_dict["cnns"])),
        }
        if exp_dict["coef"] is not None:
            npz_payload["scaler_mean"] = exp_dict["scaler_mean"].astype(np.float64)
            npz_payload["scaler_scale"] = exp_dict["scaler_scale"].astype(np.float64)
            npz_payload["coef"] = exp_dict["coef"].astype(np.float64)
            npz_payload["intercept"] = exp_dict["intercept"].astype(np.float64)
        if include_replay and exp_dict["replay"] is not None:
            npz_payload["replay_x"] = exp_dict["replay"][0].astype(np.float32)
            npz_payload["replay_y"] = exp_dict["replay"][1].astype(np.int64)
        if exp_dict["gate"] is not None and exp_dict["role"] in ("main", "second"):
            g_obj = exp_dict["gate"]
            npz_payload["gate_isqrt"] = g_obj["isqrt"].astype(np.float64)
            npz_payload["gate_vectors"] = g_obj["vectors"].astype(np.float64)
            npz_payload["gate_stats"] = np.array([g_obj["scale"], g_obj["lo"], g_obj["hi"]], dtype=np.float64)
        w_path = target_dir / f"{prefix}weights.npz"
        np.savez(w_path, **npz_payload)
        written_files.append(w_path)

        for idx_k, (full_sd, bn_sd) in enumerate(zip(exp_dict["train_states"], exp_dict["bn_train"])):
            merged_sd = {**full_sd, **bn_sd}
            pt_path = target_dir / f"{prefix}cnn_{idx_k}.pt"
            torch.save({k_s: t_v.detach().cpu() for k_s, t_v in merged_sd.items()}, pt_path)
            written_files.append(pt_path)
        return written_files

    def save_organ_artifacts(self, target_directory: Path) -> list[Path]:
        resolved_dir = ParameterTypeValidator.require_path(target_directory, "target_directory")
        resolved_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        prefix_table = (
            (self._block_a, "main", "", True),
            (self._block_a, "second", "cnn2_", True),
            (self._block_a, "third", "cnn3_", False),
            (self._block_a, "fourth", "cnn4_", False),
            (self._block_b, "main", "blkB_", False),
            (self._block_b, "second", "blkB_cnn2_", False),
            (self._block_b, "third", "blkB_cnn3_", False),
            (self._block_b, "fourth", "blkB_cnn4_", False),
        )
        for blk_rt, fk, pfx, keep_rep in prefix_table:
            if blk_rt.get(fk) is not None:
                written.extend(self._save_expert_with_prefix(resolved_dir, pfx, blk_rt[fk], include_replay=keep_rep))

        if self._csp_a is not None:
            written.extend(self._csp_a.save_organ_artifacts(resolved_dir))
        if self._spec_a is not None:
            written.extend(self._spec_a.save_organ_artifacts(resolved_dir))

        mix_out = copy.deepcopy(self._mix_metadata)
        mix_path = resolved_dir / "mix.json"
        mix_path.write_text(json.dumps(mix_out, indent=2), encoding="utf-8")
        written.append(mix_path)
        return written

    @staticmethod
    def _fit_class_map_lbfgs(sess_cls_covs: np.ndarray, tgt_cls_covs: np.ndarray, shrink: float = 0.01) -> np.ndarray:
        chan_len = sess_cls_covs.shape[-1]
        eye_c = np.eye(chan_len, dtype=np.float64)
        denom_norm = max(float((tgt_cls_covs ** 2).sum()), 1e-12)

        def _obj_and_jac(flat_m: np.ndarray) -> tuple[float, np.ndarray]:
            mat_m = flat_m.reshape(chan_len, chan_len)
            err_mat = mat_m @ sess_cls_covs @ mat_m.T - tgt_cls_covs
            diff_eye = mat_m - eye_c
            loss_f = float((err_mat ** 2).sum() / denom_norm + shrink * (diff_eye ** 2).sum() / float(chan_len))
            grad_m = 4.0 * (err_mat @ mat_m @ sess_cls_covs).sum(axis=0) / denom_norm + (2.0 * shrink / float(chan_len)) * diff_eye
            return loss_f, grad_m.reshape(-1)

        opt_out = minimize(_obj_and_jac, eye_c.reshape(-1), jac=True, method="L-BFGS-B", options={"maxiter": 250, "gtol": 1e-8})
        return opt_out.x.reshape(chan_len, chan_len)

    @classmethod
    def _fit_residual_augmentation_maps(
        cls,
        aligned_broadband_covs: np.ndarray,
        labels: np.ndarray,
        sessions: np.ndarray,
    ) -> np.ndarray:
        chan_len = aligned_broadband_covs.shape[-1]
        unique_sess = sorted(set(str(s_id) for s_id in sessions.tolist()))
        sess_class_means: dict[str, np.ndarray] = {}
        for sess_id in unique_sess:
            sess_mask = sessions == sess_id
            y_s = labels[sess_mask]
            if len(np.unique(y_s)) < 3:
                continue
            log_covs = CovarianceMatrixOperator.matrix_logarithm(aligned_broadband_covs[sess_mask])
            sess_class_means[sess_id] = np.stack(
                [CovarianceMatrixOperator.matrix_exponential(log_covs[y_s == cls_k].mean(axis=0)) for cls_k in range(3)],
                axis=0,
            )
        valid_keys = sorted(sess_class_means)
        if len(valid_keys) < 2:
            return np.eye(chan_len, dtype=np.float32)[None]
        pair_list = list(permutations(valid_keys, 2))
        if len(pair_list) > 64:
            rng_sub = np.random.default_rng(0)
            chosen_idx = np.sort(rng_sub.choice(len(pair_list), size=64, replace=False))
            pair_list = [pair_list[int(i_p)] for i_p in chosen_idx]
        maps = [cls._fit_class_map_lbfgs(sess_class_means[sa], sess_class_means[sb], shrink=0.01) for sa, sb in pair_list]
        return np.stack(maps, axis=0).astype(np.float32)

    @staticmethod
    def _fit_distance_gate(
        broadband_covs: np.ndarray,
        sessions: np.ndarray,
        r0_broad: np.ndarray,
        ea_prior: float = 10.0,
        freeze_n: int = 36,
    ) -> dict[str, Any]:
        unique_sess = sorted(set(str(s_id) for s_id in sessions.tolist()))
        end_refs = []
        for sess_id in unique_sess:
            idx_s = np.flatnonzero(sessions == sess_id)
            take_n = min(len(idx_s), freeze_n)
            c_sub = broadband_covs[idx_s[:take_n]]
            end_refs.append((ea_prior * r0_broad + c_sub.sum(axis=0)) / (ea_prior + float(take_n)))
        while len(end_refs) < 3:
            scale_f = 1.0 + 0.05 * float(len(end_refs) + 1)
            end_refs.append(r0_broad * scale_f)
        stacked_refs = np.stack(end_refs, axis=0)
        isqrt_r0 = CovarianceMatrixOperator.inverse_matrix_square_root(r0_broad)
        tang_vecs = CovarianceMatrixOperator.project_to_tangent_vector(isqrt_r0 @ stacked_refs @ isqrt_r0)
        sq_norms = (tang_vecs ** 2).sum(axis=1)
        dist_mat = np.sqrt(np.maximum(sq_norms[:, None] + sq_norms[None, :] - 2.0 * (tang_vecs @ tang_vecs.T), 0.0))
        np.fill_diagonal(dist_mat, np.inf)
        loo_min = dist_mat.min(axis=1)
        med_scale = max(float(np.median(loo_min)), 1e-6)
        q_lo, q_hi = np.quantile(loo_min / med_scale, [1.0 / 3.0, 2.0 / 3.0])
        return {
            "isqrt": isqrt_r0,
            "vectors": tang_vecs,
            "scale": med_scale,
            "lo": float(q_lo),
            "hi": float(q_hi),
        }

    @staticmethod
    def _compute_oof_soft_teachers(
        tang_std: np.ndarray,
        labels: np.ndarray,
        sessions: np.ndarray,
    ) -> np.ndarray:
        unique_sess = sorted(set(str(s_id) for s_id in sessions.tolist()))
        oof_probs = np.full((len(labels), 3), 1.0 / 3.0, dtype=np.float32)
        if len(unique_sess) < 2:
            lr_all = LogisticRegression(C=0.1, max_iter=2000)
            lr_all.fit(tang_std, labels)
            oof_probs[:, lr_all.classes_.astype(int)] = lr_all.predict_proba(tang_std).astype(np.float32)
            return oof_probs
        for sess_id in unique_sess:
            val_mask = sessions == sess_id
            tr_mask = ~val_mask
            y_tr = labels[tr_mask]
            if len(np.unique(y_tr)) < 2:
                continue
            lr_fold = LogisticRegression(C=0.1, max_iter=2000)
            lr_fold.fit(tang_std[tr_mask], y_tr)
            pred_f = lr_fold.predict_proba(tang_std[val_mask]).astype(np.float32)
            full_p = np.full(( int(val_mask.sum()), 3), 1e-4, dtype=np.float32)
            for loc_i, cls_k in enumerate(lr_fold.classes_):
                full_p[:, int(cls_k)] = pred_f[:, loc_i]
            full_p /= full_p.sum(axis=1, keepdims=True)
            oof_probs[val_mask] = full_p
        return oof_probs

    @staticmethod
    def _build_session_grouped_batches(
        sessions: np.ndarray,
        batch_size: int,
        bn_groups: int,
        cpu_gen: torch.Generator,
    ) -> list[np.ndarray]:
        slice_len = max(1, batch_size // bn_groups)
        chunks = []
        for sess_id in np.unique(sessions):
            idx_s = np.flatnonzero(sessions == sess_id)
            perm_s = torch.randperm(len(idx_s), generator=cpu_gen).numpy()
            idx_s = idx_s[perm_s]
            padded_len = -(-len(idx_s) // slice_len) * slice_len
            idx_s = np.resize(idx_s, padded_len)
            chunks.extend(idx_s.reshape(-1, slice_len))
        chunk_order = torch.randperm(len(chunks), generator=cpu_gen).numpy()
        ordered_chunks = [chunks[int(i_c)] for i_c in chunk_order]
        return [np.stack(ordered_chunks[i0 : i0 + bn_groups], axis=0) for i0 in range(0, len(ordered_chunks), bn_groups)]

    @staticmethod
    @contextlib.contextmanager
    def _deterministic_torch_context(enable_det: bool, seed_int: int):
        torch.manual_seed(int(seed_int))
        if not enable_det:
            yield
            return
        prev_det = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(prev_det)

    @classmethod
    def _train_single_cnn_seed(
        cls,
        net_module: nn.Module,
        whitened_x: np.ndarray,
        labels: np.ndarray,
        sessions: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        label_smoothing: float,
        swa_tail_fraction: float,
        seed_int: int,
        target_device: torch.device,
        deterministic: bool,
        residual_maps: np.ndarray | None = None,
        residual_prob: float = 0.0,
        soft_teachers: np.ndarray | None = None,
        distill_beta: float = 0.0,
        bn_groups: int = 0,
        max_shift: int = 12,
    ) -> nn.Module:
        with cls._deterministic_torch_context(deterministic, seed_int):
            cpu_gen = torch.Generator(device="cpu").manual_seed(int(seed_int))
            net_module = net_module.to(target_device)
            xt_all = torch.as_tensor(whitened_x, dtype=torch.float32, device=target_device)
            yt_all = torch.as_tensor(labels, dtype=torch.long, device=target_device)
            qt_all = (
                torch.as_tensor(soft_teachers, dtype=torch.float32, device=target_device)
                if (soft_teachers is not None and distill_beta > 0.0)
                else None
            )
            resid_dev = (
                torch.as_tensor(residual_maps, dtype=torch.float32, device=target_device)
                if (residual_maps is not None and residual_prob > 0.0)
                else None
            )

            row_count, chan_count, time_count = xt_all.shape
            unique_sess_count = len(np.unique(sessions))
            eff_groups = min(int(bn_groups), unique_sess_count, max(1, batch_size // 2)) if bn_groups > 1 else 0
            if eff_groups <= 1:
                eff_groups = 0

            if eff_groups > 0:
                sess_sizes = np.unique(sessions, return_counts=True)[1]
                slice_m = max(1, batch_size // eff_groups)
                total_batches = -(-int((-(-sess_sizes // slice_m)).sum()) // eff_groups)
            else:
                total_batches = (row_count + batch_size - 1) // batch_size

            total_steps = max(1, epochs * total_batches)
            adamw_opt = torch.optim.AdamW(net_module.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
            cos_sched = torch.optim.lr_scheduler.CosineAnnealingLR(adamw_opt, T_max=total_steps)
            ce_vec = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing), reduction="none")
            time_grid = torch.arange(time_count, device=target_device)
            bn_layers = [sub_m for sub_m in net_module.modules() if isinstance(sub_m, AdaptiveBatchNormConvolutionalBackbone)]

            swa_start = (epochs - max(1, int(round(swa_tail_fraction * epochs)))) if swa_tail_fraction > 0.0 else None
            swa_accum: list[torch.Tensor] | None = None
            swa_count = 0

            net_module.train()
            for ep_idx in range(epochs):
                if eff_groups > 0:
                    ep_batches = cls._build_session_grouped_batches(sessions, batch_size, eff_groups, cpu_gen)
                else:
                    perm_dev = torch.randperm(row_count, generator=cpu_gen).to(target_device)
                    ep_batches = [perm_dev[s0 : s0 + batch_size] for s0 in range(0, row_count, batch_size)]

                for raw_batch in ep_batches:
                    if eff_groups > 0:
                        for bn_l in bn_layers:
                            bn_l.groups = len(raw_batch)
                        batch_idx = torch.as_tensor(raw_batch.reshape(-1), dtype=torch.long, device=target_device)
                    else:
                        batch_idx = raw_batch

                    xb = xt_all[batch_idx]
                    if max_shift > 0 and time_count > 2 * max_shift:
                        shifts = torch.randint(-max_shift, max_shift + 1, (len(batch_idx), 1), device=target_device)
                        shifted_t = (time_grid[None, :] + shifts).clamp(0, time_count - 1)
                        xb = xb.gather(-1, shifted_t[:, None, :].expand(-1, chan_count, -1))
                    if resid_dev is not None and residual_prob > 0.0:
                        which_m = torch.randint(0, len(resid_dev), (len(batch_idx),), device=target_device)
                        mapped_x = resid_dev[which_m] @ xb
                        apply_m = torch.rand((len(batch_idx), 1, 1), device=target_device) < float(residual_prob)
                        xb = torch.where(apply_m, mapped_x, xb)

                    yb = yt_all[batch_idx]
                    logits = net_module(xb)
                    per_sample = ce_vec(logits, yb)
                    if qt_all is not None and distill_beta > 0.0:
                        kl_part = -(qt_all[batch_idx] * torch.log_softmax(logits, dim=-1)).sum(dim=1)
                        per_sample = (1.0 - float(distill_beta)) * per_sample + float(distill_beta) * kl_part
                    loss_t = per_sample.mean()
                    adamw_opt.zero_grad(set_to_none=True)
                    loss_t.backward()
                    adamw_opt.step()
                    cos_sched.step()

                if swa_start is not None and ep_idx >= swa_start:
                    with torch.no_grad():
                        cur_params = [p_t.detach().clone() for p_t in net_module.parameters()]
                        if swa_accum is None:
                            swa_accum = cur_params
                        else:
                            for acc_p, cur_p in zip(swa_accum, cur_params):
                                acc_p.add_(cur_p)
                        swa_count += 1

            for bn_l in bn_layers:
                bn_l.groups = 1
            if swa_accum is not None and swa_count > 0:
                with torch.no_grad():
                    for p_t, acc_p in zip(net_module.parameters(), swa_accum):
                        p_t.copy_(acc_p / float(swa_count))
                net_module.recompute_running_bn_statistics(xt_all, mini_batch_size=batch_size)
            return net_module.eval().cpu()

    @classmethod
    def fit_from_corpus(
        cls,
        corpus_container: WindowedCorpusStreamContainer,
        cnn_config: CnnEnsembleSectionConfiguration,
        classical_config: ClassicalHeadsSectionConfiguration,
        execution_device: str = "cpu",
        master_seed: int = 20261121,
    ) -> DualBlockCnnEnsembleCnnOrgan:
        resolved_dev = "cuda" if (execution_device == "auto" and torch.cuda.is_available()) else (
            "cpu" if execution_device == "auto" else execution_device
        )
        target_dev = torch.device(resolved_dev)

        raw_x = np.asarray(corpus_container.eeg_windows, dtype=np.float64)
        demeaned_x = raw_x - raw_x.mean(axis=-1, keepdims=True)
        labels = np.asarray(corpus_container.class_labels, dtype=np.int64)
        sessions = np.asarray(corpus_container.session_ids).astype(str)
        sfreq = float(corpus_container.sampling_rate_hz)
        chan_count = int(raw_x.shape[1])
        time_count = int(raw_x.shape[2])

        covs_7b = SpectralSignalOperator.compute_band_covariances(
            demeaned_x,
            [list(bp) for bp in cls.SIX_MAIN_BANDS_HZ],
            sampling_rate_hz=sfreq,
        )
        r0_full = covs_7b.mean(axis=0)
        r0_broad = r0_full[-1:].copy()

        sess_refs = np.empty_like(covs_7b)
        for sess_id in np.unique(sessions):
            sm = sessions == sess_id
            sess_refs[sm] = covs_7b[sm].mean(axis=0)
        whiten_all = CovarianceMatrixOperator.inverse_matrix_square_root(sess_refs)
        xw_broad = (whiten_all[:, -1] @ demeaned_x).astype(np.float32)
        aligned_broad_covs = whiten_all[:, -1] @ covs_7b[:, -1] @ whiten_all[:, -1]

        tang_6b = CovarianceMatrixOperator.project_to_tangent_vector(
            whiten_all[:, :-1] @ covs_7b[:, :-1] @ whiten_all[:, :-1]
        ).reshape(len(raw_x), -1)
        scaler_mean = tang_6b.mean(axis=0)
        scaler_scale = tang_6b.std(axis=0) + 1e-8
        tang_std = (tang_6b - scaler_mean) / scaler_scale

        ts_lr = LogisticRegression(C=0.1, max_iter=3000)
        ts_lr.fit(tang_std, labels)
        coef_ts = np.zeros((3, tang_std.shape[1]), dtype=np.float64)
        inter_ts = np.full(3, -10.0, dtype=np.float64)
        for loc_i, cls_v in enumerate(ts_lr.classes_):
            coef_ts[int(cls_v)] = ts_lr.coef_[loc_i]
            inter_ts[int(cls_v)] = ts_lr.intercept_[loc_i]

        gate_dict = cls._fit_distance_gate(covs_7b[:, -1], sessions, r0_broad[0], ea_prior=10.0, freeze_n=36)
        resid_maps = cls._fit_residual_augmentation_maps(aligned_broad_covs, labels, sessions)
        soft_q = cls._compute_oof_soft_teachers(tang_std, labels, sessions)

        rng_rep = np.random.default_rng(int(master_seed))
        rep_indices = []
        for cls_k in range(3):
            cls_pos = np.flatnonzero(labels == cls_k)
            if len(cls_pos) > 0:
                take_k = min(int(cnn_config.replay_windows_per_class), len(cls_pos))
                rep_indices.append(rng_rep.choice(cls_pos, size=take_k, replace=False))
        rep_idx = np.sort(np.concatenate(rep_indices)) if rep_indices else np.arange(min(len(labels), 6))
        replay_tup = (xw_broad[rep_idx].astype(np.float32), labels[rep_idx].astype(np.int64))

        seeds_n = int(cnn_config.seeds_per_family)
        checkpoints_list = [int(ckpt) for ckpt in cnn_config.self_train_checkpoints]

        def _fit_one_block(seed_offset: int) -> dict[str, Any]:
            base_s = int(master_seed) + int(seed_offset)
            sh25_cnns, fb9_cnns, fbm_cnns, sh49_cnns = [], [], [], []
            for s_idx in range(seeds_n):
                net_sh25 = ShallowTemporalConvolutionalBackbone(
                    n_chans=chan_count,
                    n_times=time_count,
                    n_classes=3,
                    kernel=25,
                    dropout=cnn_config.dropout_rate,
                )
                sh25_cnns.append(
                    cls._train_single_cnn_seed(
                        net_sh25,
                        xw_broad,
                        labels,
                        sessions,
                        epochs=cnn_config.shallow25_epochs,
                        batch_size=cnn_config.batch_size,
                        learning_rate=cnn_config.learning_rate,
                        weight_decay=cnn_config.weight_decay,
                        label_smoothing=cnn_config.label_smoothing,
                        swa_tail_fraction=cnn_config.swa_tail_fraction,
                        seed_int=base_s + s_idx,
                        target_device=target_dev,
                        deterministic=cnn_config.deterministic_training,
                        residual_maps=resid_maps,
                        residual_prob=cnn_config.residual_aug_probability,
                    )
                )

                net_fb9 = FilterBankConvolutionalBackbone(
                    n_chans=chan_count,
                    n_times=time_count,
                    n_classes=3,
                    bands=cls.NINE_FBC_BANDS_HZ,
                    dropout=cnn_config.dropout_rate,
                    sampling_rate_hz=sfreq,
                    sub4=False,
                    modpool=False,
                )
                fb9_cnns.append(
                    cls._train_single_cnn_seed(
                        net_fb9,
                        xw_broad,
                        labels,
                        sessions,
                        epochs=cnn_config.fbcnet_epochs,
                        batch_size=cnn_config.batch_size,
                        learning_rate=cnn_config.learning_rate,
                        weight_decay=cnn_config.weight_decay,
                        label_smoothing=cnn_config.label_smoothing,
                        swa_tail_fraction=cnn_config.swa_tail_fraction,
                        seed_int=base_s + 100 + s_idx,
                        target_device=target_dev,
                        deterministic=cnn_config.deterministic_training,
                        residual_maps=resid_maps,
                        residual_prob=cnn_config.residual_aug_probability,
                    )
                )

                net_fbm = FilterBankConvolutionalBackbone(
                    n_chans=chan_count,
                    n_times=time_count,
                    n_classes=3,
                    bands=cls.NINE_FBC_BANDS_HZ,
                    dropout=cnn_config.dropout_rate,
                    sampling_rate_hz=sfreq,
                    sub4=True,
                    modpool=True,
                )
                fbm_cnns.append(
                    cls._train_single_cnn_seed(
                        net_fbm,
                        xw_broad,
                        labels,
                        sessions,
                        epochs=cnn_config.fbmodnet_distill_epochs,
                        batch_size=cnn_config.batch_size,
                        learning_rate=cnn_config.fbmodnet_learning_rate,
                        weight_decay=cnn_config.weight_decay,
                        label_smoothing=cnn_config.label_smoothing,
                        swa_tail_fraction=cnn_config.swa_tail_fraction,
                        seed_int=base_s + 200 + s_idx,
                        target_device=target_dev,
                        deterministic=cnn_config.deterministic_training,
                        soft_teachers=soft_q,
                        distill_beta=cnn_config.distill_kl_beta,
                        bn_groups=cnn_config.distill_bn_groups,
                    )
                )

                net_sh49 = ShallowTemporalConvolutionalBackbone(
                    n_chans=chan_count,
                    n_times=time_count,
                    n_classes=3,
                    kernel=49,
                    dropout=cnn_config.dropout_rate,
                )
                sh49_cnns.append(
                    cls._train_single_cnn_seed(
                        net_sh49,
                        xw_broad,
                        labels,
                        sessions,
                        epochs=cnn_config.shallow49_distill_epochs,
                        batch_size=cnn_config.batch_size,
                        learning_rate=cnn_config.learning_rate,
                        weight_decay=cnn_config.weight_decay,
                        label_smoothing=cnn_config.label_smoothing,
                        swa_tail_fraction=cnn_config.swa_tail_fraction,
                        seed_int=base_s + 300 + s_idx,
                        target_device=target_dev,
                        deterministic=cnn_config.deterministic_training,
                        soft_teachers=soft_q,
                        distill_beta=cnn_config.distill_kl_beta,
                        bn_groups=cnn_config.distill_bn_groups,
                    )
                )

            common_fb = [[b0, b1] for b0, b1 in cls.NINE_FBC_BANDS_HZ]
            return {
                "main": {
                    "cfg": {
                        "cnn_arch": "shallow",
                        "bands": [[b0, b1] for b0, b1 in cls.SIX_MAIN_BANDS_HZ],
                        "n_seeds": seeds_n,
                        "n_times": time_count,
                        "sfreq": sfreq,
                        "dropout": cnn_config.dropout_rate,
                        "ea": "online",
                        "ea_prior": 10.0,
                        "ea_scale": "none",
                        "ea_cov": "lw",
                        "adapt_bn": 0.1,
                        "tta_lr": cnn_config.tent_learning_rate,
                        "tta_buffer": cnn_config.tent_buffer_length,
                        "tta_freeze": cnn_config.tent_freeze_window,
                        "tta_min_buf": 2,
                        "tta_div": 1.0,
                        "self_train_at": checkpoints_list,
                        "self_train_epochs": cnn_config.self_train_epochs_shallow,
                        "self_train_lr": 0.001,
                        "self_train_bs": 32,
                        "self_train_conf": 0.0,
                        "self_train_min": 8,
                        "self_train_im_w": 1.0,
                        "self_train_params": "spatial_head",
                        "self_train_replay": cnn_config.replay_windows_per_class,
                        "self_train_gate": "near",
                        "self_train_seed": 0,
                        "label_smoothing": cnn_config.label_smoothing,
                        "ts_refit": 3.0,
                        "ts_refit_at": [66, 96, 126, 156],
                        "lr_C": 0.1,
                        "use_ts": True,
                        "use_cnn": True,
                        "shallow": {"kernel": 25},
                        "fbcnet": {},
                        "fb_bands": common_fb,
                        "fb_m": 32,
                    },
                    "r0": r0_full.copy(),
                    "scaler_mean": scaler_mean.copy(),
                    "scaler_scale": scaler_scale.copy(),
                    "coef": coef_ts.copy(),
                    "intercept": inter_ts.copy(),
                    "replay": replay_tup,
                    "gate": copy.deepcopy(gate_dict),
                    "cnns": sh25_cnns,
                },
                "second": {
                    "cfg": {
                        "cnn_arch": "fbcnet",
                        "bands": [],
                        "n_seeds": seeds_n,
                        "n_times": time_count,
                        "sfreq": sfreq,
                        "dropout": cnn_config.dropout_rate,
                        "ea": "online",
                        "ea_prior": 10.0,
                        "ea_scale": "none",
                        "ea_cov": "lw",
                        "adapt_bn": 0.1,
                        "tta_lr": 0.0,
                        "self_train_at": checkpoints_list,
                        "self_train_epochs": cnn_config.self_train_epochs_fbcnet,
                        "self_train_lr": 0.001,
                        "self_train_bs": 32,
                        "self_train_conf": 0.0,
                        "self_train_min": 8,
                        "self_train_im_w": 0.0,
                        "self_train_params": "all",
                        "self_train_replay": cnn_config.replay_windows_per_class,
                        "self_train_gate": "near",
                        "self_train_seed": 0,
                        "label_smoothing": cnn_config.label_smoothing,
                        "use_ts": False,
                        "use_cnn": True,
                        "shallow": {},
                        "fbcnet": {},
                        "fb_bands": common_fb,
                        "fb_m": 32,
                    },
                    "r0": r0_broad.copy(),
                    "scaler_mean": None,
                    "scaler_scale": None,
                    "coef": None,
                    "intercept": None,
                    "replay": replay_tup,
                    "gate": copy.deepcopy(gate_dict),
                    "cnns": fb9_cnns,
                },
                "third": {
                    "cfg": {
                        "cnn_arch": "fbcnet",
                        "bands": [],
                        "n_seeds": seeds_n,
                        "n_times": time_count,
                        "sfreq": sfreq,
                        "dropout": cnn_config.dropout_rate,
                        "ea": "online",
                        "ea_prior": 10.0,
                        "ea_scale": "none",
                        "ea_cov": "lw",
                        "adapt_bn": 0.1,
                        "tta_lr": 0.0,
                        "self_train_at": [],
                        "self_train_replay": 0,
                        "self_train_gate": "",
                        "label_smoothing": cnn_config.label_smoothing,
                        "use_ts": False,
                        "use_cnn": True,
                        "shallow": {},
                        "fbcnet": {"taps": 121, "n_segments": 4, "sub4": True, "modpool": True},
                        "fb_bands": common_fb,
                        "fb_m": 32,
                    },
                    "r0": r0_broad.copy(),
                    "scaler_mean": None,
                    "scaler_scale": None,
                    "coef": None,
                    "intercept": None,
                    "replay": None,
                    "gate": None,
                    "cnns": fbm_cnns,
                },
                "fourth": {
                    "cfg": {
                        "cnn_arch": "shallow",
                        "bands": [],
                        "n_seeds": seeds_n,
                        "n_times": time_count,
                        "sfreq": sfreq,
                        "dropout": cnn_config.dropout_rate,
                        "ea": "online",
                        "ea_prior": 10.0,
                        "ea_scale": "none",
                        "ea_cov": "lw",
                        "adapt_bn": 0.1,
                        "tta_lr": 0.0,
                        "self_train_at": [],
                        "self_train_replay": 0,
                        "self_train_gate": "",
                        "label_smoothing": cnn_config.label_smoothing,
                        "use_ts": False,
                        "use_cnn": True,
                        "shallow": {"kernel": 49},
                        "fbcnet": {},
                        "fb_bands": common_fb,
                        "fb_m": 32,
                    },
                    "r0": r0_broad.copy(),
                    "scaler_mean": None,
                    "scaler_scale": None,
                    "coef": None,
                    "intercept": None,
                    "replay": None,
                    "gate": None,
                    "cnns": sh49_cnns,
                },
            }

        block_a_bundle = _fit_one_block(cnn_config.block_a_seed_offset)
        block_b_bundle = _fit_one_block(cnn_config.block_b_seed_offset)

        csp_a = SessionCspLdaRiemannianOrgan.fit_from_corpus(
            corpus_container,
            csp_weight=classical_config.session_csp_weight,
            adaptive_shrinkage=classical_config.session_csp_shrinkage_n0,
        )
        csp_b = SessionCspLdaRiemannianOrgan.fit_from_corpus(
            corpus_container,
            csp_weight=classical_config.session_csp_weight,
            adaptive_shrinkage=classical_config.session_csp_shrinkage_n0,
        )
        spec_a = ParticipantSpecialistRiemannianOrgan.fit_from_corpus(
            corpus_container,
            specialist_weight=classical_config.specialist_weight,
            kappa=classical_config.specialist_kappa,
            gap_delta=classical_config.specialist_gap_delta,
            prior_n0=classical_config.specialist_prior_n0,
        )
        spec_b = ParticipantSpecialistRiemannianOrgan.fit_from_corpus(
            corpus_container,
            specialist_weight=classical_config.specialist_weight,
            kappa=classical_config.specialist_kappa,
            gap_delta=classical_config.specialist_gap_delta,
            prior_n0=classical_config.specialist_prior_n0,
        )

        mix_meta = {
            "w_ts": cnn_config.weight_ts_lr,
            "w_cnn2": cnn_config.weight_fbcnet,
            "balance": True,
            "balance_prior": 10.0,
            "specialists": {
                "spec_w": classical_config.specialist_weight,
                "spec_kappa": classical_config.specialist_kappa,
                "spec_delta": classical_config.specialist_gap_delta,
                "spec_n0": classical_config.specialist_prior_n0,
            },
            "csp": {
                "csp_w": classical_config.session_csp_weight,
                "csp_weighted": True,
                "csp_bands": [[b0, b1] for b0, b1 in SessionCspLdaRiemannianOrgan.SIX_CSP_BANDS_HZ],
                "csp_shrink_adaptive": classical_config.session_csp_shrinkage_n0,
                "csp_soft_cov": True,
            },
            "w_cnn3": 0.0,
            "w_cnn4": 0.0,
            "readout_cnn3_w": cnn_config.readout_weight_fbmodnet,
            "readout_cnn4_w": cnn_config.readout_weight_shallow49,
            "ag3": {
                "mode": "row_l2_gated_wtst",
                "epochs": cnn_config.ag3_epochs,
                "lr": cnn_config.ag3_learning_rate,
                "decay": cnn_config.ag3_weight_decay,
            },
        }
        return cls(
            block_a_bundle=block_a_bundle,
            block_b_bundle=block_b_bundle,
            csp_organ_a=csp_a,
            csp_organ_b=csp_b,
            specialist_organ_a=spec_a,
            specialist_organ_b=spec_b,
            mix_metadata=mix_meta,
            cnn_config=cnn_config,
            classical_config=classical_config,
            execution_device=execution_device,
        )
