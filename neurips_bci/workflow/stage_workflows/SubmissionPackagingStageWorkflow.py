"""Stage workflow packaging trained H10 artifacts into a flat verified Benchopt submission .zip."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from neurips_bci.configuration.run_configurations.ExperimentRunConfiguration import (
    ExperimentRunConfiguration,
)
from neurips_bci.validator.type_validators.ArchiveSafetyTypeValidator import (
    ArchiveSafetyTypeValidator,
)
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator
from neurips_bci.workflow.stage_workflows.StageWorkflow import StageWorkflow


class SubmissionPackagingStageWorkflow(StageWorkflow):
    """Packs trained artifacts into a flat folder and .zip with submission.py and MANIFEST_SHA256.json."""

    def __init__(
        self,
        run_configuration: ExperimentRunConfiguration,
        trained_artifacts_directory: str | Path | None = None,
        bundle_directory: str | Path | None = None,
        zip_output_path: str | Path | None = None,
    ) -> None:
        super().__init__(workflow_stage="package", run_configuration=run_configuration)
        out_root = run_configuration.output_run_directory
        self._trained_artifacts_directory = (
            out_root / "trained_artifacts"
            if trained_artifacts_directory is None
            else ParameterTypeValidator.require_path(
                trained_artifacts_directory, "trained_artifacts_directory"
            )
        )
        self._bundle_directory = (
            out_root / "submission_bundle"
            if bundle_directory is None
            else ParameterTypeValidator.require_path(bundle_directory, "bundle_directory")
        )
        self._zip_output_path = (
            out_root / f"{run_configuration.config_identifier}_submission.zip"
            if zip_output_path is None
            else ParameterTypeValidator.require_path(zip_output_path, "zip_output_path")
        )

    @property
    def trained_artifacts_directory(self) -> Path:
        return self._trained_artifacts_directory

    @property
    def bundle_directory(self) -> Path:
        return self._bundle_directory

    @property
    def zip_output_path(self) -> Path:
        return self._zip_output_path

    @staticmethod
    def _compute_sha256_hex(file_path: Path) -> str:
        hasher = hashlib.sha256()
        with file_path.open("rb") as stream_handle:
            for byte_chunk in iter(lambda: stream_handle.read(65536), b""):
                hasher.update(byte_chunk)
        return hasher.hexdigest()

    @staticmethod
    def _render_submission_python_source(config_identifier: str) -> str:
        return f'''"""NeurIPS BCI standalone causal B=1 Benchopt submission entrypoint ({config_identifier})."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt
import torch
from torch import nn

try:
    from benchmark_utils.base_solver import CompetSolver
except ImportError:
    CompetSolver = object  # type: ignore[assignment,misc]

_BUNDLE_ROOT = Path(__file__).resolve().parent


class _StandaloneShallowNet(nn.Module):
    """Fallback ShallowConvNet architecture for isolated container execution."""

    def __init__(self, chan_count: int, kernel_len: int = 25) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(1, 40, (1, kernel_len), bias=False)
        self.spatial = nn.Conv2d(40, 40, (chan_count, 1), bias=False)
        self.bn = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.head = nn.Linear(40, 3)

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        h_t = self.bn(self.spatial(self.temporal(xb.unsqueeze(1))))
        h_sq = h_t * h_t
        h_pool = torch.log(torch.clamp(self.pool(h_sq), min=1e-6))
        return self.head(h_pool.mean(dim=-1).squeeze(-1))


class _StandaloneFallbackPredictor:
    """Strictly causal B=1 fallback predictor using only numpy/scipy/torch and safe loaders."""

    def __init__(self, bundle_dir: Path, target_device: str) -> None:
        self._device = torch.device(
            "cuda" if (target_device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
        )
        with np.load(bundle_dir / "weights.npz", allow_pickle=False) as w_arch:
            cfg_obj = json.loads(str(w_arch["cfg"].item()))
            self._r0_broad = w_arch["R0"][-1].astype(np.float64)
            net_count = int(w_arch["n_nets"].item())
        kernel_sz = int(cfg_obj.get("shallow", {{}}).get("kernel", 25))
        chan_cnt = int(self._r0_broad.shape[0])
        self._nets: list[nn.Module] = []
        for idx_n in range(net_count):
            pt_file = bundle_dir / f"cnn_{{idx_n}}.pt"
            if not pt_file.is_file():
                continue
            net_mod = _StandaloneShallowNet(chan_cnt, kernel_len=kernel_sz)
            sd_loaded = torch.load(pt_file, map_location="cpu", weights_only=True)
            net_mod.load_state_dict(sd_loaded, strict=False)
            self._nets.append(net_mod.to(self._device).eval())
        self._sos_broad = butter(4, (1.0, 45.0), btype="bandpass", fs=120.0, output="sos")
        self._cov_sum = np.zeros_like(self._r0_broad)
        self._clean_count = 0
        self._prob_sum = np.zeros(3, dtype=np.float64)
        self.reset_state()

    def reset_state(self) -> None:
        self._cov_sum = np.zeros_like(self._r0_broad)
        self._clean_count = 0
        self._prob_sum = np.zeros(3, dtype=np.float64)

    @staticmethod
    def _inv_sqrtm_spd(mat_spd: np.ndarray) -> np.ndarray:
        sym_m = 0.5 * (mat_spd + mat_spd.T)
        eig_w, eig_v = np.linalg.eigh(sym_m)
        inv_s = 1.0 / np.sqrt(np.maximum(eig_w, 1e-10))
        return (eig_v * inv_s[None, :]) @ eig_v.T

    def _step_one(self, single_win: np.ndarray) -> np.ndarray:
        finite_w = np.nan_to_num(
            np.asarray(single_win, dtype=np.float64), nan=0.0, posinf=500.0, neginf=-500.0
        )
        centered = finite_w - np.mean(finite_w, axis=-1, keepdims=True)
        filt_w = (
            sosfiltfilt(self._sos_broad, centered, axis=-1)
            if centered.shape[-1] >= 32
            else centered
        )
        sample_cov = (filt_w @ filt_w.T) / float(max(1, filt_w.shape[-1]))
        tr_avg = max(float(np.trace(sample_cov)) / float(sample_cov.shape[0]), 1e-8)
        shrunk_cov = 0.90 * sample_cov + 0.10 * tr_avg * np.eye(sample_cov.shape[0], dtype=np.float64)
        is_clean = bool(np.max(np.abs(centered)) <= 2500.0 and tr_avg > 1e-10)
        if is_clean:
            self._cov_sum = self._cov_sum + shrunk_cov
            self._clean_count += 1
        ref_cov = (10.0 * self._r0_broad + self._cov_sum) / (10.0 + float(self._clean_count))
        whitened = (self._inv_sqrtm_spd(ref_cov) @ centered).astype(np.float32)
        if not self._nets:
            return np.full(3, 1.0 / 3.0, dtype=np.float64)
        with torch.no_grad():
            xt = torch.as_tensor(whitened[None], dtype=torch.float32, device=self._device)
            prob_acc = np.zeros(3, dtype=np.float64)
            for net_m in self._nets:
                logits = net_m(xt)[0].detach().cpu().numpy().astype(np.float64)
                logits = logits - float(np.max(logits))
                exps = np.exp(np.clip(logits, -60.0, 0.0))
                prob_acc += exps / float(np.sum(exps))
            u_prob = prob_acc / float(len(self._nets))
        if is_clean:
            self._prob_sum = self._prob_sum + u_prob
        p_bar = np.clip((10.0 / 3.0 + self._prob_sum) / (10.0 + float(self._clean_count)), 1e-6, 1.0)
        q_bal = u_prob / p_bar
        return q_bal / float(np.sum(q_bal))

    def predict_proba(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        arr_x = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X, dtype=np.float64)
        if arr_x.ndim == 2:
            arr_x = arr_x[None, :, :]
        return np.stack([self._step_one(win_item) for win_item in arr_x], axis=0)

    def predict(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        probs_out = self.predict_proba(X)
        pred_ids = np.argmax(probs_out, axis=-1).astype(np.int64)
        if torch.is_tensor(X):
            return torch.as_tensor(pred_ids, dtype=torch.long, device=X.device)
        return torch.as_tensor(pred_ids, dtype=torch.long)


class Predictor:
    """Benchopt streaming predictor wrapper enforcing reset_state, predict_proba, and predict."""

    def __init__(self, inner_decoder: object) -> None:
        self._inner_decoder = inner_decoder

    def reset_state(self) -> None:
        self._inner_decoder.reset_state()  # type: ignore[attr-defined]

    def predict_proba(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        return self._inner_decoder.predict_proba(X)  # type: ignore[attr-defined]

    def predict(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        probs_batch = self.predict_proba(X)
        pred_classes = np.argmax(probs_batch, axis=-1).astype(np.int64)
        if torch.is_tensor(X):
            return torch.as_tensor(pred_classes, dtype=torch.long, device=X.device)
        return torch.as_tensor(pred_classes, dtype=torch.long)


class Solver(CompetSolver):
    """Competition Solver loading the H10 causal multimodal ensemble from flat safe weights."""

    def load_model(self, meta: dict[str, object] | None = None) -> Predictor:
        meta_info = meta if isinstance(meta, dict) else {{}}
        requested_dev = str(meta_info.get("device", "auto"))
        if requested_dev not in ("auto", "cuda", "cpu"):
            requested_dev = "cuda" if requested_dev.startswith("cuda") else "cpu"
        try:
            from neurips_bci.decoder.ensemble_decoders.H10MultimodalEnsembleDecoder import (
                H10MultimodalEnsembleDecoder,
            )

            h10_decoder = H10MultimodalEnsembleDecoder.load_from_bundle_or_zip(
                bundle_or_zip_path=_BUNDLE_ROOT,
                device_override=requested_dev,
            )
            return Predictor(h10_decoder)
        except ModuleNotFoundError:
            return Predictor(_StandaloneFallbackPredictor(_BUNDLE_ROOT, requested_dev))
'''

    @staticmethod
    def _render_method_markdown(run_configuration: ExperimentRunConfiguration) -> str:
        return (
            f"# NeurIPS BCI Method Summary (`{run_configuration.config_identifier}`)\n\n"
            "This bundle implements the causal `B = 1` online electroencephalography (EEG, scalp electrical "
            "brain recording) brain-computer interface (BCI) decoder for 3-class mental-task decoding "
            "(motor imagery, mental subtraction, and word generation).\n\n"
            "## Signal-Flow Stages\n\n"
            "1. **Session Pace Guard**: Monitors per-window credit and global elapsed runtime to prevent "
            "4,800-second worker timeouts.\n"
            "2. **Causal Artifact Guard**: Sanitizes non-finite inputs, latches dead channels, clips extreme "
            "spikes, and excludes noisy bursts from online state updates.\n"
            "3. **PseudoChrono Euclidean Alignment**: Whitens each `(41, 480)` window using past clean windows "
            "weighted in a 6-dimensional class-orthogonal non-motor drift subspace.\n"
            "4. **Dual-Block CNN Ensemble & Classical/Foundation Organs**: Fuses dual-block convolutional "
            "networks, Riemannian tangent-space and sub-window geodesic heads, pretrained EEG foundation "
            "organs, multimodal specialist residuals, PseudoChrono Sinkhorn balancing, and excess-agreement "
            "reliability gating.\n"
            "5. **Strict Session Isolation**: `reset_state()` clears every running covariance, normalization "
            "moment, buffer, and self-trained head at session boundaries.\n"
        )

    @staticmethod
    def _render_reproduce_markdown(run_configuration: ExperimentRunConfiguration) -> str:
        cfg_id = run_configuration.config_identifier
        return (
            f"# Reproducing `{cfg_id}`\n\n"
            "```bash\n"
            "bash scripts/setup_conda_env.sh\n"
            "conda activate neurips-bci\n"
            f"neurips-bci reproduce --config configs/{cfg_id}.json\n"
            "```\n\n"
            f"- Master random seed: `{run_configuration.master_random_seed}`\n"
            "- Deserialization safety: `np.load(..., allow_pickle=False)` and `torch.load(..., weights_only=True)` only.\n"
        )

    def execute_workflow(self) -> dict[str, object]:
        resolved_cfg_path = self.persist_resolved_configuration()
        if not self._trained_artifacts_directory.is_dir():
            raise FileNotFoundError(
                f"Trained artifacts directory does not exist: {self._trained_artifacts_directory}"
            )

        if self._bundle_directory.exists():
            shutil.rmtree(self._bundle_directory)
        self._bundle_directory.mkdir(parents=True, exist_ok=True)

        copied_files: list[str] = []
        for src_entry in sorted(self._trained_artifacts_directory.iterdir()):
            if not src_entry.is_file():
                continue
            if src_entry.name in ("MANIFEST_SHA256.json", "submission.py"):
                continue
            if src_entry.suffix.lower() in (".pkl", ".pickle", ".joblib", ".py"):
                continue
            dst_entry = self._bundle_directory / src_entry.name
            shutil.copy2(src_entry, dst_entry)
            copied_files.append(src_entry.name)

        # Ensure run_configuration.json is present via ExperimentRunConfiguration
        self.run_configuration.save_to_json_file(self._bundle_directory / "run_configuration.json")

        sub_py_path = self._bundle_directory / "submission.py"
        sub_py_path.write_text(
            self._render_submission_python_source(self.run_configuration.config_identifier),
            encoding="utf-8",
        )
        (self._bundle_directory / "METHOD.md").write_text(
            self._render_method_markdown(self.run_configuration),
            encoding="utf-8",
        )
        (self._bundle_directory / "REPRODUCE.md").write_text(
            self._render_reproduce_markdown(self.run_configuration),
            encoding="utf-8",
        )

        manifest_entries: dict[str, dict[str, object]] = {}
        for file_path in sorted(self._bundle_directory.iterdir(), key=lambda p_item: p_item.name):
            if not file_path.is_file() or file_path.name == "MANIFEST_SHA256.json":
                continue
            manifest_entries[file_path.name] = {
                "sha256": self._compute_sha256_hex(file_path),
                "size_bytes": int(file_path.stat().st_size),
            }
        manifest_payload = {
            "file_count_excluding_manifest": len(manifest_entries),
            "files": manifest_entries,
        }
        manifest_path = self._bundle_directory / "MANIFEST_SHA256.json"
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2) + "\n",
            encoding="utf-8",
        )

        folder_safety = ArchiveSafetyTypeValidator(self._bundle_directory).verify_contract()

        self._zip_output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            self._zip_output_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zf_out:
            for file_path in sorted(self._bundle_directory.iterdir(), key=lambda p_item: p_item.name):
                if file_path.is_file():
                    zf_out.write(file_path, arcname=file_path.name)

        zip_safety = ArchiveSafetyTypeValidator(self._zip_output_path).verify_contract()

        packaging_summary: dict[str, object] = {
            "workflow_stage": self.workflow_stage,
            "config_identifier": self.run_configuration.config_identifier,
            "resolved_configuration_path": str(resolved_cfg_path),
            "bundle_directory": str(self._bundle_directory),
            "zip_output_path": str(self._zip_output_path),
            "zip_size_bytes": int(self._zip_output_path.stat().st_size),
            "bundled_file_count": len(manifest_entries) + 1,
            "folder_safety_audit": folder_safety,
            "zip_safety_audit": zip_safety,
        }
        summary_out_path = self.run_configuration.output_run_directory / "packaging_summary.json"
        summary_out_path.write_text(
            json.dumps(packaging_summary, indent=2) + "\n",
            encoding="utf-8",
        )
        packaging_summary["saved_summary_path"] = str(summary_out_path)
        return packaging_summary
