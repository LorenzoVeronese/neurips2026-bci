"""Hugging Face pretrained foundation model checkpoint acquirer and float16 .npz converter."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from neurips_bci.acquisition.checkpoint_acquirers.CheckpointAcquirer import CheckpointAcquirer
from neurips_bci.configuration.section_configurations.DatasetSectionConfiguration import DatasetSectionConfiguration
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class HuggingFaceCheckpointAcquirer(CheckpointAcquirer):
    """Downloads braindecode/{eegpt,codebrain,cbramod,mirepnet}-pretrained and exports safe fp16 .npz archives."""

    CHECKPOINT_FILENAMES: dict[str, str] = {
        "eegpt": "base_eegpt_fp16.npz",
        "codebrain": "codebrain_base_fp16.npz",
        "cbramod": "cbramod_base_fp16.npz",
        "mirepnet": "mirepnet_fp16.npz",
    }

    def __init__(self, dataset_configuration: DatasetSectionConfiguration) -> None:
        repo_summary = ",".join(
            (
                dataset_configuration.eegpt_hf_repo_id,
                dataset_configuration.codebrain_hf_repo_id,
                dataset_configuration.cbramod_hf_repo_id,
                dataset_configuration.mirepnet_hf_repo_id,
            )
        )
        super().__init__(
            dataset_configuration=dataset_configuration,
            acquirer_source=f"huggingface:{repo_summary}",
        )

    def _repo_id_by_backbone_key(self) -> dict[str, str]:
        return {
            "eegpt": self._dataset_configuration.eegpt_hf_repo_id,
            "codebrain": self._dataset_configuration.codebrain_hf_repo_id,
            "cbramod": self._dataset_configuration.cbramod_hf_repo_id,
            "mirepnet": self._dataset_configuration.mirepnet_hf_repo_id,
        }

    def acquire(self) -> list[Path]:
        produced_paths: list[Path] = []
        for backbone_key in ("eegpt", "codebrain", "cbramod", "mirepnet"):
            produced_paths.append(self.acquire_single_backbone(backbone_key))
        return produced_paths

    def acquire_single_backbone(self, backbone_key: str) -> Path:
        ParameterTypeValidator.require_choice(backbone_key, tuple(self.CHECKPOINT_FILENAMES.keys()), "backbone_key")
        out_npz_path = self._dataset_configuration.pretrained_checkpoint_directory / self.CHECKPOINT_FILENAMES[
            backbone_key
        ]
        if out_npz_path.is_file():
            with np.load(out_npz_path, allow_pickle=False) as check_npz:
                if len(check_npz.files) > 0:
                    return out_npz_path

        repo_id = self._repo_id_by_backbone_key()[backbone_key]
        snapshot_dir = self._download_hf_snapshot(backbone_key, repo_id)
        return self.convert_snapshot_directory_to_fp16_npz(backbone_key, snapshot_dir, out_npz_path)

    def _download_hf_snapshot(self, backbone_key: str, repo_id: str) -> Path:
        local_snapshot_dir = self._dataset_configuration.raw_download_directory / "hf_snapshots" / backbone_key
        if (local_snapshot_dir / "model.safetensors").is_file():
            return local_snapshot_dir
        local_snapshot_dir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_snapshot_dir),
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.md"],
        )
        if not (local_snapshot_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"model.safetensors not found in downloaded snapshot {local_snapshot_dir}")
        return local_snapshot_dir

    @classmethod
    def convert_snapshot_directory_to_fp16_npz(
        cls,
        backbone_key: str,
        snapshot_directory: str | Path,
        destination_npz_path: str | Path,
    ) -> Path:
        ParameterTypeValidator.require_choice(backbone_key, tuple(cls.CHECKPOINT_FILENAMES.keys()), "backbone_key")
        resolved_snap = ParameterTypeValidator.require_path(
            snapshot_directory, "snapshot_directory", must_exist=True
        )
        safetensors_path = resolved_snap / "model.safetensors"
        if not safetensors_path.is_file():
            raise FileNotFoundError(f"Expected {safetensors_path} to exist")

        from safetensors.torch import load_file as safetensors_load_file

        loaded_torch_tensors = safetensors_load_file(str(safetensors_path))
        numpy_tensor_map = {
            tensor_key: tensor_val.detach().cpu().float().numpy()
            for tensor_key, tensor_val in loaded_torch_tensors.items()
        }
        if backbone_key == "eegpt":
            numpy_tensor_map = {
                tensor_key: tensor_arr
                for tensor_key, tensor_arr in numpy_tensor_map.items()
                if tensor_key.startswith("target_encoder.") or tensor_key == "chans_id"
            }

        cfg_file = resolved_snap / "config.json"
        cfg_json_str = cfg_file.read_text(encoding="utf-8") if cfg_file.is_file() else None
        return cls.write_fp16_checkpoint_npz(
            raw_tensor_map=numpy_tensor_map,
            destination_npz_path=destination_npz_path,
            config_json_text=cfg_json_str,
        )
