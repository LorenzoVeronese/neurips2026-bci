"""Abstract base class for pretrained backbone checkpoint acquirers and safe float16 .npz packers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

from neurips_bci.acquisition.Acquirer import Acquirer
from neurips_bci.configuration.section_configurations.DatasetSectionConfiguration import DatasetSectionConfiguration
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class CheckpointAcquirer(Acquirer, ABC):
    """Provides safe pickle-free float16 state-dict serialization, head stripping, and weight_norm folding."""

    _HEAD_PREFIXES: tuple[str, ...] = (
        "final_layer.",
        "lm_head",
        "proj_out.",
        "predictor.",
        "reconstructor.",
        "classifier.",
    )

    def __init__(
        self,
        dataset_configuration: DatasetSectionConfiguration,
        acquirer_source: str,
    ) -> None:
        if not isinstance(dataset_configuration, DatasetSectionConfiguration):
            raise TypeError(
                f"dataset_configuration must be DatasetSectionConfiguration, got {type(dataset_configuration).__name__}"
            )
        super().__init__(acquirer_source=acquirer_source)
        self._dataset_configuration = dataset_configuration

    @property
    def dataset_configuration(self) -> DatasetSectionConfiguration:
        return self._dataset_configuration

    @classmethod
    def strip_projection_heads_and_fold_weight_norm(
        cls,
        raw_tensor_map: dict[str, np.ndarray],
        extra_excluded_prefixes: tuple[str, ...] = (),
    ) -> dict[str, np.ndarray]:
        """Remove LM/classification heads and fold PyTorch weight_norm parametrizations into plain weights."""
        blocked_prefixes = cls._HEAD_PREFIXES + tuple(extra_excluded_prefixes)
        filtered_map: dict[str, np.ndarray] = {}
        wn_pairs: dict[str, dict[str, np.ndarray]] = {}

        for tensor_key, arr_obj in raw_tensor_map.items():
            if tensor_key.startswith(blocked_prefixes):
                continue
            if ".parametrizations.weight.original0" in tensor_key:
                base_prefix = tensor_key.replace(".parametrizations.weight.original0", "")
                wn_pairs.setdefault(base_prefix, {})["g"] = np.asarray(arr_obj, dtype=np.float64)
            elif ".parametrizations.weight.original1" in tensor_key:
                base_prefix = tensor_key.replace(".parametrizations.weight.original1", "")
                wn_pairs.setdefault(base_prefix, {})["v"] = np.asarray(arr_obj, dtype=np.float64)
            elif tensor_key.endswith(".weight_g"):
                base_prefix = tensor_key[: -len(".weight_g")]
                wn_pairs.setdefault(base_prefix, {})["g"] = np.asarray(arr_obj, dtype=np.float64)
            elif tensor_key.endswith(".weight_v"):
                base_prefix = tensor_key[: -len(".weight_v")]
                wn_pairs.setdefault(base_prefix, {})["v"] = np.asarray(arr_obj, dtype=np.float64)
            else:
                filtered_map[tensor_key] = np.asarray(arr_obj)

        for base_prefix, pair_dict in wn_pairs.items():
            if "g" not in pair_dict or "v" not in pair_dict:
                raise ValueError(f"Incomplete weight_norm pair for {base_prefix}")
            mag_g = pair_dict["g"]
            dir_v = pair_dict["v"]
            if mag_g.ndim < dir_v.ndim:
                expanded_shape = (mag_g.shape[0],) + (1,) * (dir_v.ndim - 1)
                mag_g = mag_g.reshape(expanded_shape)
            norm_axes = tuple(ax_i for ax_i, dim_len in enumerate(mag_g.shape) if dim_len == 1)
            v_norm = np.sqrt(np.sum(dir_v ** 2, axis=norm_axes, keepdims=True))
            folded_weight = (mag_g * (dir_v / np.maximum(v_norm, 1e-12))).astype(np.float32)
            filtered_map[f"{base_prefix}.weight"] = folded_weight

        return filtered_map

    @classmethod
    def write_fp16_checkpoint_npz(
        cls,
        raw_tensor_map: dict[str, np.ndarray],
        destination_npz_path: str | Path,
        config_json_text: str | None = None,
        extra_excluded_prefixes: tuple[str, ...] = (),
    ) -> Path:
        """Fold weight_norm, strip heads, cast weights to float16 (keeping BN running stats fp32), and save .npz."""
        resolved_dst = ParameterTypeValidator.require_path(destination_npz_path, "destination_npz_path")
        resolved_dst.parent.mkdir(parents=True, exist_ok=True)
        cleaned_map = cls.strip_projection_heads_and_fold_weight_norm(
            raw_tensor_map, extra_excluded_prefixes=extra_excluded_prefixes
        )
        packed_arrays: dict[str, np.ndarray] = {}
        for tensor_key, tensor_arr in cleaned_map.items():
            if "running_" in tensor_key:
                packed_arrays[tensor_key] = tensor_arr.astype(np.float32)
            elif "num_batches" in tensor_key or np.issubdtype(tensor_arr.dtype, np.integer):
                packed_arrays[tensor_key] = tensor_arr.astype(np.int64)
            else:
                packed_arrays[tensor_key] = tensor_arr.astype(np.float16)
        if config_json_text is not None:
            packed_arrays["cfg_json"] = np.array(str(config_json_text))

        temp_npz = resolved_dst.with_name(f".{resolved_dst.name}.tmp.npz")
        np.savez_compressed(temp_npz, **packed_arrays)
        temp_npz.replace(resolved_dst)
        return resolved_dst

    @abstractmethod
    def acquire(self) -> list[Path]:
        """Download and convert all configured pretrained checkpoints into pickle-free float16 .npz files."""
