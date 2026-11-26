"""Pure-PyTorch 12-layer Criss-Cross Transformer CBraMod foundation backbone."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from neurips_bci.backbone.foundation_backbones.FoundationBackbone import FoundationBackbone
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class CbramodFoundationBackbone(FoundationBackbone):
    """Standalone pure-PyTorch 12-layer Criss-Cross Transformer encoder compatible with `cbramod_41.npz`."""

    def __init__(
        self,
        patch_size: int = 200,
        dim_feedforward: int = 800,
        layer_count: int = 12,
        attention_heads: int = 8,
        dropout_probability: float = 0.1,
    ) -> None:
        validated_patch = ParameterTypeValidator.require_integer(patch_size, "patch_size", minimum_bound=8)
        validated_ff = ParameterTypeValidator.require_integer(dim_feedforward, "dim_feedforward", minimum_bound=8)
        validated_layers = ParameterTypeValidator.require_integer(layer_count, "layer_count", minimum_bound=1)
        validated_heads = ParameterTypeValidator.require_integer(attention_heads, "attention_heads", minimum_bound=2)
        validated_drop = ParameterTypeValidator.require_float(
            dropout_probability, "dropout_probability", minimum_bound=0.0, maximum_bound=0.95
        )
        conv_spec = (
            (25, 49, 25, 24, (5, 25)),
            (25, 3, 1, 1, (5, 25)),
            (25, 3, 1, 1, (5, 25)),
        )
        reduced_patch = validated_patch
        for _, kern_len, stride_len, pad_len, _ in conv_spec:
            reduced_patch = int((reduced_patch + 2 * pad_len - kern_len) / stride_len + 1)
        model_dim = conv_spec[-1][0] * reduced_patch
        if model_dim % 2 != 0 or validated_heads % 2 != 0:
            raise ValueError(f"CBraMod requires even model_dim ({model_dim}) and attention_heads ({validated_heads})")

        super().__init__(backbone_architecture="cbramod_crisscross_12L_200d", embedding_dimension=model_dim)
        self._patch_size = validated_patch
        self._d_model = model_dim

        proj_in_layers: list[nn.Module] = []
        prev_channels = 1
        for out_ch, kern_len, stride_len, pad_len, norm_pair in conv_spec:
            proj_in_layers.extend(
                [
                    nn.Conv2d(
                        in_channels=prev_channels,
                        out_channels=out_ch,
                        kernel_size=(1, kern_len),
                        stride=(1, stride_len),
                        padding=(0, pad_len),
                    ),
                    nn.GroupNorm(*norm_pair),
                    nn.GELU(),
                ]
            )
            prev_channels = out_ch

        patch_emb_dict = nn.ModuleDict(
            {
                "positional_encoding": nn.Sequential(
                    nn.Conv2d(
                        in_channels=model_dim,
                        out_channels=model_dim,
                        kernel_size=(19, 7),
                        stride=(1, 1),
                        padding=(9, 3),
                        groups=model_dim,
                    )
                ),
                "proj_in": nn.Sequential(*proj_in_layers),
                "spectral_proj": nn.Sequential(
                    nn.Linear(validated_patch // 2 + 1, model_dim),
                    nn.Dropout(validated_drop),
                ),
            }
        )
        patch_emb_dict.register_parameter(
            "mask_encoding", nn.Parameter(torch.zeros(validated_patch), requires_grad=False)
        )
        self.patch_embedding = patch_emb_dict

        half_dim = model_dim // 2
        half_heads = validated_heads // 2
        encoder_layers = nn.ModuleList()
        for _ in range(validated_layers):
            encoder_layers.append(
                nn.ModuleDict(
                    {
                        "self_attn_s": nn.MultiheadAttention(
                            half_dim,
                            half_heads,
                            dropout=validated_drop,
                            bias=True,
                            batch_first=True,
                        ),
                        "self_attn_t": nn.MultiheadAttention(
                            half_dim,
                            half_heads,
                            dropout=validated_drop,
                            bias=True,
                            batch_first=True,
                        ),
                        "linear1": nn.Linear(model_dim, validated_ff, bias=True),
                        "dropout": nn.Dropout(validated_drop),
                        "linear2": nn.Linear(validated_ff, model_dim, bias=True),
                        "norm1": nn.LayerNorm(model_dim, eps=1e-5),
                        "norm2": nn.LayerNorm(model_dim, eps=1e-5),
                        "dropout1": nn.Dropout(validated_drop),
                        "dropout2": nn.Dropout(validated_drop),
                    }
                )
            )
        self.encoder = nn.ModuleDict({"layers": encoder_layers})

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @classmethod
    def from_npz_archive(
        cls, npz_path: str | Path, execution_device: str | torch.device = "cpu"
    ) -> CbramodFoundationBackbone:
        resolved_npz = ParameterTypeValidator.require_path(npz_path, "npz_path", must_exist=True)
        backbone = cls()
        with np.load(resolved_npz, allow_pickle=False) as archive:
            encoder_state = {
                entry_key[5:]: torch.from_numpy(archive[entry_key].astype(np.float32))
                for entry_key in archive.files
                if entry_key.startswith("enc__")
            }
        missing_keys, unexpected_keys = backbone.load_state_dict(encoder_state, strict=False)
        allowed_unexpected = [k_str for k_str in unexpected_keys if not k_str.startswith(("proj_out.", "final_layer."))]
        if missing_keys or allowed_unexpected:
            raise ValueError(f"CBraMod state mismatch: missing={missing_keys}, unexpected={allowed_unexpected}")
        backbone = backbone.to(torch.device(execution_device)).eval()
        for param_tensor in backbone.parameters():
            param_tensor.requires_grad_(False)
        return backbone

    def export_encoder_npz_arrays(self) -> dict[str, np.ndarray]:
        return {
            f"enc__{param_key}": param_tensor.detach().cpu().to(torch.float16).numpy()
            for param_key, param_tensor in self.state_dict().items()
        }

    def encode_window_features(self, *tensor_inputs: torch.Tensor) -> torch.Tensor:
        if len(tensor_inputs) != 1:
            raise ValueError("CbramodFoundationBackbone expects a single (B, C, T) tensor")
        signal_tensor = tensor_inputs[0]
        batch_sz, chan_cnt, sample_cnt = signal_tensor.shape
        patch_cnt = sample_cnt // self._patch_size
        patched_x = signal_tensor[:, :, : patch_cnt * self._patch_size].reshape(
            batch_sz, chan_cnt, patch_cnt, self._patch_size
        )

        merged_patches = patched_x.contiguous().view(batch_sz, 1, chan_cnt * patch_cnt, self._patch_size)
        patch_emb = (
            self.patch_embedding["proj_in"](merged_patches)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_sz, chan_cnt, patch_cnt, self._d_model)
        )
        freq_mag = torch.abs(
            torch.fft.rfft(
                merged_patches.view(batch_sz * chan_cnt * patch_cnt, self._patch_size),
                dim=-1,
                norm="forward",
            )
        ).view(batch_sz, chan_cnt, patch_cnt, self._patch_size // 2 + 1)
        patch_emb = patch_emb + self.patch_embedding["spectral_proj"](freq_mag)
        pos_offset = self.patch_embedding["positional_encoding"](
            patch_emb.permute(0, 3, 1, 2).contiguous()
        ).permute(0, 2, 3, 1)
        hidden_tokens = patch_emb + pos_offset

        half_dim = self._d_model // 2
        for layer_mod in self.encoder["layers"]:  # type: ignore[attr-defined]
            normed_1 = layer_mod["norm1"](hidden_tokens)
            spatial_in = normed_1[:, :, :, :half_dim].permute(0, 2, 1, 3).contiguous().view(
                batch_sz * patch_cnt, chan_cnt, half_dim
            )
            temporal_in = normed_1[:, :, :, half_dim:].contiguous().view(
                batch_sz * chan_cnt, patch_cnt, half_dim
            )
            spatial_out = (
                layer_mod["self_attn_s"](spatial_in, spatial_in, spatial_in, need_weights=False)[0]
                .view(batch_sz, patch_cnt, chan_cnt, half_dim)
                .permute(0, 2, 1, 3)
            )
            temporal_out = layer_mod["self_attn_t"](temporal_in, temporal_in, temporal_in, need_weights=False)[0].view(
                batch_sz, chan_cnt, patch_cnt, half_dim
            )
            sa_concat = torch.cat((spatial_out, temporal_out), dim=3)
            hidden_tokens = hidden_tokens + layer_mod["dropout1"](sa_concat)
            ff_hidden = layer_mod["linear2"](layer_mod["dropout"](F.gelu(layer_mod["linear1"](layer_mod["norm2"](hidden_tokens)))))
            hidden_tokens = hidden_tokens + layer_mod["dropout2"](ff_hidden)

        return hidden_tokens

    def forward(self, signal_tensor: torch.Tensor, return_features: bool = True) -> dict[str, torch.Tensor | None] | torch.Tensor:
        token_features = self.encode_window_features(signal_tensor)
        if return_features:
            return {"features": token_features, "cls_token": None}
        return token_features
