"""Pure-PyTorch 45-channel 6-layer 256-dim MIRepNet Transformer foundation backbone."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from neurips_bci.backbone.foundation_backbones.FoundationBackbone import FoundationBackbone
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class MirepnetFoundationBackbone(FoundationBackbone):
    """Standalone pure-PyTorch 45-ch 6-layer 256-dim MIRepNet encoder compatible with `mirepnet_fp16.npz`."""

    def __init__(
        self,
        channel_count: int = 45,
        embed_dim: int = 256,
        layer_count: int = 6,
        attention_heads: int = 8,
        output_classes: int = 3,
        dropout_probability: float = 0.5,
    ) -> None:
        validated_chans = ParameterTypeValidator.require_integer(channel_count, "channel_count", minimum_bound=1)
        validated_emb = ParameterTypeValidator.require_integer(embed_dim, "embed_dim", minimum_bound=8)
        validated_layers = ParameterTypeValidator.require_integer(layer_count, "layer_count", minimum_bound=1)
        validated_heads = ParameterTypeValidator.require_integer(attention_heads, "attention_heads", minimum_bound=1)
        validated_classes = ParameterTypeValidator.require_integer(output_classes, "output_classes", minimum_bound=2)
        validated_drop = ParameterTypeValidator.require_float(
            dropout_probability, "dropout_probability", minimum_bound=0.0, maximum_bound=0.95
        )
        if validated_emb % validated_heads != 0:
            raise ValueError(f"embed_dim ({validated_emb}) must be divisible by attention_heads ({validated_heads})")

        super().__init__(backbone_architecture="mirepnet_45ch_6L_256d", embedding_dimension=validated_emb)
        self._channel_count = validated_chans
        self._attention_heads = validated_heads
        self._head_dim = validated_emb // validated_heads
        self._attention_scale = float(validated_emb ** -0.5)
        self._dropout_probability = validated_drop

        self.embedding = nn.ModuleDict(
            {
                "conv1": nn.Conv2d(1, 64, kernel_size=(1, 25)),
                "conv2": nn.Conv2d(64, 128, kernel_size=(validated_chans, 1)),
                "bn": nn.BatchNorm2d(128),
                "elu": nn.ELU(),
                "pool": nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)),
                "dropout": nn.Dropout(validated_drop),
                "projection": nn.Conv2d(128, validated_emb, kernel_size=(1, 1)),
            }
        )

        transformer_blocks = nn.ModuleList()
        for _ in range(validated_layers):
            mha_sub = nn.Module()
            mha_sub.add_module("keys", nn.Linear(validated_emb, validated_emb))
            mha_sub.add_module("queries", nn.Linear(validated_emb, validated_emb))
            mha_sub.add_module("values", nn.Linear(validated_emb, validated_emb))
            mha_sub.add_module("projection", nn.Linear(validated_emb, validated_emb))
            ffn_sub = nn.Sequential(
                nn.Linear(validated_emb, 4 * validated_emb),
                nn.GELU(),
                nn.Dropout(validated_drop),
                nn.Linear(4 * validated_emb, validated_emb),
            )
            block_pair = nn.ModuleList(
                [
                    nn.ModuleDict({"fn": nn.ModuleList([nn.LayerNorm(validated_emb), mha_sub, nn.Dropout(validated_drop)])}),
                    nn.ModuleDict({"fn": nn.ModuleList([nn.LayerNorm(validated_emb), ffn_sub, nn.Dropout(validated_drop)])}),
                ]
            )
            transformer_blocks.append(block_pair)
        self.transformer = transformer_blocks
        self.final_layer = nn.Linear(validated_emb, validated_classes)
        self._fused_stem_wb: tuple[torch.Tensor, torch.Tensor] | None = None

    def freeze_fused_stem(self) -> None:
        with torch.no_grad():
            conv1: nn.Conv2d = self.embedding["conv1"]  # type: ignore[assignment]
            conv2: nn.Conv2d = self.embedding["conv2"]  # type: ignore[assignment]
            assert conv1.bias is not None and conv2.bias is not None
            w_spat = conv2.weight.squeeze(-1).transpose(1, 2)
            fused_weight = (w_spat @ conv1.weight.flatten(1)).detach().contiguous()
            fused_bias = (conv2.weight.sum((2, 3)) @ conv1.bias + conv2.bias).detach().contiguous()
            self._fused_stem_wb = (fused_weight, fused_bias)

    @classmethod
    def from_fp16_npz(
        cls, npz_path: str | Path, execution_device: str | torch.device = "cpu"
    ) -> MirepnetFoundationBackbone:
        resolved_npz = ParameterTypeValidator.require_path(npz_path, "npz_path", must_exist=True)
        backbone = cls()
        state_tensors: dict[str, torch.Tensor] = {}
        with np.load(resolved_npz, allow_pickle=False) as archive:
            for key_str in archive.files:
                raw_arr = archive[key_str]
                state_tensors[key_str] = torch.from_numpy(
                    raw_arr.astype(np.float32) if np.issubdtype(raw_arr.dtype, np.floating) else raw_arr.copy()
                )
        backbone.load_state_dict(state_tensors, strict=True)
        backbone = backbone.to(torch.device(execution_device)).eval()
        for param_tensor in backbone.parameters():
            param_tensor.requires_grad_(False)
        backbone.freeze_fused_stem()
        return backbone

    def export_fp16_npz(self, destination_npz_path: str | Path) -> Path:
        resolved_dst = ParameterTypeValidator.require_path(destination_npz_path, "destination_npz_path")
        resolved_dst.parent.mkdir(parents=True, exist_ok=True)
        npz_payload: dict[str, np.ndarray] = {}
        for key_str, param_tensor in self.state_dict().items():
            np_arr = param_tensor.detach().cpu().numpy()
            if "running_" in key_str or "num_batches" in key_str:
                npz_payload[key_str] = np_arr
            else:
                npz_payload[key_str] = np_arr.astype(np.float16)
        np.savez_compressed(resolved_dst, **npz_payload)
        return resolved_dst

    def _forward_embedding(self, signal_tensor: torch.Tensor) -> torch.Tensor:
        if self._fused_stem_wb is not None and not self.training:
            fused_weight, fused_bias = self._fused_stem_wb
        else:
            conv1: nn.Conv2d = self.embedding["conv1"]  # type: ignore[assignment]
            conv2: nn.Conv2d = self.embedding["conv2"]  # type: ignore[assignment]
            assert conv1.bias is not None and conv2.bias is not None
            w_spat = conv2.weight.squeeze(-1).transpose(1, 2)
            fused_weight = w_spat @ conv1.weight.flatten(1)
            fused_bias = conv2.weight.sum((2, 3)) @ conv1.bias + conv2.bias
        hidden = F.conv1d(signal_tensor, fused_weight, fused_bias).unsqueeze(2)
        hidden = self.embedding["bn"](hidden)
        hidden = self.embedding["elu"](hidden)
        hidden = self.embedding["pool"](hidden)
        hidden = self.embedding["dropout"](hidden)
        hidden = self.embedding["projection"](hidden)
        return hidden.flatten(start_dim=2).transpose(1, 2).contiguous()

    def _forward_mha(self, mha_sub: nn.Module, token_seq: torch.Tensor) -> torch.Tensor:
        batch_sz, seq_len, emb_dim = token_seq.shape
        queries = (
            getattr(mha_sub, "queries")(token_seq)
            .view(batch_sz, seq_len, self._attention_heads, self._head_dim)
            .transpose(1, 2)
        )
        keys = (
            getattr(mha_sub, "keys")(token_seq)
            .view(batch_sz, seq_len, self._attention_heads, self._head_dim)
            .transpose(1, 2)
        )
        val_proj = (
            getattr(mha_sub, "values")(token_seq)
            .view(batch_sz, seq_len, self._attention_heads, self._head_dim)
            .transpose(1, 2)
        )
        attn_probs = F.softmax(torch.matmul(queries, keys.transpose(-2, -1)) * self._attention_scale, dim=-1)
        if self.training and self._dropout_probability > 0.0:
            attn_probs = F.dropout(attn_probs, p=self._dropout_probability, training=True)
        attended = torch.matmul(attn_probs, val_proj).transpose(1, 2).contiguous().view(batch_sz, seq_len, emb_dim)
        return getattr(mha_sub, "projection")(attended)

    def forward_all(self, signal_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self._forward_embedding(signal_tensor)
        for block_pair in self.transformer:
            attn_branch = block_pair[0]["fn"]  # type: ignore[index]
            normed_attn = attn_branch[0](tokens)  # type: ignore[index]
            mha_out = self._forward_mha(attn_branch[1], normed_attn)  # type: ignore[index]
            tokens = tokens + attn_branch[2](mha_out)  # type: ignore[index]

            ffn_branch = block_pair[1]["fn"]  # type: ignore[index]
            normed_ffn = ffn_branch[0](tokens)  # type: ignore[index]
            ffn_out = ffn_branch[1](normed_ffn)  # type: ignore[index]
            tokens = tokens + ffn_branch[2](ffn_out)  # type: ignore[index]
        return tokens, tokens.mean(dim=1)

    def encode_window_features(self, *tensor_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(tensor_inputs) != 1:
            raise ValueError("MirepnetFoundationBackbone expects a single (B, 45, 1000) tensor")
        return self.forward_all(tensor_inputs[0])

    def forward(self, signal_tensor: torch.Tensor) -> torch.Tensor:
        _, mean_features = self.forward_all(signal_tensor)
        return self.final_layer(mean_features)
