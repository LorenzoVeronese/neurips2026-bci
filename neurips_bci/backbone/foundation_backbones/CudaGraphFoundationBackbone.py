"""Fixed-shape CUDA Graph capture and replay wrapper for frozen foundation backbones."""

from __future__ import annotations

import os
from typing import Callable
import torch

from neurips_bci.backbone.foundation_backbones.FoundationBackbone import FoundationBackbone
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class CudaGraphFoundationBackbone(FoundationBackbone):
    """Captures and replays fixed-shape CUDA graphs with transparent eager CPU/non-CUDA fallback."""

    def __init__(
        self,
        wrapped_backbone: FoundationBackbone,
        graph_key_prefix: str = "foundation_graph",
        warmup_iterations: int = 2,
    ) -> None:
        if not isinstance(wrapped_backbone, FoundationBackbone):
            raise TypeError(
                f"wrapped_backbone must be FoundationBackbone, got {type(wrapped_backbone).__name__}"
            )
        ParameterTypeValidator.require_non_empty_string(graph_key_prefix, "graph_key_prefix")
        validated_warmup = ParameterTypeValidator.require_integer(
            warmup_iterations, "warmup_iterations", minimum_bound=1
        )
        super().__init__(
            backbone_architecture=f"cuda_graph({wrapped_backbone.backbone_architecture})",
            embedding_dimension=wrapped_backbone.embedding_dimension,
        )
        self._wrapped_backbone = wrapped_backbone
        self._graph_key_prefix = graph_key_prefix
        self._warmup_iterations = validated_warmup
        self._captured_runners: dict[tuple[object, ...], object] = {}

    @property
    def wrapped_backbone(self) -> FoundationBackbone:
        return self._wrapped_backbone

    @property
    def graph_key_prefix(self) -> str:
        return self._graph_key_prefix

    @property
    def warmup_iterations(self) -> int:
        return self._warmup_iterations

    def parameter_count(self) -> int:
        return self._wrapped_backbone.parameter_count()

    def clear_captured_graphs(self) -> None:
        self._captured_runners.clear()

    @staticmethod
    def execute_cached_single_input_graph(
        host_module: torch.nn.Module,
        cache_key: tuple[object, ...],
        eager_callable: Callable[[torch.Tensor], tuple[torch.Tensor | None, ...]],
        input_tensor: torch.Tensor,
        warmup_steps: int = 2,
    ) -> tuple[torch.Tensor | None, ...] | None:
        """Capture once on `host_module` under `cache_key` and replay on CUDA; return `None` for eager fallback."""
        if not (input_tensor.is_cuda and os.environ.get("S17_NO_GRAPH", "0") != "1"):
            return None
        graph_registry: dict[tuple[object, ...], object] = host_module.__dict__.setdefault("_s17_cg", {})
        cached_entry = graph_registry.get(cache_key)
        if cached_entry is None:
            try:
                cuda_dev = input_tensor.device
                static_input = torch.zeros_like(input_tensor)
                static_input.copy_(input_tensor)
                capture_stream = torch.cuda.Stream(device=cuda_dev)
                capture_stream.wait_stream(torch.cuda.current_stream(cuda_dev))
                with torch.cuda.stream(capture_stream), torch.no_grad():
                    for _ in range(max(1, int(warmup_steps))):
                        _ = eager_callable(static_input)
                torch.cuda.current_stream(cuda_dev).wait_stream(capture_stream)
                cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cuda_graph, stream=capture_stream), torch.no_grad():
                    static_outputs = eager_callable(static_input)
                cached_entry = (cuda_graph, static_input, static_outputs)
            except Exception:
                torch.cuda.synchronize(input_tensor.device)
                cached_entry = False
            graph_registry[cache_key] = cached_entry
        if cached_entry is False:
            return None
        cuda_graph, static_input, static_outputs = cached_entry  # type: ignore[misc]
        static_input.copy_(input_tensor)
        cuda_graph.replay()
        return tuple(None if out_item is None else out_item.clone() for out_item in static_outputs)

    def execute_callable(
        self,
        sub_key: tuple[object, ...],
        eager_callable: Callable[..., tuple[torch.Tensor | None, ...]],
        *tensor_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        """Run multi-input `eager_callable` via captured CUDA graph when on CUDA, else eagerly."""
        if not tensor_inputs:
            raise ValueError("execute_callable requires at least one tensor input")
        can_use_cuda_graph = (
            all(inp_t.is_cuda for inp_t in tensor_inputs)
            and len({inp_t.device for inp_t in tensor_inputs}) == 1
            and os.environ.get("S17_NO_GRAPH", "0") != "1"
        )
        if not can_use_cuda_graph:
            with torch.no_grad():
                return eager_callable(*tensor_inputs)

        shape_signature = tuple((tuple(inp_t.shape), inp_t.dtype, str(inp_t.device)) for inp_t in tensor_inputs)
        full_key = (self._graph_key_prefix, sub_key, shape_signature)
        cached_entry = self._captured_runners.get(full_key)
        if cached_entry is None:
            try:
                cuda_dev = tensor_inputs[0].device
                static_inputs = tuple(torch.zeros_like(inp_t) for inp_t in tensor_inputs)
                for st_in, live_in in zip(static_inputs, tensor_inputs):
                    st_in.copy_(live_in)
                capture_stream = torch.cuda.Stream(device=cuda_dev)
                capture_stream.wait_stream(torch.cuda.current_stream(cuda_dev))
                with torch.cuda.stream(capture_stream), torch.no_grad():
                    for _ in range(self._warmup_iterations):
                        _ = eager_callable(*static_inputs)
                torch.cuda.current_stream(cuda_dev).wait_stream(capture_stream)
                cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cuda_graph, stream=capture_stream), torch.no_grad():
                    static_outputs = eager_callable(*static_inputs)
                cached_entry = (cuda_graph, static_inputs, static_outputs)
            except Exception:
                torch.cuda.synchronize(tensor_inputs[0].device)
                cached_entry = False
            self._captured_runners[full_key] = cached_entry

        if cached_entry is False:
            with torch.no_grad():
                return eager_callable(*tensor_inputs)

        cuda_graph, static_inputs, static_outputs = cached_entry  # type: ignore[misc]
        for st_in, live_in in zip(static_inputs, tensor_inputs):
            st_in.copy_(live_in)
        cuda_graph.replay()
        return tuple(None if out_item is None else out_item.clone() for out_item in static_outputs)

    def encode_window_features(
        self, *tensor_inputs: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        def _wrapped_call(*inner_inputs: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
            encoded_out = self._wrapped_backbone.encode_window_features(*inner_inputs)
            if isinstance(encoded_out, tuple):
                return encoded_out
            return (encoded_out,)

        replayed_tuple = self.execute_callable(("encode_window_features",), _wrapped_call, *tensor_inputs)
        if len(replayed_tuple) == 1 and replayed_tuple[0] is not None:
            return replayed_tuple[0]
        return tuple(item_t for item_t in replayed_tuple if item_t is not None)

    def forward(self, *tensor_inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return self.encode_window_features(*tensor_inputs)
