"""Abstract parent for the StreamContainer sub-family."""

from __future__ import annotations

from abc import ABC

from neurips_bci.container.Container import Container


class StreamContainer(Container, ABC):
    """Base class for windowed corpora, single-session causal streams, and CV splits."""

    def __init__(self, container_kind: str) -> None:
        super().__init__(container_kind=container_kind)
