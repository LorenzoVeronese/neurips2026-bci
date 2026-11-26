"""Abstract base class for cross-validation and warmup fold split acquirers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from neurips_bci.acquisition.Acquirer import Acquirer
from neurips_bci.container.stream_containers.ValidationFoldStreamContainer import ValidationFoldStreamContainer
from neurips_bci.container.stream_containers.WindowedCorpusStreamContainer import WindowedCorpusStreamContainer


class SplitAcquirer(Acquirer, ABC):
    """Base contract for deterministic zero-leakage validation fold builders."""

    def __init__(
        self,
        corpus_container: WindowedCorpusStreamContainer,
        acquirer_source: str,
    ) -> None:
        if not isinstance(corpus_container, WindowedCorpusStreamContainer):
            raise TypeError(
                f"corpus_container must be WindowedCorpusStreamContainer, got {type(corpus_container).__name__}"
            )
        super().__init__(acquirer_source=acquirer_source)
        self._corpus_container = corpus_container

    @property
    def corpus_container(self) -> WindowedCorpusStreamContainer:
        return self._corpus_container

    @abstractmethod
    def acquire(self) -> list[ValidationFoldStreamContainer]:
        """Generate ordered cross-validation fold containers without train/test leakage."""
