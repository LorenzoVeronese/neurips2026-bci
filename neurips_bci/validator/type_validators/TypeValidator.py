"""Abstract parent for the TypeValidator sub-family."""

from __future__ import annotations

from abc import ABC

from neurips_bci.validator.Validator import Validator


class TypeValidator(Validator, ABC):
    """Base class for type, shape, range, and static safety validators."""

    def __init__(self, validator_scope: str) -> None:
        super().__init__(validator_scope=validator_scope)
