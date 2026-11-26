"""Construction-time type, range, shape, and wiring validator."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np

from neurips_bci.validator.type_validators.TypeValidator import TypeValidator


class ParameterTypeValidator(TypeValidator):
    """Checks plain typed constructor arguments and array shapes before computation."""

    def __init__(self, validator_scope: str = "constructor_wiring") -> None:
        super().__init__(validator_scope=validator_scope)

    def verify_contract(self) -> bool:
        return True

    @staticmethod
    def require_integer(candidate_integer: object, parameter_label: str, minimum_bound: int | None = None) -> int:
        if isinstance(candidate_integer, bool) or not isinstance(candidate_integer, int):
            raise TypeError(f"{parameter_label} must be int, got {type(candidate_integer).__name__}")
        if minimum_bound is not None and candidate_integer < minimum_bound:
            raise ValueError(f"{parameter_label} must be >= {minimum_bound}, got {candidate_integer}")
        return candidate_integer

    @staticmethod
    def require_float(
        candidate_scalar: object,
        parameter_label: str,
        minimum_bound: float | None = None,
        maximum_bound: float | None = None,
        strict_lower: bool = False,
    ) -> float:
        if isinstance(candidate_scalar, bool) or not isinstance(candidate_scalar, (int, float)):
            raise TypeError(f"{parameter_label} must be float or int, got {type(candidate_scalar).__name__}")
        converted_float = float(candidate_scalar)
        if not np.isfinite(converted_float):
            raise ValueError(f"{parameter_label} must be finite, got {converted_float}")
        if minimum_bound is not None:
            if strict_lower and converted_float <= minimum_bound:
                raise ValueError(f"{parameter_label} must be > {minimum_bound}, got {converted_float}")
            if not strict_lower and converted_float < minimum_bound:
                raise ValueError(f"{parameter_label} must be >= {minimum_bound}, got {converted_float}")
        if maximum_bound is not None and converted_float > maximum_bound:
            raise ValueError(f"{parameter_label} must be <= {maximum_bound}, got {converted_float}")
        return converted_float

    @staticmethod
    def require_boolean(candidate_flag: object, parameter_label: str) -> bool:
        if not isinstance(candidate_flag, bool):
            raise TypeError(f"{parameter_label} must be bool, got {type(candidate_flag).__name__}")
        return candidate_flag

    @staticmethod
    def require_non_empty_string(candidate_text: object, parameter_label: str) -> str:
        if not isinstance(candidate_text, str) or not candidate_text.strip():
            raise TypeError(f"{parameter_label} must be a non-empty str")
        return candidate_text

    @staticmethod
    def require_choice(candidate_option: str, allowed_options: Sequence[str], parameter_label: str) -> str:
        if not isinstance(candidate_option, str) or candidate_option not in allowed_options:
            raise ValueError(f"{parameter_label} must be one of {tuple(allowed_options)}, got {candidate_option!r}")
        return candidate_option

    @staticmethod
    def require_path(candidate_path: str | Path, parameter_label: str, must_exist: bool = False) -> Path:
        if not isinstance(candidate_path, (str, Path)):
            raise TypeError(f"{parameter_label} must be str or Path, got {type(candidate_path).__name__}")
        resolved_path = Path(candidate_path)
        if must_exist and not resolved_path.exists():
            raise FileNotFoundError(f"{parameter_label} does not exist at {resolved_path}")
        return resolved_path

    @staticmethod
    def require_numpy_array(
        candidate_array: object,
        parameter_label: str,
        expected_ndim: int | None = None,
        expected_shape: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        if not isinstance(candidate_array, np.ndarray):
            raise TypeError(f"{parameter_label} must be numpy.ndarray, got {type(candidate_array).__name__}")
        if expected_ndim is not None and candidate_array.ndim != expected_ndim:
            raise ValueError(
                f"{parameter_label} must have {expected_ndim} dimensions, got shape {candidate_array.shape}"
            )
        if expected_shape is not None and candidate_array.shape != expected_shape:
            raise ValueError(
                f"{parameter_label} must have shape {expected_shape}, got {candidate_array.shape}"
            )
        return candidate_array
