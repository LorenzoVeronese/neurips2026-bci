"""Linear subspace, SVD delta compression, generalized Fisher, and 3D spherical-spline operator."""

from __future__ import annotations

import mne
import numpy as np
from scipy.linalg import eigh

from neurips_bci.operator.matrix_operators.MatrixOperator import MatrixOperator
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class SubspaceMatrixOperator(MatrixOperator):
    """Provides low-rank SVD delta codecs, generalized Fisher discriminant solvers, and montage splines."""

    PROTEUS_41_CHANNELS: tuple[str, ...] = (
        "C2", "CP2", "P2", "Pz", "CPz", "Cz", "C1", "CP1", "P1", "P3",
        "PO7", "P5", "TP7", "CP5", "CP3", "C3", "PO8", "P6", "P4", "TP8",
        "CP6", "CP4", "C6", "C4", "FC6", "FC4", "FC2", "F6", "F4", "F2",
        "AF4", "AF3", "F5", "F3", "F1", "Fz", "FC5", "FC3", "FC1", "FCz",
        "C5",
    )

    def __init__(self, default_svd_rank: int = 64) -> None:
        super().__init__(operator_domain="linear_subspace_and_montage")
        self._default_svd_rank = ParameterTypeValidator.require_integer(
            default_svd_rank, "default_svd_rank", minimum_bound=1
        )

    @property
    def default_svd_rank(self) -> int:
        return self._default_svd_rank

    def describe_operator(self) -> str:
        return f"Linear subspace & spherical-spline operator (default_svd_rank={self._default_svd_rank})"

    @staticmethod
    def compress_weight_delta_svd(
        fine_tuned_tensor: np.ndarray,
        base_fp16_tensor: np.ndarray,
        max_rank: int = 64,
        min_matrix_elements: int = 32768,
    ) -> dict[str, np.ndarray]:
        delta_fp32 = np.asarray(fine_tuned_tensor, dtype=np.float32) - np.asarray(
            base_fp16_tensor, dtype=np.float16
        ).astype(np.float32)
        if delta_fp32.ndim == 2 and delta_fp32.size >= min_matrix_elements and min(delta_fp32.shape) > max_rank:
            left_u, singular_vals, right_vh = np.linalg.svd(delta_fp32, full_matrices=False)
            effective_rank = min(max_rank, len(singular_vals))
            scaled_u16 = (left_u[:, :effective_rank] * singular_vals[:effective_rank][None, :]).astype(np.float16)
            trunc_vh16 = right_vh[:effective_rank, :].astype(np.float16)
            return {"u16": scaled_u16, "vh16": trunc_vh16}
        return {"exact": np.asarray(fine_tuned_tensor, dtype=np.float32)}

    @staticmethod
    def reconstruct_svd_state_tensor(
        base_fp16_tensor: np.ndarray,
        scaled_u16: np.ndarray,
        trunc_vh16: np.ndarray,
    ) -> np.ndarray:
        reconstructed_delta = scaled_u16.astype(np.float32) @ trunc_vh16.astype(np.float32)
        return base_fp16_tensor.astype(np.float32) + reconstructed_delta

    @staticmethod
    def solve_generalized_discriminant(
        between_scatter: np.ndarray,
        within_scatter: np.ndarray,
        subspace_dim: int,
        within_shrinkage: float = 0.25,
    ) -> np.ndarray:
        dim_count = within_scatter.shape[0]
        trace_scale = max(float(np.trace(within_scatter)) / float(dim_count), 1e-8)
        regularized_within = (1.0 - within_shrinkage) * 0.5 * (within_scatter + within_scatter.T) + (
            within_shrinkage * trace_scale * np.eye(dim_count, dtype=np.float64)
        )
        sym_between = 0.5 * (between_scatter + between_scatter.T)
        eig_vals, eig_vecs = eigh(sym_between, regularized_within)
        top_indices = np.argsort(eig_vals)[::-1][:subspace_dim]
        projection_basis = eig_vecs[:, top_indices]
        column_norms = np.linalg.norm(projection_basis, axis=0, keepdims=True)
        return projection_basis / np.maximum(column_norms, 1e-12)

    @staticmethod
    def build_spherical_spline_mapping(
        source_channel_labels: list[str] | tuple[str, ...],
        target_channel_labels: list[str] | tuple[str, ...],
        regularization: float = 1e-5,
    ) -> np.ndarray:
        standard_montage = mne.channels.make_standard_montage("standard_1005")
        position_lookup = {
            ch_label.lower(): pos_vec
            for ch_label, pos_vec in standard_montage.get_positions()["ch_pos"].items()
        }
        source_coords = np.stack([position_lookup[ch_label.lower()] for ch_label in source_channel_labels], axis=0)
        target_coords = np.stack([position_lookup[ch_label.lower()] for ch_label in target_channel_labels], axis=0)
        source_unit = source_coords / np.maximum(np.linalg.norm(source_coords, axis=1, keepdims=True), 1e-12)
        target_unit = target_coords / np.maximum(np.linalg.norm(target_coords, axis=1, keepdims=True), 1e-12)

        def legendre_g(cos_matrix: np.ndarray, terms: int = 50, stiff_m: int = 4) -> np.ndarray:
            clipped_cos = np.clip(cos_matrix, -1.0, 1.0)
            p_prev = np.ones_like(clipped_cos)
            p_curr = clipped_cos.copy()
            accum = ((2.0 * 1.0 + 1.0) / ((1.0 * 2.0) ** stiff_m)) * p_curr
            for deg_n in range(2, terms + 1):
                p_next = ((2 * deg_n - 1) * clipped_cos * p_curr - (deg_n - 1) * p_prev) / float(deg_n)
                accum = accum + ((2.0 * deg_n + 1.0) / ((deg_n * (deg_n + 1.0)) ** stiff_m)) * p_next
                p_prev, p_curr = p_curr, p_next
            return accum / (4.0 * np.pi)

        g_src_src = legendre_g(source_unit @ source_unit.T) + regularization * np.eye(len(source_unit))
        g_tgt_src = legendre_g(target_unit @ source_unit.T)
        n_src = len(source_unit)
        augmented_system = np.zeros((n_src + 1, n_src + 1), dtype=np.float64)
        augmented_system[:n_src, :n_src] = g_src_src
        augmented_system[:n_src, n_src] = 1.0
        augmented_system[n_src, :n_src] = 1.0
        rhs_identity = np.vstack([np.eye(n_src, dtype=np.float64), np.zeros((1, n_src), dtype=np.float64)])
        spline_coeffs = np.linalg.solve(augmented_system, rhs_identity)
        mapping_matrix = g_tgt_src @ spline_coeffs[:n_src, :] + spline_coeffs[n_src : n_src + 1, :]
        source_lower = {ch_label.lower(): idx for idx, ch_label in enumerate(source_channel_labels)}
        for tgt_idx, tgt_label in enumerate(target_channel_labels):
            if tgt_label.lower() in source_lower:
                mapping_matrix[tgt_idx, :] = 0.0
                mapping_matrix[tgt_idx, source_lower[tgt_label.lower()]] = 1.0
        return mapping_matrix
