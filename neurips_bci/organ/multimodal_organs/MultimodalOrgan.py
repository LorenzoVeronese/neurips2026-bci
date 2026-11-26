"""Abstract parent class for multimodal test-time-training organs in NeurIPS BCI."""

from __future__ import annotations

from abc import ABC
import numpy as np

from neurips_bci.operator.matrix_operators.CovarianceMatrixOperator import CovarianceMatrixOperator
from neurips_bci.operator.matrix_operators.SubspaceMatrixOperator import SubspaceMatrixOperator
from neurips_bci.operator.signal_operators.SpectralSignalOperator import SpectralSignalOperator
from neurips_bci.organ.Organ import Organ


class MultimodalOrgan(Organ, ABC):
    """Shared subspace and log-probability helpers for cross-modal TTT organs (`ANCHOR-MM`, `S3D`)."""

    def __init__(self, organ_identifier: str) -> None:
        super().__init__(organ_identifier=organ_identifier)

    @staticmethod
    def fit_non_motor_run_subspace(
        windows: np.ndarray,
        session_ids: np.ndarray,
        subspace_dim: int = 6,
        within_shrinkage: float = 0.25,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit a `subspace_dim`-D generalized Fisher subspace across session/run means of 246-D NonMotor features."""
        phi_rows = np.stack(
            [SpectralSignalOperator.extract_non_motor_246_features(win_item) for win_item in windows], axis=0
        )
        mu_phi = phi_rows.mean(axis=0)
        sd_phi = np.maximum(phi_rows.std(axis=0), 1e-6)
        z_phi = (phi_rows - mu_phi[None, :]) / sd_phi[None, :]
        unique_runs = np.unique(session_ids)
        feat_d = z_phi.shape[1]
        if len(unique_runs) < 2:
            _, _, vh_mat = np.linalg.svd(z_phi, full_matrices=False)
            eff_d = min(int(subspace_dim), vh_mat.shape[0])
            u_run = np.zeros((feat_d, int(subspace_dim)), dtype=np.float64)
            u_run[:, :eff_d] = vh_mat[:eff_d].T
            return mu_phi, sd_phi, u_run

        between_mat = np.zeros((feat_d, feat_d), dtype=np.float64)
        within_mat = np.zeros((feat_d, feat_d), dtype=np.float64)
        global_mean = z_phi.mean(axis=0)
        for run_key in unique_runs:
            run_rows = z_phi[session_ids == run_key]
            if len(run_rows) == 0:
                continue
            run_mean = run_rows.mean(axis=0)
            diff_m = run_mean - global_mean
            between_mat += float(len(run_rows)) * np.outer(diff_m, diff_m)
            centered_run = run_rows - run_mean[None, :]
            within_mat += centered_run.T @ centered_run
        between_mat /= max(float(len(z_phi)), 1.0)
        within_mat /= max(float(len(z_phi)), 1.0)
        u_run = SubspaceMatrixOperator.solve_generalized_discriminant(
            between_scatter=between_mat,
            within_scatter=within_mat,
            subspace_dim=int(subspace_dim),
            within_shrinkage=within_shrinkage,
        )
        return mu_phi, sd_phi, u_run

    @staticmethod
    def centered_log_prob_1d(probabilities: np.ndarray) -> np.ndarray:
        return CovarianceMatrixOperator.centered_log_probabilities(np.asarray(probabilities, dtype=np.float64))

    @staticmethod
    def softmax_1d(logits: np.ndarray) -> np.ndarray:
        return CovarianceMatrixOperator.stable_softmax(np.asarray(logits, dtype=np.float64)[None])[0]
