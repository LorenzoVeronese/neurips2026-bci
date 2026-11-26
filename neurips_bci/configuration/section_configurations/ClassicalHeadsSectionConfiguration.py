"""Typed configuration for PseudoChrono, ArtifactGuard, Riemannian heads, ANCHOR-MM, and S3D."""

from __future__ import annotations

from neurips_bci.configuration.section_configurations.SectionConfiguration import SectionConfiguration
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class ClassicalHeadsSectionConfiguration(SectionConfiguration):
    """Defines hyperparameters for ArtifactGuard, PseudoChrono, CSP, Specialists, TSR4, RSNSE, ANCHOR-MM, and S3D."""

    def __init__(
        self,
        artifact_z_amplitude: float = 8.0,
        artifact_z_logdet: float = 8.0,
        artifact_prior_n0: int = 10,
        artifact_clip_z: float = 32.0,
        artifact_clip_target: float = 8.0,
        pseudochrono_dim: int = 6,
        pseudochrono_ea_tau: float = 1.3,
        pseudochrono_bn_tau: float = 1.3,
        session_csp_weight: float = 0.10,
        session_csp_shrinkage_n0: float = 10.0,
        specialist_weight: float = 0.20,
        specialist_kappa: float = 30.0,
        specialist_gap_delta: float = 0.05,
        specialist_prior_n0: float = 20.0,
        enable_tsr4: bool = True,
        tsr4_pc_tau: float = 1.5,
        enable_rsnse: bool = True,
        rsnse_mlp_seeds: int = 5,
        enable_anchor_mm: bool = True,
        anchor_mm_gpu_float64: bool = True,
        anchor_mm_lam_dm: float = 0.52,
        anchor_mm_lam_proto: float = 0.13,
        anchor_mm_c4_direct_lam: float = 0.20,
        enable_s3d: bool = True,
        s3d_subspace_dim: int = 16,
        s3d_ridge_lambda0: float = 36.0,
        weight_s3d: float = 0.12,
    ) -> None:
        super().__init__(configuration_scope="classical_riemannian_and_multimodal_heads")
        self._artifact_z_amplitude = ParameterTypeValidator.require_float(
            artifact_z_amplitude, "artifact_z_amplitude", 1.0
        )
        self._artifact_z_logdet = ParameterTypeValidator.require_float(artifact_z_logdet, "artifact_z_logdet", 1.0)
        self._artifact_prior_n0 = ParameterTypeValidator.require_integer(artifact_prior_n0, "artifact_prior_n0", 1)
        self._artifact_clip_z = ParameterTypeValidator.require_float(artifact_clip_z, "artifact_clip_z", 1.0)
        self._artifact_clip_target = ParameterTypeValidator.require_float(
            artifact_clip_target, "artifact_clip_target", 1.0
        )
        self._pseudochrono_dim = ParameterTypeValidator.require_integer(pseudochrono_dim, "pseudochrono_dim", 1)
        self._pseudochrono_ea_tau = ParameterTypeValidator.require_float(pseudochrono_ea_tau, "pseudochrono_ea_tau", 0.0)
        self._pseudochrono_bn_tau = ParameterTypeValidator.require_float(pseudochrono_bn_tau, "pseudochrono_bn_tau", 0.0)
        self._session_csp_weight = ParameterTypeValidator.require_float(session_csp_weight, "session_csp_weight", 0.0)
        self._session_csp_shrinkage_n0 = ParameterTypeValidator.require_float(
            session_csp_shrinkage_n0, "session_csp_shrinkage_n0", 0.0
        )
        self._specialist_weight = ParameterTypeValidator.require_float(specialist_weight, "specialist_weight", 0.0)
        self._specialist_kappa = ParameterTypeValidator.require_float(specialist_kappa, "specialist_kappa", 0.0)
        self._specialist_gap_delta = ParameterTypeValidator.require_float(
            specialist_gap_delta, "specialist_gap_delta", 0.0, strict_lower=True
        )
        self._specialist_prior_n0 = ParameterTypeValidator.require_float(specialist_prior_n0, "specialist_prior_n0", 0.0)
        self._enable_tsr4 = ParameterTypeValidator.require_boolean(enable_tsr4, "enable_tsr4")
        self._tsr4_pc_tau = ParameterTypeValidator.require_float(tsr4_pc_tau, "tsr4_pc_tau", 0.0)
        self._enable_rsnse = ParameterTypeValidator.require_boolean(enable_rsnse, "enable_rsnse")
        self._rsnse_mlp_seeds = ParameterTypeValidator.require_integer(rsnse_mlp_seeds, "rsnse_mlp_seeds", 1)
        self._enable_anchor_mm = ParameterTypeValidator.require_boolean(enable_anchor_mm, "enable_anchor_mm")
        self._anchor_mm_gpu_float64 = ParameterTypeValidator.require_boolean(
            anchor_mm_gpu_float64, "anchor_mm_gpu_float64"
        )
        self._anchor_mm_lam_dm = ParameterTypeValidator.require_float(anchor_mm_lam_dm, "anchor_mm_lam_dm", 0.0)
        self._anchor_mm_lam_proto = ParameterTypeValidator.require_float(
            anchor_mm_lam_proto, "anchor_mm_lam_proto", 0.0
        )
        self._anchor_mm_c4_direct_lam = ParameterTypeValidator.require_float(
            anchor_mm_c4_direct_lam, "anchor_mm_c4_direct_lam", 0.0
        )
        self._enable_s3d = ParameterTypeValidator.require_boolean(enable_s3d, "enable_s3d")
        self._s3d_subspace_dim = ParameterTypeValidator.require_integer(s3d_subspace_dim, "s3d_subspace_dim", 2)
        self._s3d_ridge_lambda0 = ParameterTypeValidator.require_float(
            s3d_ridge_lambda0, "s3d_ridge_lambda0", 0.0, strict_lower=True
        )
        self._weight_s3d = ParameterTypeValidator.require_float(weight_s3d, "weight_s3d", 0.0, 1.0)

    @property
    def artifact_z_amplitude(self) -> float:
        return self._artifact_z_amplitude

    @property
    def artifact_z_logdet(self) -> float:
        return self._artifact_z_logdet

    @property
    def artifact_prior_n0(self) -> int:
        return self._artifact_prior_n0

    @property
    def artifact_clip_z(self) -> float:
        return self._artifact_clip_z

    @property
    def artifact_clip_target(self) -> float:
        return self._artifact_clip_target

    @property
    def pseudochrono_dim(self) -> int:
        return self._pseudochrono_dim

    @property
    def pseudochrono_ea_tau(self) -> float:
        return self._pseudochrono_ea_tau

    @property
    def pseudochrono_bn_tau(self) -> float:
        return self._pseudochrono_bn_tau

    @property
    def session_csp_weight(self) -> float:
        return self._session_csp_weight

    @property
    def session_csp_shrinkage_n0(self) -> float:
        return self._session_csp_shrinkage_n0

    @property
    def specialist_weight(self) -> float:
        return self._specialist_weight

    @property
    def specialist_kappa(self) -> float:
        return self._specialist_kappa

    @property
    def specialist_gap_delta(self) -> float:
        return self._specialist_gap_delta

    @property
    def specialist_prior_n0(self) -> float:
        return self._specialist_prior_n0

    @property
    def enable_tsr4(self) -> bool:
        return self._enable_tsr4

    @property
    def tsr4_pc_tau(self) -> float:
        return self._tsr4_pc_tau

    @property
    def enable_rsnse(self) -> bool:
        return self._enable_rsnse

    @property
    def rsnse_mlp_seeds(self) -> int:
        return self._rsnse_mlp_seeds

    @property
    def enable_anchor_mm(self) -> bool:
        return self._enable_anchor_mm

    @property
    def anchor_mm_gpu_float64(self) -> bool:
        return self._anchor_mm_gpu_float64

    @property
    def anchor_mm_lam_dm(self) -> float:
        return self._anchor_mm_lam_dm

    @property
    def anchor_mm_lam_proto(self) -> float:
        return self._anchor_mm_lam_proto

    @property
    def anchor_mm_c4_direct_lam(self) -> float:
        return self._anchor_mm_c4_direct_lam

    @property
    def enable_s3d(self) -> bool:
        return self._enable_s3d

    @property
    def s3d_subspace_dim(self) -> int:
        return self._s3d_subspace_dim

    @property
    def s3d_ridge_lambda0(self) -> float:
        return self._s3d_ridge_lambda0

    @property
    def weight_s3d(self) -> float:
        return self._weight_s3d

    def to_dictionary(self) -> dict[str, object]:
        return {
            "artifact_z_amplitude": self._artifact_z_amplitude,
            "artifact_z_logdet": self._artifact_z_logdet,
            "artifact_prior_n0": self._artifact_prior_n0,
            "artifact_clip_z": self._artifact_clip_z,
            "artifact_clip_target": self._artifact_clip_target,
            "pseudochrono_dim": self._pseudochrono_dim,
            "pseudochrono_ea_tau": self._pseudochrono_ea_tau,
            "pseudochrono_bn_tau": self._pseudochrono_bn_tau,
            "session_csp_weight": self._session_csp_weight,
            "session_csp_shrinkage_n0": self._session_csp_shrinkage_n0,
            "specialist_weight": self._specialist_weight,
            "specialist_kappa": self._specialist_kappa,
            "specialist_gap_delta": self._specialist_gap_delta,
            "specialist_prior_n0": self._specialist_prior_n0,
            "enable_tsr4": self._enable_tsr4,
            "tsr4_pc_tau": self._tsr4_pc_tau,
            "enable_rsnse": self._enable_rsnse,
            "rsnse_mlp_seeds": self._rsnse_mlp_seeds,
            "enable_anchor_mm": self._enable_anchor_mm,
            "anchor_mm_gpu_float64": self._anchor_mm_gpu_float64,
            "anchor_mm_lam_dm": self._anchor_mm_lam_dm,
            "anchor_mm_lam_proto": self._anchor_mm_lam_proto,
            "anchor_mm_c4_direct_lam": self._anchor_mm_c4_direct_lam,
            "enable_s3d": self._enable_s3d,
            "s3d_subspace_dim": self._s3d_subspace_dim,
            "s3d_ridge_lambda0": self._s3d_ridge_lambda0,
            "weight_s3d": self._weight_s3d,
        }
