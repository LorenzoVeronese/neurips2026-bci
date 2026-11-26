"""Typed configuration for the 4 foundation model organs (EEGPT, CodeBrain, CBraMod, MIRepNet)."""

from __future__ import annotations

from neurips_bci.configuration.section_configurations.SectionConfiguration import SectionConfiguration
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class FoundationSectionConfiguration(SectionConfiguration):
    """Defines seed counts, SVD/LoRA ranks, training epochs, and fusion weights for foundation models."""

    def __init__(
        self,
        enable_eegpt_bank: bool = True,
        eegpt_lat52_seed_count: int = 8,
        eegpt_rms56_seed_count: int = 3,
        eegpt_clean4_seed_count: int = 6,
        eegpt_rms4c_seed_count: int = 2,
        eegpt_svd_rank: int = 64,
        eegpt_lat52_epochs: int = 16,
        eegpt_rms56_epochs: int = 22,
        eegpt_backbone_lr: float = 1e-4,
        eegpt_head_lr: float = 5e-4,
        eegpt_layer_decay: float = 0.75,
        eegpt_refit_checkpoints: tuple[int, ...] = (66, 126),
        eegpt_alpha_clean4: float = 0.35,
        eegpt_alpha_rms4c: float = 0.65,
        enable_codebrain: bool = True,
        codebrain_lora_seed_count: int = 2,
        codebrain_lora_rank: int = 8,
        codebrain_lora_epochs: int = 12,
        weight_codebrain: float = 0.15,
        enable_cbramod: bool = True,
        weight_cbramod: float = 0.10,
        enable_mirepnet: bool = True,
        weight_mirepnet: float = 0.10,
        collapsed_svd_components: int = 128,
        readout_logistic_c: float = 0.02,
    ) -> None:
        super().__init__(configuration_scope="foundation_model_zoo")
        self._enable_eegpt_bank = ParameterTypeValidator.require_boolean(enable_eegpt_bank, "enable_eegpt_bank")
        self._eegpt_lat52_seed_count = ParameterTypeValidator.require_integer(
            eegpt_lat52_seed_count, "eegpt_lat52_seed_count", 0
        )
        self._eegpt_rms56_seed_count = ParameterTypeValidator.require_integer(
            eegpt_rms56_seed_count, "eegpt_rms56_seed_count", 0
        )
        self._eegpt_clean4_seed_count = ParameterTypeValidator.require_integer(
            eegpt_clean4_seed_count, "eegpt_clean4_seed_count", 0
        )
        self._eegpt_rms4c_seed_count = ParameterTypeValidator.require_integer(
            eegpt_rms4c_seed_count, "eegpt_rms4c_seed_count", 0
        )
        self._eegpt_svd_rank = ParameterTypeValidator.require_integer(eegpt_svd_rank, "eegpt_svd_rank", 1)
        self._eegpt_lat52_epochs = ParameterTypeValidator.require_integer(eegpt_lat52_epochs, "eegpt_lat52_epochs", 1)
        self._eegpt_rms56_epochs = ParameterTypeValidator.require_integer(eegpt_rms56_epochs, "eegpt_rms56_epochs", 1)
        self._eegpt_backbone_lr = ParameterTypeValidator.require_float(
            eegpt_backbone_lr, "eegpt_backbone_lr", 0.0, strict_lower=True
        )
        self._eegpt_head_lr = ParameterTypeValidator.require_float(
            eegpt_head_lr, "eegpt_head_lr", 0.0, strict_lower=True
        )
        self._eegpt_layer_decay = ParameterTypeValidator.require_float(
            eegpt_layer_decay, "eegpt_layer_decay", 0.0, 1.0, strict_lower=True
        )
        self._eegpt_refit_checkpoints = tuple(int(ckpt) for ckpt in eegpt_refit_checkpoints)
        self._eegpt_alpha_clean4 = ParameterTypeValidator.require_float(
            eegpt_alpha_clean4, "eegpt_alpha_clean4", 0.0, 1.0
        )
        self._eegpt_alpha_rms4c = ParameterTypeValidator.require_float(eegpt_alpha_rms4c, "eegpt_alpha_rms4c", 0.0, 1.0)
        self._enable_codebrain = ParameterTypeValidator.require_boolean(enable_codebrain, "enable_codebrain")
        self._codebrain_lora_seed_count = ParameterTypeValidator.require_integer(
            codebrain_lora_seed_count, "codebrain_lora_seed_count", 0
        )
        self._codebrain_lora_rank = ParameterTypeValidator.require_integer(codebrain_lora_rank, "codebrain_lora_rank", 1)
        self._codebrain_lora_epochs = ParameterTypeValidator.require_integer(
            codebrain_lora_epochs, "codebrain_lora_epochs", 1
        )
        self._weight_codebrain = ParameterTypeValidator.require_float(weight_codebrain, "weight_codebrain", 0.0, 1.0)
        self._enable_cbramod = ParameterTypeValidator.require_boolean(enable_cbramod, "enable_cbramod")
        self._weight_cbramod = ParameterTypeValidator.require_float(weight_cbramod, "weight_cbramod", 0.0, 1.0)
        self._enable_mirepnet = ParameterTypeValidator.require_boolean(enable_mirepnet, "enable_mirepnet")
        self._weight_mirepnet = ParameterTypeValidator.require_float(weight_mirepnet, "weight_mirepnet", 0.0, 1.0)
        self._collapsed_svd_components = ParameterTypeValidator.require_integer(
            collapsed_svd_components, "collapsed_svd_components", 2
        )
        self._readout_logistic_c = ParameterTypeValidator.require_float(
            readout_logistic_c, "readout_logistic_c", 0.0, strict_lower=True
        )

    @property
    def enable_eegpt_bank(self) -> bool:
        return self._enable_eegpt_bank

    @property
    def eegpt_lat52_seed_count(self) -> int:
        return self._eegpt_lat52_seed_count

    @property
    def eegpt_rms56_seed_count(self) -> int:
        return self._eegpt_rms56_seed_count

    @property
    def eegpt_clean4_seed_count(self) -> int:
        return self._eegpt_clean4_seed_count

    @property
    def eegpt_rms4c_seed_count(self) -> int:
        return self._eegpt_rms4c_seed_count

    @property
    def eegpt_svd_rank(self) -> int:
        return self._eegpt_svd_rank

    @property
    def eegpt_lat52_epochs(self) -> int:
        return self._eegpt_lat52_epochs

    @property
    def eegpt_rms56_epochs(self) -> int:
        return self._eegpt_rms56_epochs

    @property
    def eegpt_backbone_lr(self) -> float:
        return self._eegpt_backbone_lr

    @property
    def eegpt_head_lr(self) -> float:
        return self._eegpt_head_lr

    @property
    def eegpt_layer_decay(self) -> float:
        return self._eegpt_layer_decay

    @property
    def eegpt_refit_checkpoints(self) -> tuple[int, ...]:
        return self._eegpt_refit_checkpoints

    @property
    def eegpt_alpha_clean4(self) -> float:
        return self._eegpt_alpha_clean4

    @property
    def eegpt_alpha_rms4c(self) -> float:
        return self._eegpt_alpha_rms4c

    @property
    def enable_codebrain(self) -> bool:
        return self._enable_codebrain

    @property
    def codebrain_lora_seed_count(self) -> int:
        return self._codebrain_lora_seed_count

    @property
    def codebrain_lora_rank(self) -> int:
        return self._codebrain_lora_rank

    @property
    def codebrain_lora_epochs(self) -> int:
        return self._codebrain_lora_epochs

    @property
    def weight_codebrain(self) -> float:
        return self._weight_codebrain

    @property
    def enable_cbramod(self) -> bool:
        return self._enable_cbramod

    @property
    def weight_cbramod(self) -> float:
        return self._weight_cbramod

    @property
    def enable_mirepnet(self) -> bool:
        return self._enable_mirepnet

    @property
    def weight_mirepnet(self) -> float:
        return self._weight_mirepnet

    @property
    def collapsed_svd_components(self) -> int:
        return self._collapsed_svd_components

    @property
    def readout_logistic_c(self) -> float:
        return self._readout_logistic_c

    def to_dictionary(self) -> dict[str, object]:
        return {
            "enable_eegpt_bank": self._enable_eegpt_bank,
            "eegpt_lat52_seed_count": self._eegpt_lat52_seed_count,
            "eegpt_rms56_seed_count": self._eegpt_rms56_seed_count,
            "eegpt_clean4_seed_count": self._eegpt_clean4_seed_count,
            "eegpt_rms4c_seed_count": self._eegpt_rms4c_seed_count,
            "eegpt_svd_rank": self._eegpt_svd_rank,
            "eegpt_lat52_epochs": self._eegpt_lat52_epochs,
            "eegpt_rms56_epochs": self._eegpt_rms56_epochs,
            "eegpt_backbone_lr": self._eegpt_backbone_lr,
            "eegpt_head_lr": self._eegpt_head_lr,
            "eegpt_layer_decay": self._eegpt_layer_decay,
            "eegpt_refit_checkpoints": list(self._eegpt_refit_checkpoints),
            "eegpt_alpha_clean4": self._eegpt_alpha_clean4,
            "eegpt_alpha_rms4c": self._eegpt_alpha_rms4c,
            "enable_codebrain": self._enable_codebrain,
            "codebrain_lora_seed_count": self._codebrain_lora_seed_count,
            "codebrain_lora_rank": self._codebrain_lora_rank,
            "codebrain_lora_epochs": self._codebrain_lora_epochs,
            "weight_codebrain": self._weight_codebrain,
            "enable_cbramod": self._enable_cbramod,
            "weight_cbramod": self._weight_cbramod,
            "enable_mirepnet": self._enable_mirepnet,
            "weight_mirepnet": self._weight_mirepnet,
            "collapsed_svd_components": self._collapsed_svd_components,
            "readout_logistic_c": self._readout_logistic_c,
        }
