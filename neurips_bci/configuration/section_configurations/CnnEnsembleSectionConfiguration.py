"""Typed configuration for the 160-network Dual-Block CNN40 ensemble."""

from __future__ import annotations

from neurips_bci.configuration.section_configurations.SectionConfiguration import SectionConfiguration
from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator


class CnnEnsembleSectionConfiguration(SectionConfiguration):
    """Defines architectures, training schedules, augmentations, Tent, AG3, and self-training for CNN40."""

    def __init__(
        self,
        seeds_per_family: int = 20,
        block_a_seed_offset: int = 200,
        block_b_seed_offset: int = 1200,
        shallow25_epochs: int = 50,
        fbcnet_epochs: int = 150,
        fbmodnet_distill_epochs: int = 36,
        shallow49_distill_epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        fbmodnet_learning_rate: float = 1.35e-3,
        weight_decay: float = 1e-2,
        dropout_rate: float = 0.5,
        label_smoothing: float = 0.1,
        residual_aug_probability: float = 0.5,
        distill_kl_beta: float = 0.6,
        distill_bn_groups: int = 4,
        swa_tail_fraction: float = 0.3,
        replay_windows_per_class: int = 100,
        tent_learning_rate: float = 5e-4,
        tent_buffer_length: int = 64,
        tent_freeze_window: int = 130,
        self_train_checkpoints: tuple[int, ...] = (66, 126),
        self_train_epochs_shallow: int = 5,
        self_train_epochs_fbcnet: int = 10,
        self_train_epoch_scale: float = 0.5,
        self_train_teacher_gate_threshold: float = 0.58,
        ag3_epochs: int = 10,
        ag3_learning_rate: float = 3e-3,
        ag3_weight_decay: float = 0.06,
        weight_ts_lr: float = 0.05,
        weight_shallow25: float = 0.55,
        weight_fbcnet: float = 0.40,
        readout_weight_fbmodnet: float = 0.15,
        readout_weight_shallow49: float = 0.10,
        shallow_stoff_anchor_weight: float = 0.06,
        deterministic_training: bool = True,
    ) -> None:
        super().__init__(configuration_scope="cnn40_dual_block_ensemble")
        self._seeds_per_family = ParameterTypeValidator.require_integer(seeds_per_family, "seeds_per_family", 1)
        self._block_a_seed_offset = ParameterTypeValidator.require_integer(block_a_seed_offset, "block_a_seed_offset", 0)
        self._block_b_seed_offset = ParameterTypeValidator.require_integer(block_b_seed_offset, "block_b_seed_offset", 0)
        self._shallow25_epochs = ParameterTypeValidator.require_integer(shallow25_epochs, "shallow25_epochs", 1)
        self._fbcnet_epochs = ParameterTypeValidator.require_integer(fbcnet_epochs, "fbcnet_epochs", 1)
        self._fbmodnet_distill_epochs = ParameterTypeValidator.require_integer(
            fbmodnet_distill_epochs, "fbmodnet_distill_epochs", 1
        )
        self._shallow49_distill_epochs = ParameterTypeValidator.require_integer(
            shallow49_distill_epochs, "shallow49_distill_epochs", 1
        )
        self._batch_size = ParameterTypeValidator.require_integer(batch_size, "batch_size", 4)
        self._learning_rate = ParameterTypeValidator.require_float(learning_rate, "learning_rate", 0.0, strict_lower=True)
        self._fbmodnet_learning_rate = ParameterTypeValidator.require_float(
            fbmodnet_learning_rate, "fbmodnet_learning_rate", 0.0, strict_lower=True
        )
        self._weight_decay = ParameterTypeValidator.require_float(weight_decay, "weight_decay", 0.0)
        self._dropout_rate = ParameterTypeValidator.require_float(dropout_rate, "dropout_rate", 0.0, 0.95)
        self._label_smoothing = ParameterTypeValidator.require_float(label_smoothing, "label_smoothing", 0.0, 0.5)
        self._residual_aug_probability = ParameterTypeValidator.require_float(
            residual_aug_probability, "residual_aug_probability", 0.0, 1.0
        )
        self._distill_kl_beta = ParameterTypeValidator.require_float(distill_kl_beta, "distill_kl_beta", 0.0, 1.0)
        self._distill_bn_groups = ParameterTypeValidator.require_integer(distill_bn_groups, "distill_bn_groups", 0)
        self._swa_tail_fraction = ParameterTypeValidator.require_float(swa_tail_fraction, "swa_tail_fraction", 0.0, 1.0)
        self._replay_windows_per_class = ParameterTypeValidator.require_integer(
            replay_windows_per_class, "replay_windows_per_class", 1
        )
        self._tent_learning_rate = ParameterTypeValidator.require_float(tent_learning_rate, "tent_learning_rate", 0.0)
        self._tent_buffer_length = ParameterTypeValidator.require_integer(tent_buffer_length, "tent_buffer_length", 1)
        self._tent_freeze_window = ParameterTypeValidator.require_integer(tent_freeze_window, "tent_freeze_window", 0)
        self._self_train_checkpoints = tuple(int(ckpt) for ckpt in self_train_checkpoints)
        self._self_train_epochs_shallow = ParameterTypeValidator.require_integer(
            self_train_epochs_shallow, "self_train_epochs_shallow", 0
        )
        self._self_train_epochs_fbcnet = ParameterTypeValidator.require_integer(
            self_train_epochs_fbcnet, "self_train_epochs_fbcnet", 0
        )
        self._self_train_epoch_scale = ParameterTypeValidator.require_float(
            self_train_epoch_scale, "self_train_epoch_scale", 0.0, 4.0
        )
        self._self_train_teacher_gate_threshold = ParameterTypeValidator.require_float(
            self_train_teacher_gate_threshold, "self_train_teacher_gate_threshold", 0.0, 1.0
        )
        self._ag3_epochs = ParameterTypeValidator.require_integer(ag3_epochs, "ag3_epochs", 0)
        self._ag3_learning_rate = ParameterTypeValidator.require_float(ag3_learning_rate, "ag3_learning_rate", 0.0)
        self._ag3_weight_decay = ParameterTypeValidator.require_float(ag3_weight_decay, "ag3_weight_decay", 0.0)
        self._weight_ts_lr = ParameterTypeValidator.require_float(weight_ts_lr, "weight_ts_lr", 0.0, 1.0)
        self._weight_shallow25 = ParameterTypeValidator.require_float(weight_shallow25, "weight_shallow25", 0.0, 1.0)
        self._weight_fbcnet = ParameterTypeValidator.require_float(weight_fbcnet, "weight_fbcnet", 0.0, 1.0)
        self._readout_weight_fbmodnet = ParameterTypeValidator.require_float(
            readout_weight_fbmodnet, "readout_weight_fbmodnet", 0.0, 1.0
        )
        self._readout_weight_shallow49 = ParameterTypeValidator.require_float(
            readout_weight_shallow49, "readout_weight_shallow49", 0.0, 1.0
        )
        self._shallow_stoff_anchor_weight = ParameterTypeValidator.require_float(
            shallow_stoff_anchor_weight, "shallow_stoff_anchor_weight", 0.0, 1.0
        )
        self._deterministic_training = ParameterTypeValidator.require_boolean(
            deterministic_training, "deterministic_training"
        )

    @property
    def seeds_per_family(self) -> int:
        return self._seeds_per_family

    @property
    def block_a_seed_offset(self) -> int:
        return self._block_a_seed_offset

    @property
    def block_b_seed_offset(self) -> int:
        return self._block_b_seed_offset

    @property
    def shallow25_epochs(self) -> int:
        return self._shallow25_epochs

    @property
    def fbcnet_epochs(self) -> int:
        return self._fbcnet_epochs

    @property
    def fbmodnet_distill_epochs(self) -> int:
        return self._fbmodnet_distill_epochs

    @property
    def shallow49_distill_epochs(self) -> int:
        return self._shallow49_distill_epochs

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def fbmodnet_learning_rate(self) -> float:
        return self._fbmodnet_learning_rate

    @property
    def weight_decay(self) -> float:
        return self._weight_decay

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate

    @property
    def label_smoothing(self) -> float:
        return self._label_smoothing

    @property
    def residual_aug_probability(self) -> float:
        return self._residual_aug_probability

    @property
    def distill_kl_beta(self) -> float:
        return self._distill_kl_beta

    @property
    def distill_bn_groups(self) -> int:
        return self._distill_bn_groups

    @property
    def swa_tail_fraction(self) -> float:
        return self._swa_tail_fraction

    @property
    def replay_windows_per_class(self) -> int:
        return self._replay_windows_per_class

    @property
    def tent_learning_rate(self) -> float:
        return self._tent_learning_rate

    @property
    def tent_buffer_length(self) -> int:
        return self._tent_buffer_length

    @property
    def tent_freeze_window(self) -> int:
        return self._tent_freeze_window

    @property
    def self_train_checkpoints(self) -> tuple[int, ...]:
        return self._self_train_checkpoints

    @property
    def self_train_epochs_shallow(self) -> int:
        return self._self_train_epochs_shallow

    @property
    def self_train_epochs_fbcnet(self) -> int:
        return self._self_train_epochs_fbcnet

    @property
    def self_train_epoch_scale(self) -> float:
        return self._self_train_epoch_scale

    @property
    def self_train_teacher_gate_threshold(self) -> float:
        return self._self_train_teacher_gate_threshold

    @property
    def ag3_epochs(self) -> int:
        return self._ag3_epochs

    @property
    def ag3_learning_rate(self) -> float:
        return self._ag3_learning_rate

    @property
    def ag3_weight_decay(self) -> float:
        return self._ag3_weight_decay

    @property
    def weight_ts_lr(self) -> float:
        return self._weight_ts_lr

    @property
    def weight_shallow25(self) -> float:
        return self._weight_shallow25

    @property
    def weight_fbcnet(self) -> float:
        return self._weight_fbcnet

    @property
    def readout_weight_fbmodnet(self) -> float:
        return self._readout_weight_fbmodnet

    @property
    def readout_weight_shallow49(self) -> float:
        return self._readout_weight_shallow49

    @property
    def shallow_stoff_anchor_weight(self) -> float:
        return self._shallow_stoff_anchor_weight

    @property
    def deterministic_training(self) -> bool:
        return self._deterministic_training

    def to_dictionary(self) -> dict[str, object]:
        return {
            "seeds_per_family": self._seeds_per_family,
            "block_a_seed_offset": self._block_a_seed_offset,
            "block_b_seed_offset": self._block_b_seed_offset,
            "shallow25_epochs": self._shallow25_epochs,
            "fbcnet_epochs": self._fbcnet_epochs,
            "fbmodnet_distill_epochs": self._fbmodnet_distill_epochs,
            "shallow49_distill_epochs": self._shallow49_distill_epochs,
            "batch_size": self._batch_size,
            "learning_rate": self._learning_rate,
            "fbmodnet_learning_rate": self._fbmodnet_learning_rate,
            "weight_decay": self._weight_decay,
            "dropout_rate": self._dropout_rate,
            "label_smoothing": self._label_smoothing,
            "residual_aug_probability": self._residual_aug_probability,
            "distill_kl_beta": self._distill_kl_beta,
            "distill_bn_groups": self._distill_bn_groups,
            "swa_tail_fraction": self._swa_tail_fraction,
            "replay_windows_per_class": self._replay_windows_per_class,
            "tent_learning_rate": self._tent_learning_rate,
            "tent_buffer_length": self._tent_buffer_length,
            "tent_freeze_window": self._tent_freeze_window,
            "self_train_checkpoints": list(self._self_train_checkpoints),
            "self_train_epochs_shallow": self._self_train_epochs_shallow,
            "self_train_epochs_fbcnet": self._self_train_epochs_fbcnet,
            "self_train_epoch_scale": self._self_train_epoch_scale,
            "self_train_teacher_gate_threshold": self._self_train_teacher_gate_threshold,
            "ag3_epochs": self._ag3_epochs,
            "ag3_learning_rate": self._ag3_learning_rate,
            "ag3_weight_decay": self._ag3_weight_decay,
            "weight_ts_lr": self._weight_ts_lr,
            "weight_shallow25": self._weight_shallow25,
            "weight_fbcnet": self._weight_fbcnet,
            "readout_weight_fbmodnet": self._readout_weight_fbmodnet,
            "readout_weight_shallow49": self._readout_weight_shallow49,
            "shallow_stoff_anchor_weight": self._shallow_stoff_anchor_weight,
            "deterministic_training": self._deterministic_training,
        }
