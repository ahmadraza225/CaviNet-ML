"""The training configuration (section 11.4), read from YAML (ml/configs/train.yaml).

The defaults here are the section 11.4 values; the YAML file states them all explicitly.
Unknown keys and invalid values are refused, so a typo cannot silently change the method.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from cavinet_ml.model.network import DEFAULT_ARCHITECTURE


class ConfigError(Exception):
    """The training configuration is invalid; the message names the setting."""


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "monai_resnet18"
    dropout: float = 0.3
    pretrained: str = "medicalnet"  # medicalnet, none, or a file path


@dataclass(frozen=True)
class LossConfig:
    name: str = "bce_with_logits"
    pos_weight: str | float = "auto"  # auto = #NTM / #TB of the training fold


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class ScheduleConfig:
    name: str = "cosine"
    warmup_epochs: int = 3


@dataclass(frozen=True)
class EpochConfig:
    max: int = 60
    early_stopping_metric: str = "val_auc"
    patience: int = 12


@dataclass(frozen=True)
class BatchConfig:
    size: int = 4
    effective_size: int = 16
    amp: bool = True
    auto_reduce_on_oom: bool = True


@dataclass(frozen=True)
class AugmentationConfig:
    flip_left_right_p: float = 0.5
    rotate_degrees: float = 10.0
    scale: tuple[float, float] = (0.9, 1.1)
    translate_voxels: float = 8.0
    intensity_scale: float = 0.10
    gaussian_noise_std: float = 0.01


@dataclass(frozen=True)
class DataConfig:
    num_workers: int = 4


@dataclass(frozen=True)
class LoggingConfig:
    tensorboard: bool = True
    csv: bool = True
    curves_png: bool = True


_SECTIONS = {
    "model": ModelConfig,
    "loss": LossConfig,
    "optimizer": OptimizerConfig,
    "schedule": ScheduleConfig,
    "epochs": EpochConfig,
    "batch": BatchConfig,
    "augmentation": AugmentationConfig,
    "data": DataConfig,
    "logging": LoggingConfig,
}
_ARCHITECTURES = {"monai_resnet18": DEFAULT_ARCHITECTURE}


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    epochs: EpochConfig = field(default_factory=EpochConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["augmentation"]["scale"] = list(self.augmentation.scale)
        return values

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def architecture(self) -> dict[str, Any]:
        return {**_ARCHITECTURES[self.model.architecture], "dropout": self.model.dropout}

    def accumulation_steps(self, batch_size: int) -> int:
        return max(1, math.ceil(self.batch.effective_size / batch_size))

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False)


def _section(name: str, cls: type, values: Any) -> Any:
    if not isinstance(values, dict):
        raise ConfigError(f"{name}: expected a mapping of settings")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ConfigError(f"{name}: unknown setting(s) {', '.join(unknown)}")
    converted = {}
    for key, value in values.items():
        default = getattr(cls(), key)
        where = f"{name}.{key}"
        if key == "pos_weight":
            if value != "auto" and not (isinstance(value, int | float) and value > 0):
                raise ConfigError(f"{where}: expected auto or a positive number")
            converted[key] = value
        elif key == "pretrained":
            converted[key] = str(value)
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ConfigError(f"{where}: expected true or false")
            converted[key] = value
        elif isinstance(default, int):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"{where}: expected a whole number")
            converted[key] = value
        elif isinstance(default, float):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ConfigError(f"{where}: expected a number")
            converted[key] = float(value)
        elif isinstance(default, tuple):
            if not (isinstance(value, list | tuple) and len(value) == 2):
                raise ConfigError(f"{where}: expected [low, high]")
            converted[key] = (float(value[0]), float(value[1]))
        else:
            converted[key] = str(value)
    return replace(cls(), **converted)


def validate(config: TrainingConfig) -> TrainingConfig:
    checks = [
        (config.model.architecture in _ARCHITECTURES, "model.architecture must be monai_resnet18"),
        (0 <= config.model.dropout < 1, "model.dropout must be between 0 and 1"),
        (config.loss.name == "bce_with_logits", "loss.name must be bce_with_logits"),
        (config.optimizer.name == "adamw", "optimizer.name must be adamw"),
        (config.optimizer.learning_rate > 0, "optimizer.learning_rate must be positive"),
        (config.optimizer.weight_decay >= 0, "optimizer.weight_decay must not be negative"),
        (config.schedule.name == "cosine", "schedule.name must be cosine"),
        (config.schedule.warmup_epochs >= 0, "schedule.warmup_epochs must not be negative"),
        (config.epochs.max >= 1, "epochs.max must be at least 1"),
        (
            config.epochs.early_stopping_metric == "val_auc",
            "epochs.early_stopping_metric must be val_auc",
        ),
        (config.epochs.patience >= 1, "epochs.patience must be at least 1"),
        (config.batch.size >= 1, "batch.size must be at least 1"),
        (
            config.batch.effective_size >= config.batch.size,
            "batch.effective_size must be ≥ batch.size",
        ),
        (
            0 <= config.augmentation.flip_left_right_p <= 1,
            "augmentation.flip_left_right_p must be 0-1",
        ),
        (
            config.augmentation.rotate_degrees >= 0,
            "augmentation.rotate_degrees must not be negative",
        ),
        (
            0 < config.augmentation.scale[0] <= config.augmentation.scale[1],
            "augmentation.scale must be [low, high] with 0 < low ≤ high",
        ),
        (
            config.augmentation.translate_voxels >= 0,
            "augmentation.translate_voxels must not be negative",
        ),
        (0 <= config.augmentation.intensity_scale < 1, "augmentation.intensity_scale must be 0-1"),
        (
            config.augmentation.gaussian_noise_std >= 0,
            "augmentation.gaussian_noise_std must not be negative",
        ),
        (config.data.num_workers >= 0, "data.num_workers must not be negative"),
    ]
    for ok, message in checks:
        if not ok:
            raise ConfigError(message)
    return config


def config_from_dict(values: dict[str, Any]) -> TrainingConfig:
    if not isinstance(values, dict):
        raise ConfigError("the configuration must be a mapping")
    unknown = sorted(set(values) - set(_SECTIONS) - {"seed"})
    if unknown:
        raise ConfigError(f"unknown section(s) {', '.join(unknown)}")
    parts: dict[str, Any] = {}
    if "seed" in values:
        if isinstance(values["seed"], bool) or not isinstance(values["seed"], int):
            raise ConfigError("seed: expected a whole number")
        parts["seed"] = values["seed"]
    for name, cls in _SECTIONS.items():
        if name in values:
            parts[name] = _section(name, cls, values[name])
    return validate(TrainingConfig(**parts))


def load_config(path: Path | str) -> TrainingConfig:
    import yaml

    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"{path} not found")
    try:
        values = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ConfigError(f"{path} is not valid YAML: {error}") from error
    return config_from_dict(values or {})


def with_overrides(
    config: TrainingConfig,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    pretrained: str | None = None,
) -> TrainingConfig:
    """Command-line overrides (recorded in the saved run configuration)."""
    if epochs is not None:
        config = replace(config, epochs=replace(config.epochs, max=epochs))
    if batch_size is not None:
        effective = max(config.batch.effective_size, batch_size)
        config = replace(
            config, batch=replace(config.batch, size=batch_size, effective_size=effective)
        )
    if num_workers is not None:
        config = replace(config, data=replace(config.data, num_workers=num_workers))
    if pretrained is not None:
        config = replace(config, model=replace(config.model, pretrained=pretrained))
    return validate(config)
