"""Section 11.4 configuration, augmentation and the section 11.2 MedicalNet start."""

import math
from pathlib import Path

import pytest
import torch

from cavinet_ml.config import DEFAULT_PREPROCESSING
from cavinet_ml.model.network import build_model
from cavinet_ml.training.augment import (
    AugmentationParameters,
    air_value,
    augment_batch,
    sample_parameters,
)
from cavinet_ml.training.config import (
    AugmentationConfig,
    ConfigError,
    TrainingConfig,
    config_from_dict,
    load_config,
    with_overrides,
)
from cavinet_ml.training.pretrained import initialise, load_backbone, read_state_dict

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_train_yaml_states_every_section_11_4_value():
    config = load_config(CONFIGS / "train.yaml")
    assert config == TrainingConfig()  # the YAML and the defaults agree
    assert config.seed == 42
    assert (config.model.architecture, config.model.dropout, config.model.pretrained) == (
        "monai_resnet18",
        0.3,
        "medicalnet",
    )
    assert (config.loss.name, config.loss.pos_weight) == ("bce_with_logits", "auto")
    assert (
        config.optimizer.name,
        config.optimizer.learning_rate,
        config.optimizer.weight_decay,
    ) == (
        "adamw",
        1e-4,
        1e-4,
    )
    assert (config.schedule.name, config.schedule.warmup_epochs) == ("cosine", 3)
    assert (config.epochs.max, config.epochs.early_stopping_metric, config.epochs.patience) == (
        60,
        "val_auc",
        12,
    )
    assert (config.batch.size, config.batch.effective_size, config.batch.amp) == (4, 16, True)
    assert config.accumulation_steps(4) == 4 and config.accumulation_steps(3) == 6
    aug = config.augmentation
    assert (aug.flip_left_right_p, aug.rotate_degrees, aug.scale, aug.translate_voxels) == (
        0.5,
        10.0,
        (0.9, 1.1),
        8.0,
    )
    assert (aug.intensity_scale, aug.gaussian_noise_std) == (0.10, 0.01)
    assert config.logging.tensorboard and config.logging.csv and config.logging.curves_png
    architecture = config.architecture()
    assert architecture["dropout"] == 0.3 and architecture["layers"] == [2, 2, 2, 2]


def test_ci_yaml_changes_only_what_a_cpu_run_needs():
    ci = load_config(CONFIGS / "ci.yaml")
    full = TrainingConfig()
    assert ci.epochs.max == 1 and ci.batch.size == 2 and ci.data.num_workers == 0
    assert ci.model.pretrained == "none"
    assert ci.augmentation == full.augmentation and ci.optimizer == full.optimizer
    assert ci.architecture() == full.architecture()


@pytest.mark.parametrize(
    "values, message",
    [
        ({"optimiser": {}}, "unknown section"),
        ({"batch": {"sizes": 4}}, "unknown setting"),
        ({"batch": {"size": "four"}}, "whole number"),
        ({"batch": {"amp": "yes"}}, "true or false"),
        ({"batch": {"size": 8, "effective_size": 4}}, "effective_size"),
        ({"optimizer": {"learning_rate": -1}}, "learning_rate must be positive"),
        ({"loss": {"pos_weight": "big"}}, "auto or a positive number"),
        ({"model": {"architecture": "resnet50"}}, "monai_resnet18"),
        ({"augmentation": {"scale": [1.1]}}, r"\[low, high\]"),
        ({"augmentation": {"scale": [1.1, 0.9]}}, "low ≤ high"),
        ({"seed": "x"}, "seed"),
        ({"epochs": []}, "mapping"),
    ],
)
def test_invalid_configuration_is_refused(values, message):
    with pytest.raises(ConfigError, match=message):
        config_from_dict(values)


def test_missing_or_broken_yaml_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "none.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("batch: [unclosed")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(bad)


def test_overrides_and_hash():
    config = TrainingConfig()
    changed = with_overrides(config, epochs=5, batch_size=2, num_workers=0, pretrained="none")
    assert (
        changed.epochs.max == 5 and changed.batch.size == 2 and changed.batch.effective_size == 16
    )
    assert changed.data.num_workers == 0 and changed.model.pretrained == "none"
    assert changed.sha256() != config.sha256() and config.sha256() == TrainingConfig().sha256()
    assert with_overrides(config, batch_size=32).batch.effective_size == 32
    assert "learning_rate: 0.0001" in config.to_yaml()


# --- augmentation -------------------------------------------------------------------------------


def _params(
    n=1, flip=False, angles=(0.0, 0.0, 0.0), scale=1.0, shift=(0.0, 0.0, 0.0), intensity=1.0
):
    return AugmentationParameters(
        flip=torch.tensor([flip] * n),
        angles=torch.tensor([angles] * n),
        scale=torch.tensor([scale] * n),
        shift=torch.tensor([shift] * n),
        intensity=torch.tensor([intensity] * n),
    )


NO_NOISE = AugmentationConfig(gaussian_noise_std=0.0)


def _blob(size=16):
    x = torch.zeros(1, 1, size, size, size)
    x[0, 0, 6:10, 6:10, 2:6] = 1.0  # left part of the x axis
    return x


def test_air_value_is_minus_1024_hu_in_the_lung_window():
    assert air_value(DEFAULT_PREPROCESSING) == pytest.approx((-1024 + 1350) / 1500)


def test_identity_parameters_leave_the_volume_unchanged():
    x = torch.rand(2, 1, 12, 12, 12)
    out = augment_batch(x, NO_NOISE, torch.Generator(), 0.2, _params(2))
    assert torch.allclose(out, x, atol=1e-5)


def test_flip_is_left_right():
    x = _blob()
    out = augment_batch(x, NO_NOISE, torch.Generator(), 0.0, _params(flip=True))
    assert torch.allclose(out, x.flip(-1), atol=1e-5)


def test_shift_moves_the_anatomy_and_fills_with_air():
    x = _blob()
    out = augment_batch(x, NO_NOISE, torch.Generator(), 0.25, _params(shift=(4.0, 0.0, 0.0)))
    assert out[0, 0, 7, 7, 2:6].mean() < 0.5 and out[0, 0, 7, 7, 6:10].mean() > 0.9
    assert out[0, 0, 7, 7, 0] == pytest.approx(0.25)  # pulled in from outside: air


def test_scale_and_intensity():
    x = _blob()
    bigger = augment_batch(x, NO_NOISE, torch.Generator(), 0.0, _params(scale=1.5))
    assert bigger.sum() > x.sum() * 1.5
    brighter = augment_batch(x, NO_NOISE, torch.Generator(), 0.0, _params(intensity=1.1))
    assert torch.allclose(brighter, x * 1.1, atol=1e-5)


def test_rotation_turns_the_anatomy():
    x = _blob()
    out = augment_batch(
        x, NO_NOISE, torch.Generator(), 0.0, _params(angles=(0.0, 0.0, math.pi / 2))
    )
    assert not torch.allclose(out, x, atol=1e-3)
    assert out.sum() == pytest.approx(x.sum(), rel=0.05)  # a rotation keeps the volume


def test_sampled_parameters_stay_within_section_11_4_ranges():
    params = sample_parameters(4000, AugmentationConfig(), torch.Generator().manual_seed(0))
    assert abs(params.flip.float().mean().item() - 0.5) < 0.03
    assert params.angles.abs().max() <= math.radians(10) + 1e-6
    assert 0.9 <= params.scale.min() and params.scale.max() <= 1.1
    assert params.shift.abs().max() <= 8.0
    assert 0.9 <= params.intensity.min() and params.intensity.max() <= 1.1


def test_augmentation_is_reproducible_and_noise_has_the_right_size():
    x = torch.full((2, 1, 16, 16, 16), 0.5)
    a = augment_batch(x, AugmentationConfig(), torch.Generator().manual_seed(1), 0.2)
    b = augment_batch(x, AugmentationConfig(), torch.Generator().manual_seed(1), 0.2)
    assert torch.equal(a, b)
    noisy = augment_batch(
        x, AugmentationConfig(gaussian_noise_std=0.01), torch.Generator(), 0.5, _params(2)
    )
    assert (noisy - x).std().item() == pytest.approx(0.01, rel=0.1)
    with pytest.raises(ValueError):
        augment_batch(torch.zeros(1, 8, 8, 8), NO_NOISE, torch.Generator(), 0.0)


# --- MedicalNet ---------------------------------------------------------------------------


def _medicalnet_like(model) -> dict:
    """The model's own backbone weights under MedicalNet's key names, plus its segmentation head."""
    state = {f"module.{k}": v.clone() + 1.0 for k, v in model.backbone.state_dict().items()}
    state["module.conv_seg.0.weight"] = torch.zeros(3)
    return state


def test_none_means_training_from_scratch():
    assert (
        initialise(build_model({"block_inplanes": [4, 8, 16, 32]}), "none")
        == "trained from scratch"
    )


def test_medicalnet_weights_load_into_the_backbone():
    model = build_model({"block_inplanes": [4, 8, 16, 32]})
    state = _medicalnet_like(model)
    text = initialise(model, "medicalnet", download=lambda: state, log=lambda m: None)
    assert text.startswith("initialised from MedicalNet ResNet-18")
    assert torch.equal(model.backbone.conv1.weight, state["module.conv1.weight"])


def test_medicalnet_from_a_file(tmp_path):
    model = build_model({"block_inplanes": [4, 8, 16, 32]})
    path = tmp_path / "resnet_18_23dataset.pth"
    torch.save({"state_dict": _medicalnet_like(model)}, path)
    assert len(read_state_dict(path)) > 10
    assert "resnet_18_23dataset.pth" in initialise(model, str(path), log=lambda m: None)


@pytest.mark.parametrize(
    "source, reason",
    [
        ("download fails", "offline"),
        ("wrong shapes", "do not fit the network"),
        ("missing file", "not found"),
    ],
)
def test_unusable_weights_mean_training_from_scratch_with_the_reason(tmp_path, source, reason):
    model = build_model({"block_inplanes": [4, 8, 16, 32]})
    before = model.backbone.conv1.weight.clone()

    def fail():
        raise OSError("offline")

    if source == "download fails":
        text = initialise(model, "medicalnet", download=fail, log=lambda m: None)
    elif source == "wrong shapes":
        other = build_model({"block_inplanes": [8, 16, 32, 64]})
        text = initialise(
            model, "medicalnet", download=lambda: _medicalnet_like(other), log=lambda m: None
        )
    else:
        text = initialise(model, str(tmp_path / "none.pth"), log=lambda m: None)
    assert (
        text.startswith("trained from scratch (MedicalNet weights unavailable:") and reason in text
    )
    assert torch.equal(model.backbone.conv1.weight, before)


def test_load_backbone_counts_tensors():
    model = build_model({"block_inplanes": [4, 8, 16, 32]})
    assert load_backbone(model, _medicalnet_like(model)) == len(model.backbone.state_dict())
