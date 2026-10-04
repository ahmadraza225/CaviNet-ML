"""Section 11.2: start from MedicalNet ResNet-18 weights (pretrained on 23 3D medical datasets)
when available, otherwise train from scratch. The outcome is recorded in the run and the
model card.

`pretrained` is "medicalnet" (download via MONAI from Hugging Face, TencentMedicalNet), "none",
or the path of a downloaded resnet_18_23dataset.pth. Only the backbone is loaded; the final
layer is new. All backbone weights must match, otherwise training starts from scratch and the
reason is recorded.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from cavinet_ml.model.network import CaviNetResNet

MEDICALNET_SOURCE = "MedicalNet ResNet-18, 23 datasets (TencentMedicalNet/MedicalNet-Resnet18)"


class PretrainedError(Exception):
    """The pretrained weights could not be used; the message says why."""


def _download_medicalnet() -> dict[str, torch.Tensor]:
    from monai.networks.nets.resnet import get_pretrained_resnet_medicalnet

    state = get_pretrained_resnet_medicalnet(18, device="cpu", datasets23=True)
    if not state:
        raise PretrainedError("the MedicalNet download returned no weights")
    return state


def read_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001
        raise PretrainedError(f"{path} cannot be read safely ({type(error).__name__})") from error
    state = loaded.get("state_dict", loaded) if isinstance(loaded, dict) else None
    if not isinstance(state, dict):
        raise PretrainedError(f"{path} does not hold a state dict")
    return state


def load_backbone(model: CaviNetResNet, state: dict[str, Any]) -> int:
    """Copy MedicalNet weights (keys like `module.layer1.0.conv1.weight`) into
    `model.backbone`; returns the number of tensors loaded."""
    target = model.backbone.state_dict()
    renamed = {k.removeprefix("module."): v for k, v in state.items()}
    usable = {k: v for k, v in renamed.items() if k in target}
    missing = [k for k in target if k not in usable]
    wrong = [k for k, v in usable.items() if tuple(v.shape) != tuple(target[k].shape)]
    if missing or wrong:
        detail = f"{len(missing)} missing" + (f", {len(wrong)} with another shape" if wrong else "")
        raise PretrainedError(f"the weights do not fit the network ({detail})")
    model.backbone.load_state_dict({k: v.float() for k, v in usable.items()}, strict=True)
    return len(usable)


def initialise(
    model: CaviNetResNet,
    pretrained: str,
    *,
    download: Callable[[], dict[str, torch.Tensor]] = _download_medicalnet,
    log: Callable[[str], None] = print,
) -> str:
    """Load the requested weights; returns the description stored in the model card."""
    choice = (pretrained or "none").strip()
    if choice.lower() == "none":
        return "trained from scratch"
    try:
        if choice.lower() == "medicalnet":
            state, source = download(), MEDICALNET_SOURCE
        else:
            path = Path(choice).expanduser()
            if not path.is_file():
                raise PretrainedError(f"{path} not found")
            state, source = read_state_dict(path), f"MedicalNet ResNet-18 from {path.name}"
        count = load_backbone(model, state)
    except Exception as error:  # noqa: BLE001 - any failure means training from scratch
        reason = str(error) or type(error).__name__
        log(f"MedicalNet weights unavailable ({reason}); training from scratch.")
        return f"trained from scratch (MedicalNet weights unavailable: {reason})"
    log(f"Initialised the backbone from {source} ({count} tensors).")
    return f"initialised from {source}"
