"""Section 11.6: the model bundle `cavinet_model.pth`.

One file, loadable with torch.load(..., weights_only=True): it holds only tensors and plain
values (dict, list, str, int, float, bool, None). Fold weights are stored as float16.
"""

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

FORMAT_VERSION = 1
LABEL_MAP = {1: "TB", 0: "NTM"}
DEFAULT_THRESHOLD = 0.5
DEFAULT_BANDS = {"high": 0.80, "moderate": 0.65}

REQUIRED_KEYS = (
    "format_version",
    "model_name",
    "created_at",
    "git_commit",
    "is_demo",
    "architecture",
    "fold_state_dicts",
    "temperature",
    "decision_threshold",
    "confidence_bands",
    "preprocessing",
    "label_map",
    "metrics",
    "data_manifest_sha256",
)
_PLAIN = (str, int, float, bool, type(None))


class BundleError(Exception):
    """The file is not a valid CaviNet model bundle."""


def _check_plain(value: Any, where: str) -> None:
    if isinstance(value, _PLAIN):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str | int):
                raise BundleError(f"{where}: keys must be strings or integers")
            _check_plain(item, f"{where}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_plain(item, f"{where}[{index}]")
        return
    raise BundleError(f"{where}: {type(value).__name__} is not a plain value")


def validate(bundle: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_KEYS if key not in bundle]
    if missing:
        raise BundleError(f"missing keys: {', '.join(missing)}")
    if bundle["format_version"] != FORMAT_VERSION:
        raise BundleError(f"unsupported format_version {bundle['format_version']}")
    folds = bundle["fold_state_dicts"]
    if not isinstance(folds, list) or not folds:
        raise BundleError("fold_state_dicts must be a non-empty list")
    for index, state in enumerate(folds):
        if not all(isinstance(t, torch.Tensor) for t in state.values()):
            raise BundleError(f"fold {index}: state dict values must be tensors")
    for key in REQUIRED_KEYS:
        if key != "fold_state_dicts":
            _check_plain(bundle[key], key)
    if not isinstance(bundle["is_demo"], bool):
        raise BundleError("is_demo must be true or false")
    return bundle


def half_state_dict(state: dict[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    """Floating-point tensors as float16 (counters such as num_batches_tracked unchanged)."""
    return OrderedDict(
        (name, t.detach().cpu().half() if t.is_floating_point() else t.detach().cpu().clone())
        for name, t in state.items()
    )


def save_bundle(bundle: dict[str, Any], path: Path | str) -> Path:
    bundle = dict(bundle)
    bundle["fold_state_dicts"] = [half_state_dict(s) for s in bundle["fold_state_dicts"]]
    validate(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(bundle, temporary)
    temporary.replace(path)  # never leave a half-written bundle behind
    return path


def load_bundle(path: Path | str, *, mmap: bool = False) -> dict[str, Any]:
    """Load with weights_only=True (no arbitrary code can run from the file)."""
    try:
        bundle = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=mmap)
    except FileNotFoundError:
        raise
    except Exception as error:  # noqa: BLE001 - pickle, zip and type errors alike
        raise BundleError(f"cannot read the model file: {type(error).__name__}") from error
    if not isinstance(bundle, dict):
        raise BundleError("the model file does not hold a bundle")
    return validate(bundle)


def metadata(bundle: dict[str, Any]) -> dict[str, Any]:
    """Everything except the weights (for the admin model page, FR-09.4)."""
    info = {key: value for key, value in bundle.items() if key != "fold_state_dicts"}
    info["folds"] = len(bundle["fold_state_dicts"])
    buffers = ("running_mean", "running_var", "num_batches_tracked")
    info["parameters_per_fold"] = int(
        sum(
            t.numel()
            for name, t in bundle["fold_state_dicts"][0].items()
            if not name.endswith(buffers)
        )
    )
    return info


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
