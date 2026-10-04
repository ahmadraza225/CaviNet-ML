"""`cavinet-ml export` (FR-10.6, section 11.6): the model bundle, model_card.json and
docs/MODEL_CARD.md.

The bundle holds the best checkpoint of every fold (float16), the temperature, the decision
threshold and bands, the section 11.1 parameters the cache was made with, the label map, the
out-of-fold metrics (the locked-test metrics are added by `evaluate --split test`) and the
manifest hash. It is saved and checked with the Phase 5 code (`save_bundle`, `load_bundle`
with weights_only=True, `Ensemble`). A model built from the synthetic look-alike dataset is
always marked `is_demo`, so it can never pass for a real model.
"""

import hashlib
import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from cavinet_ml import __version__
from cavinet_ml.dataset.cache import cache_config, load_volume
from cavinet_ml.dataset.manifest import file_sha256, read_summary
from cavinet_ml.dataset.split import load_splits
from cavinet_ml.fetch import write_model_card
from cavinet_ml.inference.ensemble import Ensemble
from cavinet_ml.model.bundle import (
    DEFAULT_BANDS,
    DEFAULT_THRESHOLD,
    FORMAT_VERSION,
    LABEL_MAP,
    half_state_dict,
    load_bundle,
    metadata,
    save_bundle,
)
from cavinet_ml.provenance import git_commit
from cavinet_ml.training.calibrate import read_calibration
from cavinet_ml.training.config import config_from_dict
from cavinet_ml.training.model_card import write_model_card_md
from cavinet_ml.training.trainer import TrainingError

REAL_DATASET = (
    "Mycobacterial CT Imaging Dataset (Tianjin Haihe Hospital, 2014-2024), Kaggle "
    "damianhan/dicom-dataset, CC BY 4.0."
)
SYNTHETIC_DATASET = (
    "Synthetic look-alike of the Kaggle dataset (cavinet_ml.dataset.synthetic); no patient data."
)


def json_safe(value: Any) -> Any:
    """Plain JSON values only: NaN and infinity become None (undefined, e.g. NPV when no
    patient is predicted NTM). The application stores the metrics in a PostgreSQL JSON column,
    which refuses NaN."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int | float) or hasattr(value, "item"):
        number = value.item() if hasattr(value, "item") else value
        if isinstance(number, float) and not math.isfinite(number):
            return None
        return number
    return value


def weights_sha256(fold_state_dicts: list[dict[str, torch.Tensor]]) -> str:
    """Hash of the stored (float16) weights; unchanged when only metadata is updated."""
    digest = hashlib.sha256()
    for state in fold_state_dicts:
        for name, tensor in sorted(half_state_dict(state).items()):
            digest.update(name.encode())
            digest.update(tensor.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _fold_runs(runs_dir: Path, n_folds: int) -> list[dict[str, Any]]:
    runs = []
    for fold in range(n_folds):
        run_dir = runs_dir / f"fold_{fold}"
        last, best, info = run_dir / "last.pt", run_dir / "best.pt", run_dir / "run.json"
        if not (last.is_file() and best.is_file() and info.is_file()):
            raise TrainingError(f"Fold {fold} is not trained yet ({run_dir}).")
        state = torch.load(last, map_location="cpu", weights_only=True)
        if not state.get("finished"):
            raise TrainingError(f"Fold {fold} has not finished training; run it again to resume.")
        runs.append(
            {
                "fold": fold,
                "best": torch.load(best, map_location="cpu", weights_only=True),
                "info": json.loads(info.read_text()),
                "config": (run_dir / "config.yaml").read_text(),
            }
        )
    if len({r["info"]["config_sha256"] for r in runs}) != 1:
        raise TrainingError("The folds were trained with different configurations.")
    return runs


def export_bundle(
    *,
    runs_dir: Path,
    splits_path: Path,
    manifest_path: Path,
    cache_dir: Path,
    calibration_path: Path,
    out: Path,
    model_card_md: Path | None,
    model_name: str | None = None,
    model_version: str = "model-v1",
    demo: bool = False,
    log: Callable[[str], None] = print,
) -> Path:
    import yaml

    splits = load_splits(splits_path)
    runs = _fold_runs(runs_dir, splits.n_folds)
    calibration = read_calibration(calibration_path)
    config = config_from_dict(yaml.safe_load(runs[0]["config"]))
    synthetic = bool(read_summary(manifest_path).get("dataset", {}).get("synthetic"))
    is_demo = demo or synthetic
    name = model_name or (
        "CaviNet rehearsal model (synthetic data)" if is_demo else "CaviNet TB vs NTM ensemble"
    )
    states = [r["best"]["model"] for r in runs]
    initialisations = {r["info"]["initialisation"] for r in runs}
    counts = splits.raw.get("counts", {})
    bundle = {
        "format_version": FORMAT_VERSION,
        "model_name": name,
        "model_version": model_version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "is_demo": is_demo,
        "architecture": config.architecture(),
        "initialisation": "; ".join(sorted(initialisations)),
        "fold_state_dicts": states,
        "temperature": float(calibration["temperature"]),
        "decision_threshold": DEFAULT_THRESHOLD,
        "confidence_bands": dict(DEFAULT_BANDS),
        "preprocessing": cache_config(cache_dir).to_dict(),
        "label_map": dict(LABEL_MAP),
        "metrics": json_safe(
            {
                "dataset": SYNTHETIC_DATASET if synthetic else REAL_DATASET,
                "oof": {
                    **calibration["oof"],
                    "n": calibration["n"],
                    "temperature": calibration["temperature"],
                },
                "youden_threshold_dev": calibration["youden_threshold"],
                "locked_test": None,
            }
        ),
        "data_manifest_sha256": file_sha256(manifest_path),
        "splits_sha256": file_sha256(splits_path),
        "weights_sha256": weights_sha256(states),
        "toolkit_version": __version__,
        "training": json_safe(
            {
                "config": config.to_dict(),
                "data": {
                    "patients": counts.get("patients"),
                    "excluded": counts.get("excluded"),
                    "dev": counts.get("dev", {}).get("n"),
                    "test": counts.get("test", {}).get("n"),
                },
                "folds": [
                    {
                        "fold": r["fold"],
                        "best_epoch": r["info"]["best_epoch"],
                        "best_val_auc": r["info"]["best_val_auc"],
                        "epochs_run": r["info"]["epochs_run"],
                        "batch_size": r["info"]["batch_size_used"],
                        "pos_weight": r["info"]["pos_weight"],
                    }
                    for r in runs
                ],
            }
        ),
    }
    save_bundle(bundle, out)
    check = verify_bundle(out, cache_dir, splits.dev[0])
    card = write_model_card(out)
    if model_card_md is not None:
        write_model_card_md(json.loads(card.read_text()), model_card_md)
    log(
        f"Wrote {out} ({out.stat().st_size / 1e6:.0f} MB, {len(states)} folds, "
        f"{'DEMO' if is_demo else 'real'}), {card}"
        + (f" and {model_card_md}" if model_card_md else "")
        + f". Check: it loads with weights_only=True and gives p(TB) = {check:.3f} "
        f"for {splits.dev[0]}."
    )
    return out


def verify_bundle(path: Path, cache_dir: Path, case_id: str) -> float:
    """Load the bundle exactly as the application does and run one cached volume through it."""
    ensemble = Ensemble(load_bundle(path))
    probability = ensemble.predict(
        load_volume(cache_dir, case_id, mmap=False)
    ).decision.probability_tb
    if not 0.0 <= probability <= 1.0:
        raise TrainingError("the exported bundle produced an invalid probability")
    return probability


def update_bundle_metrics(
    path: Path, locked_test: dict[str, Any], model_card_md: Path | None
) -> None:
    """Add the locked-test results to the bundle (weights unchanged), model_card.json and
    MODEL_CARD.md."""
    bundle = load_bundle(path)
    before = bundle.get("weights_sha256") or weights_sha256(bundle["fold_state_dicts"])
    bundle["metrics"] = {**bundle["metrics"], "locked_test": json_safe(locked_test)}
    if weights_sha256(bundle["fold_state_dicts"]) != before:
        raise TrainingError(
            "The model weights changed since export; refusing to update the bundle."
        )
    save_bundle(bundle, path)
    card = write_model_card(path)
    if model_card_md is not None:
        write_model_card_md(json.loads(card.read_text()), model_card_md)


def bundle_metadata(path: Path) -> dict[str, Any]:
    return metadata(load_bundle(path, mmap=True))
