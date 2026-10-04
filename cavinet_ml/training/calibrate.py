"""`cavinet-ml calibrate` (FR-10.5, section 11.5): temperature scaling on out-of-fold logits.

Every development patient has exactly one out-of-fold logit, from the fold model that did not
train on it. The temperature minimising their negative log-likelihood is fitted with the
Phase 5 code (`cavinet_ml.inference.calibration`). Per-fold and pooled out-of-fold AUC, Brier
score and expected calibration error (before and after) are recorded, and the Youden-optimal
threshold of the development set is stored for reference (section 11.5; the app uses 0.50).
"""

import csv
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from cavinet_ml.dataset.split import load_splits
from cavinet_ml.evaluation.metrics import (
    brier_score,
    expected_calibration_error,
    log_loss,
    roc_auc,
    youden_threshold,
)
from cavinet_ml.evaluation.stats import bootstrap
from cavinet_ml.inference.calibration import calibrated_probability, fit_temperature
from cavinet_ml.training.trainer import TrainingError


def read_oof(runs_dir: Path, n_folds: int) -> list[dict[str, Any]]:
    rows = []
    for fold in range(n_folds):
        path = runs_dir / f"fold_{fold}" / "oof.csv"
        if not path.is_file():
            raise TrainingError(f"{path} not found; train fold {fold} first.")
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "case_id": row["case_id"],
                        "fold": int(row["fold"]),
                        "label": int(row["label"]),
                        "logit": float(row["logit"]),
                    }
                )
    return rows


def oof_predictions_path(calibration_path: Path) -> Path:
    return calibration_path.with_name("oof_predictions.csv")


def calibrate(
    *, runs_dir: Path, splits_path: Path, out: Path, log: Callable[[str], None] = print
) -> dict[str, Any]:
    splits = load_splits(splits_path)
    rows = read_oof(runs_dir, splits.n_folds)
    found = [r["case_id"] for r in rows]
    dev = set(splits.dev)
    duplicated = sorted({c for c in found if found.count(c) > 1})
    missing = sorted(dev - set(found))
    extra = sorted(set(found) - dev)
    if duplicated or missing or extra:
        raise TrainingError(
            "The out-of-fold predictions do not match the development folds "
            f"(missing {missing[:5]}, duplicated {duplicated[:5]}, not in the folds {extra[:5]})."
        )
    labels = np.array([r["label"] for r in rows])
    logits = np.array([r["logit"] for r in rows])
    temperature = fit_temperature(logits, labels)
    raw = np.array([calibrated_probability(z, 1.0) for z in logits])
    calibrated = np.array([calibrated_probability(z, temperature) for z in logits])
    for row, p in zip(rows, calibrated, strict=True):
        row["probability"] = float(p)

    per_fold = []
    for fold in range(splits.n_folds):
        mask = np.array([r["fold"] == fold for r in rows])
        per_fold.append(
            {"fold": fold, "n": int(mask.sum()), "auc": roc_auc(labels[mask], logits[mask])}
        )
    auc = bootstrap(labels, logits, {"auc": roc_auc})["auc"]
    result = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "temperature": float(temperature),
        "n": int(labels.size),
        "n_tb": int(labels.sum()),
        "per_fold": per_fold,
        "oof": {
            "auc": auc["value"],
            "auc_ci": auc["ci"],
            "auc_per_fold": [f["auc"] for f in per_fold],
            "brier_before": brier_score(labels, raw),
            "brier": brier_score(labels, calibrated),
            "ece_before": expected_calibration_error(labels, raw),
            "ece": expected_calibration_error(labels, calibrated),
            "nll_before": log_loss(labels, raw),
            "nll": log_loss(labels, calibrated),
        },
        "youden_threshold": youden_threshold(labels, calibrated),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with oof_predictions_path(out).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["case_id", "fold", "label", "logit", "probability"]
        )
        writer.writeheader()
        writer.writerows(rows)
    fold_aucs = ", ".join(f"{f['auc']:.3f}" for f in per_fold)
    log(
        f"Temperature {temperature:.4f} fitted on {labels.size} out-of-fold predictions. "
        f"Pooled out-of-fold AUC {auc['value']:.4f} "
        f"(95% CI {auc['ci'][0]:.3f}-{auc['ci'][1]:.3f}); "
        f"per fold {fold_aucs}. "
        f"ECE {result['oof']['ece_before']:.3f} → {result['oof']['ece']:.3f}. Written to {out}."
    )
    return result


def read_calibration(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingError(f"{path} not found; run `cavinet-ml calibrate` first.")
    return json.loads(path.read_text(encoding="utf-8"))
