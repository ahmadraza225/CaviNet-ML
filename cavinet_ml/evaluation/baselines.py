"""`cavinet-ml baseline` and `cavinet-ml shortcut-check` (FR-10.8, sections 4.3, 4.4, 11.7).

- Clinical-only baseline (H2): logistic regression on age, sex and the 12 symptoms.
- Shortcut check: logistic regression on scan metadata only (scanner manufacturer,
  reconstruction kernel, slice thickness). If it separates TB from NTM well, the CT model
  may be learning scanner differences rather than disease. Reported either way.

Both use exactly the CT model's splits: out-of-fold predictions on the development folds
(each fold predicted by a model fitted on the other folds), and for the locked test set a
model fitted on the whole development set. Missing values are imputed from the training
part only; classes are weighted so the 0.5 threshold is meaningful.
"""

import csv
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from cavinet_ml.dataset.manifest import read_manifest
from cavinet_ml.dataset.patient_index import SYMPTOM_COLUMNS
from cavinet_ml.dataset.split import Splits, load_splits
from cavinet_ml.evaluation.metrics import roc_auc
from cavinet_ml.evaluation.stats import bootstrap

CLINICAL_NUMERIC = ("age", "sex_male", *SYMPTOM_COLUMNS)
METADATA_CATEGORICAL = ("manufacturer_group", "kernel")
METADATA_NUMERIC = ("slice_thickness_mm",)
KINDS = {
    "clinical": "Clinical-only baseline (age, sex, 12 symptoms)",
    "metadata": "Shortcut check: metadata-only model (manufacturer, kernel, slice thickness)",
}


def _value(value: Any) -> float:
    return float("nan") if value is None or value == "" else float(value)


def features(rows: list[dict[str, Any]], kind: str):
    """A pandas DataFrame of the model's inputs, one row per patient."""
    import pandas as pd

    if kind == "clinical":
        table = {
            "age": [_value(r["age"]) for r in rows],
            "sex_male": [
                1.0 if r["sex"] == "male" else 0.0 if r["sex"] == "female" else np.nan for r in rows
            ],
            **{c: [_value(r[c]) for r in rows] for c in SYMPTOM_COLUMNS},
        }
    elif kind == "metadata":
        table = {
            "manufacturer_group": [r["manufacturer_group"] or "unknown" for r in rows],
            "kernel": [(r["kernel"] or "unknown").strip().upper() for r in rows],
            "slice_thickness_mm": [_value(r["slice_thickness_mm"]) for r in rows],
        }
    else:
        raise ValueError(f"unknown baseline {kind!r}")
    return pd.DataFrame(table, index=[r["case_id"] for r in rows])


def make_model(kind: str):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if kind == "clinical":
        columns = ColumnTransformer(
            [
                ("age", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["age"]),
                (
                    "binary",
                    SimpleImputer(strategy="most_frequent"),
                    ["sex_male", *SYMPTOM_COLUMNS],
                ),
            ]
        )
    else:
        columns = ColumnTransformer(
            [
                ("categories", OneHotEncoder(handle_unknown="ignore"), list(METADATA_CATEGORICAL)),
                (
                    "thickness",
                    make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                    list(METADATA_NUMERIC),
                ),
            ]
        )
    return Pipeline(
        [
            ("features", columns),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]
    )


def _probability(model, x) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def out_of_fold(
    kind: str, rows_by_id: dict[str, dict[str, Any]], splits: Splits
) -> dict[str, float]:
    """Each development patient predicted by a model fitted on the other folds."""
    predictions: dict[str, float] = {}
    for fold in range(splits.n_folds):
        train = [rows_by_id[c] for c in splits.train_ids(fold)]
        val = [rows_by_id[c] for c in splits.val_ids(fold)]
        model = make_model(kind).fit(features(train, kind), [r["label"] for r in train])
        for case_id, p in zip(
            splits.val_ids(fold), _probability(model, features(val, kind)), strict=True
        ):
            predictions[case_id] = float(p)
    return predictions


def fit_dev_predict_test(
    kind: str, rows_by_id: dict[str, dict[str, Any]], splits: Splits
) -> tuple[dict[str, float], dict[str, Any]]:
    """A model fitted on all development patients, applied to the locked test set."""
    dev = [rows_by_id[c] for c in splits.dev]
    test = [rows_by_id[c] for c in splits.test]
    model = make_model(kind).fit(features(dev, kind), [r["label"] for r in dev])
    names = model.named_steps["features"].get_feature_names_out()
    coefficients = dict(
        zip(
            (str(n).split("__", 1)[-1] for n in names),
            model.named_steps["model"].coef_[0],
            strict=True,
        )
    )
    probabilities = _probability(model, features(test, kind))
    return (
        {c: float(p) for c, p in zip(splits.test, probabilities, strict=True)},
        {"coefficients": {k: round(float(v), 4) for k, v in coefficients.items()}},
    )


def run_baseline(
    kind: str,
    *,
    manifest_path: Path,
    splits_path: Path,
    out_dir: Path,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Cross-validated baseline on the development folds (never the test set)."""
    splits = load_splits(splits_path)
    rows_by_id = {r["case_id"]: r for r in read_manifest(manifest_path)}
    predictions = out_of_fold(kind, rows_by_id, splits)
    ids = splits.dev
    labels = np.array([rows_by_id[c]["label"] for c in ids])
    scores = np.array([predictions[c] for c in ids])
    auc = bootstrap(labels, scores, {"auc": roc_auc})["auc"]
    per_fold = [
        roc_auc([rows_by_id[c]["label"] for c in f], [predictions[c] for c in f])
        for f in splits.folds
    ]
    summary = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "kind": kind,
        "description": KINDS[kind],
        "n": len(ids),
        "auc": auc["value"],
        "auc_ci": auc["ci"],
        "auc_per_fold": per_fold,
    }
    if kind == "metadata":
        summary["interpretation"] = shortcut_interpretation(auc["value"], auc["ci"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{kind}_oof.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "fold", "label", "probability"])
        for case_id in ids:
            writer.writerow(
                [
                    case_id,
                    splits.fold_of(case_id),
                    rows_by_id[case_id]["label"],
                    f"{predictions[case_id]:.6f}",
                ]
            )
    (out_dir / f"{kind}.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(
        f"{KINDS[kind]}: out-of-fold AUC {auc['value']:.4f} (95% CI {auc['ci'][0]:.3f}-"
        f"{auc['ci'][1]:.3f}) on {len(ids)} development patients; per fold "
        f"{', '.join(f'{a:.3f}' for a in per_fold)}. Written to {out_dir / f'{kind}_oof.csv'}."
    )
    if kind == "metadata":
        log(summary["interpretation"])
    return summary


def shortcut_interpretation(auc: float, ci: list[float]) -> str:
    if ci[0] > 0.5 and auc >= 0.65:
        return (
            f"WARNING: scan metadata alone separates TB from NTM (AUC {auc:.3f}, CI lower bound "
            f"{ci[0]:.3f} > 0.50). The CT model may partly learn scanner differences; report this."
        )
    if ci[0] > 0.5:
        return (
            f"Scan metadata carries a weak signal (AUC {auc:.3f}, CI lower bound "
            f"{ci[0]:.3f} > 0.50); "
            "compare it with the CT model and report it."
        )
    return f"Scan metadata alone does not separate TB from NTM (AUC {auc:.3f}, CI includes 0.50)."


def read_oof_predictions(path: Path, column: str = "probability") -> dict[str, tuple[int, float]]:
    """{case_id: (label, score)} from an out-of-fold CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    with path.open(newline="") as handle:
        return {r["case_id"]: (int(r["label"]), float(r[column])) for r in csv.DictReader(handle)}
