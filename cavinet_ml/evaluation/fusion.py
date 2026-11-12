"""`cavinet-ml fusion` (M-12, FR-12.2, hypothesis H3): symptom-assisted prediction.

A logistic regression on the CT model's logit plus age, sex and the 12 symptoms:

1. Development set: the CT input is each patient's out-of-fold logit (oof_predictions.csv),
   so the fusion never sees a CT score from a model trained on that patient. The fusion's own
   out-of-fold predictions use exactly the CT model's folds (each fold predicted by a fusion
   fitted on the other folds) and are compared with CT-only by DeLong (for information).
2. The final fusion is fitted on all development patients. Its parameters are written to
   <work>/fusion/cavinet_fusion.json (format in `cavinet_ml.fusion`), tied to the model's
   weights by SHA-256.
3. Locked test set, once: the fusion is applied to the frozen model's ensemble logits
   (evaluation/test_predictions.csv) and compared with CT-only by DeLong on the same
   patients. H3 is supported if the combined AUC is higher with p < 0.05. A second run needs
   --force with --reason and is recorded in docs/audit/evaluation_runs.jsonl.

The CT model is not run again: this needs only the toolkit's output files, not the scans.
The report section "Symptom-assisted prediction (H3)" is added to docs/EVALUATION_REPORT.md.
"""

import csv
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from cavinet_ml.dataset.manifest import read_manifest
from cavinet_ml.dataset.split import Splits, load_splits
from cavinet_ml.evaluation.evaluate import (
    EvaluationError,
    EvaluationRefused,
    append_log,
    read_log,
)
from cavinet_ml.evaluation.metrics import roc_auc
from cavinet_ml.evaluation.report_sections import set_section, set_summary_row
from cavinet_ml.evaluation.stats import bootstrap, delong_test, metrics_with_ci
from cavinet_ml.fusion import BINARY, CONTINUOUS, FEATURES, FORMAT, FUSION_FILE, FusionModel
from cavinet_ml.workspace import Workspace

SECTION = "## Symptom-assisted prediction (H3)"
SUMMARY_KEY = "| H3:"
LOG_SPLIT = "fusion-test"


def _value(value: Any) -> float:
    return float("nan") if value is None or value == "" else float(value)


def feature_rows(
    ids: list[str], rows: dict[str, dict[str, Any]], ct_logits: dict[str, float]
) -> np.ndarray:
    """One row per patient in FEATURES order; NaN where a value is missing."""
    table = []
    for case_id in ids:
        row = rows[case_id]
        sex = row["sex"]
        values = {
            "ct_logit": ct_logits[case_id],
            "age": _value(row["age"]),
            "sex_male": 1.0 if sex == "male" else 0.0 if sex == "female" else float("nan"),
            **{c: _value(row[c]) for c in BINARY if c != "sex_male"},
        }
        table.append([values[name] for name in FEATURES])
    return np.array(table, dtype=np.float64)


def _mode(column: np.ndarray) -> float:
    known = column[~np.isnan(column)]
    if known.size == 0:
        return 0.0
    return 1.0 if known.mean() > 0.5 else 0.0


def fit(x: np.ndarray, labels: list[int] | np.ndarray, *, c: float = 1.0) -> dict[str, Any]:
    """Fusion parameters fitted on these patients (imputation and scaling from them only)."""
    from sklearn.linear_model import LogisticRegression

    preprocessing: dict[str, dict[str, float]] = {"median": {}, "mean": {}, "scale": {}, "mode": {}}
    for i, name in enumerate(FEATURES):
        column = x[:, i]
        if name in CONTINUOUS:
            known = column[~np.isnan(column)]
            median = float(np.median(known)) if known.size else 0.0
            filled = np.where(np.isnan(column), median, column)
            scale = float(filled.std())
            preprocessing["median"][name] = median
            preprocessing["mean"][name] = float(filled.mean())
            preprocessing["scale"][name] = scale if scale > 0 else 1.0
        else:
            preprocessing["mode"][name] = _mode(column)
    params: dict[str, Any] = {
        "format": FORMAT,
        "features": list(FEATURES),
        "preprocessing": preprocessing,
        "coefficients": {name: 0.0 for name in FEATURES},
        "intercept": 0.0,
        "version": "unfitted",
    }
    model = FusionModel(params)
    prepared = np.array(
        [model.prepared(dict(zip(FEATURES, row, strict=True))) for row in x], dtype=np.float64
    )
    regression = LogisticRegression(class_weight="balanced", C=c, max_iter=5000)
    regression.fit(prepared, np.asarray(labels))
    params["coefficients"] = {
        name: float(v) for name, v in zip(FEATURES, regression.coef_[0], strict=True)
    }
    params["intercept"] = float(regression.intercept_[0])
    params["regularisation_c"] = c
    params["threshold"] = 0.5
    return params


def predict(params: dict[str, Any], x: np.ndarray) -> list[float]:
    model = FusionModel(params)
    return [model.probability(dict(zip(FEATURES, row, strict=True))) for row in x]


def out_of_fold(
    splits: Splits, rows: dict[str, dict[str, Any]], ct_logits: dict[str, float]
) -> dict[str, float]:
    """Each development patient predicted by a fusion fitted on the other folds."""
    predictions: dict[str, float] = {}
    for fold in range(splits.n_folds):
        train, val = splits.train_ids(fold), splits.val_ids(fold)
        params = fit(feature_rows(train, rows, ct_logits), [rows[c]["label"] for c in train])
        for case_id, p in zip(
            val, predict(params, feature_rows(val, rows, ct_logits)), strict=True
        ):
            predictions[case_id] = p
    return predictions


def _read_csv(path: Path, what: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise EvaluationError(f"{path} not found; {what}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _comparison_statement(c: dict[str, Any], where: str) -> str:
    text = (
        f"Combined AUC {c['auc_a']:.3f} vs CT-only AUC {c['auc_b']:.3f} on the same {c['n']} "
        f"{where} patients: difference {c['difference']:+.3f}"
    )
    if c["p_value"] is None:
        return text + f"; DeLong test not computable: {c['note']}"
    return text + f" (95% CI {c['ci'][0]:+.3f} to {c['ci'][1]:+.3f}), DeLong p = {c['p_value']:.4f}"


def _previous_runs(log_path: Path) -> list[dict[str, Any]]:
    return [
        e
        for e in read_log(log_path)
        if e.get("split") == LOG_SPLIT and e.get("status") == "started"
    ]


def run_fusion(
    ws: Workspace,
    *,
    force: bool = False,
    reason: str | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    out_dir = ws.work / "fusion"
    test_path = ws.evaluation / "test_predictions.csv"
    results_path = ws.evaluation / "test_results.json"
    splits = load_splits(ws.splits)
    rows = {r["case_id"]: r for r in read_manifest(ws.manifest)}
    oof = {
        r["case_id"]: float(r["logit"])
        for r in _read_csv(ws.oof_predictions, "run `cavinet-ml calibrate` first.")
    }
    test_rows = _read_csv(test_path, "evaluate the test split first (`cavinet-ml evaluate`).")
    if not results_path.is_file():
        raise EvaluationError(f"{results_path} not found; evaluate the test split first.")
    bundle = json.loads(results_path.read_text(encoding="utf-8"))["bundle"]

    dev_ids = [c for c in splits.dev if c in oof]
    if len(dev_ids) != len(splits.dev):
        raise EvaluationError(
            f"{len(splits.dev) - len(dev_ids)} development patients have no out-of-fold logit."
        )
    test_ids = list(splits.test)
    by_id = {r["case_id"]: r for r in test_rows}
    missing = [c for c in test_ids if c not in by_id]
    if missing:
        raise EvaluationError(f"test_predictions.csv lacks {len(missing)} test patients.")

    previous = _previous_runs(ws.evaluation_log)
    if previous and not force:
        append_log(
            ws.evaluation_log,
            {"split": LOG_SPLIT, "status": "refused", "previous_runs": len(previous)},
        )
        raise EvaluationRefused(
            f"The fusion was already evaluated on the locked test set ({len(previous)} run(s), "
            f"first on {previous[0]['time']}; see {ws.evaluation_log}). Re-running needs "
            "--force with --reason, and is reported."
        )
    if force and not (reason and reason.strip()):
        raise EvaluationError("--force needs --reason explaining why the test set is used again.")

    # 1. Development set: out-of-fold fusion vs CT-only (information only).
    dev_labels = [rows[c]["label"] for c in dev_ids]
    dev_ct = [oof[c] for c in dev_ids]
    fusion_oof = out_of_fold(splits, rows, oof)
    dev_fused = [fusion_oof[c] for c in dev_ids]
    dev_auc = bootstrap(dev_labels, dev_fused, {"auc": roc_auc})["auc"]
    dev_comparison = delong_test(dev_labels, dev_fused, dev_ct)
    log(
        f"Development set (out-of-fold, {len(dev_ids)} patients): "
        + _comparison_statement(dev_comparison, "development")
        + "."
    )

    # 2. Final fusion on all development patients.
    params = fit(feature_rows(dev_ids, rows, oof), dev_labels)
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    version = f"{bundle.get('model_version') or 'model'}-fusion"
    params.update(
        version=version,
        is_demo=bool(bundle.get("is_demo")),
        created_at=created_at,
        bundle_weights_sha256=bundle["weights_sha256"],
        model_version=bundle.get("model_version"),
        trained_on=(
            f"out-of-fold CT logits and clinical values of {len(dev_ids)} development patients"
        ),
    )

    # 3. Locked test set, once.
    run_number = len(previous) + 1
    append_log(
        ws.evaluation_log,
        {
            "split": LOG_SPLIT,
            "status": "started",
            "run": run_number,
            "forced": force,
            "reason": reason,
            "weights_sha256": bundle["weights_sha256"],
        },
    )
    test_labels = [int(by_id[c]["label"]) for c in test_ids]
    if any(test_labels[i] != rows[c]["label"] for i, c in enumerate(test_ids)):
        raise EvaluationError("test_predictions.csv and the manifest disagree on labels.")
    test_logits = {c: float(by_id[c]["ct_logit"]) for c in test_ids}
    test_ct = [float(by_id[c]["ct_probability"]) for c in test_ids]
    test_fused = predict(params, feature_rows(test_ids, rows, test_logits))
    fused_metrics = metrics_with_ci(test_labels, test_fused, 0.5)
    ct_metrics = metrics_with_ci(test_labels, test_ct, 0.5)
    comparison = delong_test(test_labels, test_fused, test_ct)
    supported = (
        comparison["p_value"] is not None
        and comparison["difference"] > 0
        and comparison["p_value"] < 0.05
    )
    test = {
        "n": len(test_ids),
        "n_tb": int(sum(test_labels)),
        "combined": fused_metrics,
        "ct_only": ct_metrics,
        "comparison": comparison,
    }
    h3 = {
        "supported": supported,
        "statement": _comparison_statement(comparison, "test")
        + "; supported if the combined model is better with p < 0.05.",
    }
    params["metrics"] = {
        "test_auc": fused_metrics["auc"]["value"],
        "test_auc_ci": fused_metrics["auc"]["ci"],
        "ct_only_test_auc": ct_metrics["auc"]["value"],
        "dev_oof_auc": dev_auc["value"],
        "h3_supported": supported,
    }
    results = {
        "created_at": created_at,
        "run": run_number,
        "forced": force,
        "reason": reason,
        "model_version": bundle.get("model_version"),
        "weights_sha256": bundle["weights_sha256"],
        "development": {
            "n": len(dev_ids),
            "combined_auc": dev_auc["value"],
            "combined_auc_ci": dev_auc["ci"],
            "comparison": dev_comparison,
        },
        "test": test,
        "hypothesis": h3,
        "coefficients": params["coefficients"],
        "intercept": params["intercept"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / FUSION_FILE).write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    (out_dir / "fusion_results.json").write_text(
        json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8"
    )
    with (out_dir / "fusion_oof.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "fold", "label", "probability"])
        for case_id, label in zip(dev_ids, dev_labels, strict=True):
            writer.writerow([case_id, splits.fold_of(case_id), label, f"{fusion_oof[case_id]:.6f}"])
    with (out_dir / "fusion_test_predictions.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.writer(h)
        writer.writerow(["case_id", "label", "ct_probability", "combined_probability"])
        for i, case_id in enumerate(test_ids):
            writer.writerow([case_id, test_labels[i], f"{test_ct[i]:.6f}", f"{test_fused[i]:.6f}"])
    update_report(ws.report, results)
    append_log(
        ws.evaluation_log,
        {
            "split": LOG_SPLIT,
            "status": "completed",
            "run": run_number,
            "auc": fused_metrics["auc"]["value"],
            "auc_ci": fused_metrics["auc"]["ci"],
            "n": len(test_ids),
            "h3_supported": supported,
            "weights_sha256": bundle["weights_sha256"],
        },
    )
    log(
        f"Locked test set: {h3['statement']} H3 {'supported' if supported else 'NOT supported'}. "
        f"Fusion parameters: {out_dir / FUSION_FILE}. Report: {ws.report}."
    )
    return results


# --- report -----------------------------------------------------------------------------------

_NAMES = {
    "ct_logit": "CT model logit",
    "age": "Age",
    "sex_male": "Male sex",
}


def _feature_name(name: str) -> str:
    from cavinet_ml.fusion import SYMPTOM_NAMES

    return _NAMES.get(name) or SYMPTOM_NAMES.get(name, name)


def _ci(stat: dict[str, Any], digits: int = 3) -> str:
    return f"{stat['value']:.{digits}f} ({stat['ci'][0]:.{digits}f} to {stat['ci'][1]:.{digits}f})"


def report_section(results: dict[str, Any]) -> list[str]:
    test, dev, h3 = results["test"], results["development"], results["hypothesis"]
    verdict = "**supported**" if h3["supported"] else "**not supported**"
    lines = [
        SECTION,
        "",
        "A logistic regression combines the CT model's logit with age, sex and the 12 "
        "symptoms (FR-12.2). It was fitted on the out-of-fold CT logits of the "
        f"{dev['n']} development patients and evaluated once on the {test['n']} patients of "
        f"the locked test set. H3 (combined better than CT alone): {verdict}.",
        "",
        "| Measure (locked test set, threshold 0.50) | CT only | CT + symptoms |",
        "|---|---|---|",
    ]
    for key, label in (
        ("auc", "AUC (95% CI)"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("brier", "Brier score"),
    ):
        if key in test["ct_only"] and key in test["combined"]:
            lines.append(
                f"| {label} | {_ci(test['ct_only'][key])} | {_ci(test['combined'][key])} |"
            )
    lines += [
        "",
        f"- Test set: {h3['statement']}",
        "- Development set (out-of-fold, for information): "
        + _comparison_statement(dev["comparison"], "development")
        + ".",
        "",
        "Fusion coefficients (inputs standardised for the CT logit and age; positive favours TB):",
        "",
        "| Input | Coefficient |",
        "|---|---|",
        *(
            f"| {_feature_name(name)} | {value:+.3f} |"
            for name, value in results["coefficients"].items()
        ),
        "",
        "Computed with `cavinet-ml fusion` from the frozen model's outputs "
        f"({results['model_version']}); the CT model itself was not evaluated again.",
        "",
    ]
    return lines


def summary_row(results: dict[str, Any]) -> str:
    h3 = results["hypothesis"]
    verdict = "**Supported**" if h3["supported"] else "**Not supported**"
    return f"| H3: CT plus symptoms is better than CT alone | {verdict}. {h3['statement']} |"


def update_report(report: Path, results: dict[str, Any]) -> Path:
    text = report.read_text(encoding="utf-8") if report.is_file() else ""
    text = set_section(text, SECTION, "\n".join(report_section(results)))
    text = set_summary_row(text, SUMMARY_KEY, summary_row(results), after="| H2:")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    return report
