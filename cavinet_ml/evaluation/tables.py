"""`cavinet-ml tables`: thesis-ready tables from the toolkit's outputs (Phase 9).

Reads what the earlier steps wrote (manifest, splits, fold runs, calibration, baselines and the
locked test results) and writes one CSV per table plus TABLES.md with all of them, to
docs/tables/ by default. Nothing is recomputed and the test set is not evaluated again.
"""

import csv
import json
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from cavinet_ml.dataset.manifest import read_manifest
from cavinet_ml.dataset.split import load_splits
from cavinet_ml.evaluation.baselines import KINDS
from cavinet_ml.training.calibrate import read_calibration
from cavinet_ml.workspace import Workspace

METRICS = (
    ("auc", "AUC", False),
    ("sensitivity", "Sensitivity (TB)", True),
    ("specificity", "Specificity (NTM)", True),
    ("ppv", "PPV", True),
    ("npv", "NPV", True),
    ("accuracy", "Accuracy", True),
    ("balanced_accuracy", "Balanced accuracy", True),
    ("f1", "F1", False),
    ("brier", "Brier score", False),
    ("ece", "Expected calibration error", False),
)
CLASSES = ((1, "TB"), (0, "NTM"))


class TablesError(FileNotFoundError):
    """A toolkit output the tables need is missing."""


def _num(value: float | None, digits: int = 3) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{100 * value:.1f}%"


def _with_ci(value: float | None, ci: Sequence[float] | None, pct: bool = False) -> str:
    fmt = _pct if pct else _num
    if value is None:
        return "–"
    if not ci:
        return fmt(value)
    return f"{fmt(value)} ({fmt(ci[0])} to {fmt(ci[1])})"


def _read_json(path: Path, step: str) -> dict[str, Any]:
    if not path.is_file():
        raise TablesError(f"{path} not found; run `cavinet-ml {step}` first.")
    return json.loads(path.read_text(encoding="utf-8"))


# --- the tables: (title, header, rows, note) ----------------------------------------------------

Table = tuple[str, list[str], list[list[str]], str]


def dataset_table(ws: Workspace) -> Table:
    splits = load_splits(ws.splits)
    rows = {r["case_id"]: r for r in read_manifest(ws.manifest)}
    groups = {"Development": splits.dev, "Locked test": splits.test}
    groups["All"] = splits.dev + splits.test
    manufacturers = sorted({rows[c]["manufacturer_group"] for c in groups["All"]})
    header = ["Characteristic"] + [f"{g} {name}" for g in groups for _, name in CLASSES]
    columns = [
        [rows[c] for c in ids if rows[c]["label"] == label]
        for ids in groups.values()
        for label, _ in CLASSES
    ]

    def ages(patients):
        values = [p["age"] for p in patients if p["age"] is not None]
        if len(values) < 2:
            return "–"
        return f"{statistics.mean(values):.1f} ({statistics.stdev(values):.1f})"

    def share(patients, test) -> str:
        n = sum(1 for p in patients if test(p))
        return f"{n} ({100 * n / len(patients):.0f}%)" if patients else "0"

    body = [
        ["Patients"] + [str(len(c)) for c in columns],
        ["Age, mean (SD), years"] + [ages(c) for c in columns],
        ["Female"] + [share(c, lambda p: p["sex"] == "female") for c in columns],
    ]
    body += [
        [f"Scanner: {name}"]
        + [share(c, lambda p, m=name: p["manufacturer_group"] == m) for c in columns]
        for name in manufacturers
    ]
    note = (
        f"{len(splits.excluded)} patients were excluded before splitting "
        f"({', '.join(sorted(splits.excluded))}). Split seed {splits.seed}, stratified by "
        f"{', '.join(splits.raw.get('stratify_by', []))}."
    )
    return "Table 1. Patients by split and class", header, body, note


def folds_table(ws: Workspace) -> Table:
    calibration = read_calibration(ws.calibration)
    oof_auc = {f["fold"]: f["auc"] for f in calibration["per_fold"]}
    header = [
        "Fold",
        "Training patients",
        "Validation patients",
        "Best epoch",
        "Epochs run",
        "Best validation AUC",
        "Out-of-fold AUC (best checkpoint)",
    ]
    body = []
    for fold in sorted(oof_auc):
        run = _read_json(ws.runs / f"fold_{fold}" / "run.json", f"train --fold {fold}")
        body.append(
            [
                str(fold),
                str(run["train_patients"]["n"]),
                str(run["val_patients"]["n"]),
                str(run["best_epoch"]),
                str(run["epochs_run"]),
                _num(run["best_val_auc"]),
                _num(oof_auc[fold]),
            ]
        )
    return "Table 2. Training per fold", header, body, "Early stopping on the validation AUC."


def cross_validation_table(ws: Workspace) -> Table:
    calibration = read_calibration(ws.calibration)
    oof = calibration["oof"]
    header = ["Measure", "Before calibration", "After calibration"]
    body = [
        ["Pooled out-of-fold AUC (95% CI)", _with_ci(oof["auc"], oof["auc_ci"]), "same"],
        ["Brier score", _num(oof["brier_before"]), _num(oof["brier"])],
        ["Expected calibration error", _num(oof["ece_before"]), _num(oof["ece"])],
        ["Temperature", "1 (none)", _num(calibration["temperature"], 4)],
    ]
    note = (
        f"{calibration['n']} development patients, each predicted by the fold model that did "
        "not see it."
    )
    return "Table 3. Cross-validation and calibration", header, body, note


def test_metrics_table(test: dict[str, Any]) -> Table:
    ct = test["ct"]
    youden = ct["youden_threshold"]
    header = ["Metric", "At threshold 0.50 (95% CI)", f"At Youden threshold {youden:.3f} (95% CI)"]
    body = [
        [
            label,
            _with_ci(ct["metrics"][key]["value"], ct["metrics"][key]["ci"], pct),
            _with_ci(ct["at_youden"][key]["value"], ct["at_youden"][key]["ci"], pct),
        ]
        for key, label, pct in METRICS
    ]
    c = ct["confusion"]
    note = (
        f"Locked test set, evaluated once: {test['n']} patients ({test['n_tb']} TB, "
        f"{test['n_ntm']} NTM). At 0.50: {c['tp']} TB and {c['tn']} NTM correct, {c['fn']} TB "
        f"called NTM, {c['fp']} NTM called TB. The application uses 0.50; the Youden threshold "
        "was chosen on the development set."
    )
    return "Table 4. Locked test set performance", header, body, note


def baselines_table(ws: Workspace, test: dict[str, Any]) -> Table:
    header = [
        "Model",
        "Development AUC (95% CI)",
        "Test AUC (95% CI)",
        "CT minus model on test (95% CI)",
        "DeLong p",
    ]
    oof = read_calibration(ws.calibration)["oof"]
    ct_test = test["ct"]["metrics"]["auc"]
    body = [
        [
            "CT model (this work)",
            _with_ci(oof["auc"], oof["auc_ci"]),
            _with_ci(ct_test["value"], ct_test["ci"]),
            "–",
            "–",
        ]
    ]
    for kind, description in KINDS.items():
        on_test = test["baselines"].get(kind)
        if on_test is None:
            continue
        dev_path = ws.baselines / f"{kind}.json"
        dev = json.loads(dev_path.read_text(encoding="utf-8")) if dev_path.is_file() else None
        comparison = on_test["comparison_with_ct"]
        p = comparison.get("p_value")
        body.append(
            [
                description,
                _with_ci(dev["auc"], dev["auc_ci"]) if dev else "–",
                _with_ci(on_test["auc"], on_test["auc_ci"]),
                _with_ci(comparison["difference"], comparison.get("ci")),
                "–" if p is None else f"{p:.3f}",
            ]
        )
    note = test["baselines"].get("metadata", {}).get("interpretation", "")
    return "Table 5. CT model against the baselines", header, body, note


def subgroups_table(test: dict[str, Any]) -> Table:
    header = ["Group", "Value", "Patients", "TB", "NTM", "AUC (95% CI)"]
    body = [
        [
            row["group"].capitalize(),
            row["value"],
            str(row["n"]),
            str(row["n_tb"]),
            str(row["n_ntm"]),
            _with_ci(row["auc"], row["auc_ci"])
            if row.get("auc") is not None
            else row.get("note", "–"),
        ]
        for row in test["subgroups"]
    ]
    return "Table 6. Locked test set AUC by subgroup", header, body, ""


def hypotheses_table(test: dict[str, Any]) -> Table:
    header = ["Hypothesis", "Supported", "Evidence"]
    body = [
        [name, "Yes" if item["supported"] else "No", item["statement"]]
        for name, item in sorted(test["hypotheses"].items())
    ]
    return "Table 7. Hypotheses", header, body, ""


# --- writing ------------------------------------------------------------------------------------

FILES = (
    "1_dataset",
    "2_folds",
    "3_cross_validation",
    "4_test_metrics",
    "5_baselines",
    "6_subgroups",
    "7_hypotheses",
)


def _markdown(table: Table) -> list[str]:
    title, header, body, note = table
    escape = lambda cell: cell.replace("|", "\\|")  # noqa: E731
    lines = [f"## {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(escape(c) for c in row) + " |" for row in body]
    return lines + ([""] + [note] if note else []) + [""]


def export_tables(
    ws: Workspace, out: Path | None = None, log: Callable[[str], None] = print
) -> list[Path]:
    out = out or ws.docs / "tables"
    test = _read_json(ws.evaluation / "test_results.json", "evaluate --split test")
    tables = [
        dataset_table(ws),
        folds_table(ws),
        cross_validation_table(ws),
        test_metrics_table(test),
        baselines_table(ws, test),
        subgroups_table(test),
        hypotheses_table(test),
    ]
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (_, header, body, _) in zip(FILES, tables, strict=True):
        path = out / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(body)
        written.append(path)
    synthetic = " (SYNTHETIC REHEARSAL: not real results)" if test["bundle"]["is_demo"] else ""
    lines = [
        f"# CaviNet thesis tables{synthetic}",
        "",
        f"Generated by `cavinet-ml tables` from the toolkit's outputs for "
        f"{test['bundle']['model_name']} ({test['bundle'].get('model_version') or 'no version'}). "
        "Do not edit by hand. Each table is also a CSV file in this folder. Figures are in "
        "docs/figures/.",
        "",
    ]
    for table in tables:
        lines += _markdown(table)
    summary = out / "TABLES.md"
    summary.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    written.append(summary)
    log(f"Wrote {len(written)} files to {out}.")
    return written
