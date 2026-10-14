"""`cavinet-ml evaluate` (FR-10.7, section 11.7) and `cavinet-ml compare` (FR-10.8).

- `--split dev`: the cross-validation results (per-fold and pooled out-of-fold AUC,
  calibration, the baselines on the same folds). Can be run any number of times.
- `--split test`: the locked test set, exactly once, after the model is frozen. Every run,
  including refused ones, is appended to the evaluation log (docs/audit/evaluation_runs.jsonl,
  committed). If the log already holds a test run, the command refuses unless --force is given
  together with --reason. The frozen bundle is used through the application's own code
  (`Ensemble`); its locked-test results are then written into the bundle (weights unchanged),
  model_card.json and docs/MODEL_CARD.md.

Results: docs/EVALUATION_REPORT.md with figures in docs/figures/.
"""

import csv
import getpass
import json
import socket
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from cavinet_ml import __version__
from cavinet_ml.dataset.cache import cache_config, load_volume
from cavinet_ml.dataset.manifest import AGE_BANDS, read_manifest, read_summary
from cavinet_ml.dataset.split import load_splits
from cavinet_ml.evaluation import figures
from cavinet_ml.evaluation.baselines import (
    KINDS,
    fit_dev_predict_test,
    read_oof_predictions,
    shortcut_interpretation,
)
from cavinet_ml.evaluation.metrics import (
    calibration_bins,
    roc_auc,
    roc_curve,
    threshold_metrics,
)
from cavinet_ml.evaluation.report import write_report
from cavinet_ml.evaluation.stats import bootstrap, delong_ci, delong_test, metrics_with_ci
from cavinet_ml.inference.ensemble import Ensemble
from cavinet_ml.model.bundle import file_sha256 as bundle_file_sha256
from cavinet_ml.model.bundle import load_bundle
from cavinet_ml.provenance import git_commit
from cavinet_ml.training.calibrate import read_calibration
from cavinet_ml.training.export import update_bundle_metrics, weights_sha256
from cavinet_ml.workspace import Workspace

TARGET_AUC = 0.75  # section 5.3
SUBGROUP_MIN_PER_CLASS = 2


class EvaluationError(Exception):
    """The evaluation cannot run; the message says why."""


class EvaluationRefused(EvaluationError):
    """A second run on the locked test set without --force."""


# --- the run log (FR-10.7) ------------------------------------------------------------------


def read_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def append_log(path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "user": _user(),
        "host": socket.gethostname(),
        "git_commit": git_commit(),
        "toolkit_version": __version__,
        **entry,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def _user() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return "unknown"


def previous_test_runs(path: Path) -> list[dict[str, Any]]:
    """Earlier test-split runs that got as far as the test set (one "started" entry each,
    whether they completed or failed)."""
    return [e for e in read_log(path) if e.get("split") == "test" and e.get("status") == "started"]


# --- shared pieces ----------------------------------------------------------------------------


def _histories(runs_dir: Path, n_folds: int) -> dict[int, list[dict[str, Any]]]:
    histories = {}
    for fold in range(n_folds):
        path = runs_dir / f"fold_{fold}" / "train_log.csv"
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                histories[fold] = [
                    {
                        k: float(v) if k not in ("epoch", "batch_size") else int(v)
                        for k, v in row.items()
                    }
                    for row in csv.DictReader(handle)
                ]
    return histories


def _baseline_dev(ws: Workspace, kind: str) -> dict[str, Any] | None:
    path = ws.baselines / f"{kind}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _curve(label: str, labels, scores, auc_ci=None) -> dict[str, Any]:
    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        "label": label,
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc(labels, scores),
        "auc_ci": auc_ci,
    }


def development_results(ws: Workspace, log: Callable[[str], None]) -> dict[str, Any]:
    splits = load_splits(ws.splits)
    calibration = read_calibration(ws.calibration)
    oof = read_oof_predictions(ws.oof_predictions, "probability")
    ids = [c for c in splits.dev if c in oof]
    labels = [oof[c][0] for c in ids]
    ct = [oof[c][1] for c in ids]
    results: dict[str, Any] = {
        "calibration": calibration,
        "folds": calibration["per_fold"],
        "baselines": {},
        "comparisons": {},
    }
    curves = [_curve("CT model (out-of-fold)", labels, ct, calibration["oof"]["auc_ci"])]
    for kind in KINDS:
        summary = _baseline_dev(ws, kind)
        if summary is None:
            continue
        predictions = read_oof_predictions(ws.baselines / f"{kind}_oof.csv")
        results["baselines"][kind] = summary
        scores = [predictions[c][1] for c in ids]
        results["comparisons"][kind] = delong_test(labels, ct, scores)
        curves.append(_curve(KINDS[kind].split(":")[0], labels, scores, summary["auc_ci"]))
    histories = _histories(ws.runs, splits.n_folds)
    paths = {}
    if histories:
        paths["training_curves"] = figures.training_curves(
            histories, ws.figures / "training_curves.png"
        )
    paths["roc_dev"] = figures.roc_figure(
        curves, ws.figures / "roc_dev.png", "ROC, development set (out-of-fold)"
    )
    paths["reliability_dev"] = figures.reliability_figure(
        calibration_bins(labels, ct),
        ws.figures / "reliability_dev.png",
        calibration["oof"]["ece"],
        "Reliability, development set (out-of-fold)",
    )
    results["figures"] = {k: str(v) for k, v in paths.items()}
    return results


def _data_summary(ws: Workspace) -> dict[str, Any]:
    splits = load_splits(ws.splits)
    summary = read_summary(ws.manifest)
    return {
        "synthetic": bool(summary.get("dataset", {}).get("synthetic")),
        "counts": splits.raw.get("counts", {}),
        "excluded": splits.excluded,
        "seed": splits.seed,
        "test_fraction": splits.test_fraction,
        "n_folds": splits.n_folds,
        "stratify_by": splits.raw.get("stratify_by", []),
        "manifest_counts": summary.get("counts", {}),
    }


def evaluate_dev(ws: Workspace, log: Callable[[str], None] = print) -> dict[str, Any]:
    dev = development_results(ws, log)
    test_path = ws.evaluation / "test_results.json"
    test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.is_file() else None
    write_report(ws.report, data=_data_summary(ws), dev=dev, test=test)
    append_log(
        ws.evaluation_log,
        {"split": "dev", "status": "completed", "auc": dev["calibration"]["oof"]["auc"]},
    )
    log(
        f"Cross-validation report written to {ws.report} (pooled out-of-fold AUC "
        f"{dev['calibration']['oof']['auc']:.4f}); figures in {ws.figures}."
    )
    return dev


# --- the locked test set ------------------------------------------------------------------------


def _subgroups(
    rows: dict[str, dict[str, Any]], ids: list[str], labels, scores
) -> list[dict[str, Any]]:
    out = []
    groups = {
        "sex": ["male", "female"],
        "age band": list(AGE_BANDS),
        "manufacturer": sorted({rows[c]["manufacturer_group"] for c in ids}),
    }
    keys = {"sex": "sex", "age band": "age_band", "manufacturer": "manufacturer_group"}
    y, s = np.asarray(labels), np.asarray(scores)
    for group, values in groups.items():
        for value in values:
            mask = np.array([rows[c][keys[group]] == value for c in ids])
            n_tb, n_ntm = int(y[mask].sum()), int((1 - y[mask]).sum())
            row = {
                "group": group,
                "value": value,
                "n": int(mask.sum()),
                "n_tb": n_tb,
                "n_ntm": n_ntm,
            }
            if min(n_tb, n_ntm) >= SUBGROUP_MIN_PER_CLASS:
                result = bootstrap(y[mask], s[mask], {"auc": roc_auc})["auc"]
                row.update(auc=result["value"], auc_ci=result["ci"])
            else:
                row.update(auc=None, auc_ci=None, note="too few cases of one class for an AUC")
            if row["n"]:
                out.append(row)
    return out


def _predict_test(
    bundle_path: Path, ws: Workspace, ids: list[str]
) -> tuple[list[float], list[float]]:
    bundle = load_bundle(bundle_path)
    if bundle["preprocessing"] != cache_config(ws.cache).to_dict():
        raise EvaluationError(
            "The bundle's preprocessing parameters differ from those of the cache."
        )
    ensemble = Ensemble(bundle)
    logits, probabilities = [], []
    for case_id in ids:
        prediction = ensemble.predict(load_volume(ws.cache, case_id, mmap=False))
        logits.append(prediction.logit)
        probabilities.append(prediction.decision.probability_tb)
    return logits, probabilities


def evaluate_test(
    ws: Workspace,
    *,
    bundle_path: Path,
    force: bool = False,
    reason: str | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    if not bundle_path.is_file():
        raise EvaluationError(f"{bundle_path} not found; run `cavinet-ml export` first.")
    previous = previous_test_runs(ws.evaluation_log)
    if previous and not force:
        append_log(
            ws.evaluation_log,
            {"split": "test", "status": "refused", "previous_runs": len(previous)},
        )
        raise EvaluationRefused(
            f"The locked test set was already evaluated ({len(previous)} run(s), first on "
            f"{previous[0]['time']} by {previous[0].get('user')}; see {ws.evaluation_log}). "
            "It may be "
            "used only once. Re-running needs --force with --reason, and is reported."
        )
    if force and not (reason and reason.strip()):
        raise EvaluationError(
            "--force needs --reason explaining why the test set is evaluated again."
        )
    bundle_sha = bundle_file_sha256(bundle_path)
    bundle = load_bundle(bundle_path, mmap=True)
    weights = weights_sha256(bundle["fold_state_dicts"])
    run_number = len(previous) + 1
    append_log(
        ws.evaluation_log,
        {
            "split": "test",
            "status": "started",
            "run": run_number,
            "forced": force,
            "reason": reason,
            "bundle_sha256": bundle_sha,
            "weights_sha256": weights,
            "is_demo": bool(bundle["is_demo"]),
        },
    )
    try:
        results = _locked_test(ws, bundle_path, bundle, run_number, force, reason, weights, log)
    except Exception as error:
        append_log(
            ws.evaluation_log,
            {
                "split": "test",
                "status": "failed",
                "run": run_number,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    append_log(
        ws.evaluation_log,
        {
            "split": "test",
            "status": "completed",
            "run": run_number,
            "forced": force,
            "auc": results["ct"]["metrics"]["auc"]["value"],
            "auc_ci": results["ct"]["metrics"]["auc"]["ci"],
            "n": results["n"],
            "weights_sha256": weights,
        },
    )
    return results


def _locked_test(
    ws: Workspace,
    bundle_path: Path,
    bundle: dict[str, Any],
    run_number: int,
    forced: bool,
    reason: str | None,
    weights: str,
    log: Callable[[str], None],
) -> dict[str, Any]:
    splits = load_splits(ws.splits)
    rows = {r["case_id"]: r for r in read_manifest(ws.manifest)}
    ids = list(splits.test)
    labels = [rows[c]["label"] for c in ids]
    log(
        f"Evaluating the frozen model on the {len(ids)} patients of the locked test set "
        f"(run {run_number})…"
    )
    logits, ct = _predict_test(bundle_path, ws, ids)
    calibration = read_calibration(ws.calibration)
    youden = float(calibration["youden_threshold"])

    metrics = metrics_with_ci(labels, ct, 0.5)
    at_youden = metrics_with_ci(labels, ct, youden)
    confusion = {k: threshold_metrics(labels, ct, 0.5)[k] for k in ("tp", "tn", "fp", "fn")}
    bins = calibration_bins(labels, ct)

    baselines: dict[str, Any] = {}
    scores: dict[str, list[float]] = {"ct": ct}
    for kind in KINDS:
        predictions, model_info = fit_dev_predict_test(kind, rows, splits)
        s = [predictions[c] for c in ids]
        scores[kind] = s
        auc = bootstrap(labels, s, {"auc": roc_auc})["auc"]
        baselines[kind] = {
            "description": KINDS[kind],
            "auc": auc["value"],
            "auc_ci": auc["ci"],
            "comparison_with_ct": delong_test(labels, ct, s),
            **model_info,
        }
    baselines["metadata"]["interpretation"] = shortcut_interpretation(
        baselines["metadata"]["auc"], baselines["metadata"]["auc_ci"]
    )
    subgroups = _subgroups(rows, ids, labels, ct)

    auc = metrics["auc"]
    h2 = baselines["clinical"]["comparison_with_ct"]
    hypotheses = {
        "H1": {
            "supported": auc["ci"][0] > 0.5,
            "statement": (
                f"Test AUC {auc['value']:.3f} (95% CI {auc['ci'][0]:.3f} to {auc['ci'][1]:.3f}); "
                f"supported if the lower bound is above 0.50. Target AUC ≥ {TARGET_AUC}: "
                f"{'met' if auc['value'] >= TARGET_AUC else 'not met'}."
            ),
            "target_met": auc["value"] >= TARGET_AUC,
        },
        "H2": {
            "supported": h2["p_value"] is not None
            and h2["difference"] > 0
            and h2["p_value"] < 0.05,
            "statement": (
                f"CT AUC {h2['auc_a']:.3f} vs clinical-only AUC {h2['auc_b']:.3f} on the same "
                f"{h2['n']} patients: difference {h2['difference']:+.3f}"
                + (
                    f" (95% CI {h2['ci'][0]:+.3f} to {h2['ci'][1]:+.3f}), DeLong p = "
                    f"{h2['p_value']:.4f}"
                    if h2["p_value"] is not None
                    else f"; DeLong test not computable: {h2['note']}"
                )
                + "; supported if the CT model is better with p < 0.05."
            ),
        },
    }
    paths = {
        "roc_test": figures.roc_figure(
            [
                _curve("CT model", labels, ct, auc["ci"]),
                _curve(
                    "Clinical-only", labels, scores["clinical"], baselines["clinical"]["auc_ci"]
                ),
                _curve(
                    "Metadata-only", labels, scores["metadata"], baselines["metadata"]["auc_ci"]
                ),
            ],
            ws.figures / "roc_test.png",
            "ROC, locked test set",
        ),
        "reliability_test": figures.reliability_figure(
            bins,
            ws.figures / "reliability_test.png",
            metrics["ece"]["value"],
            "Reliability, locked test set",
        ),
        "confusion_test": figures.confusion_figure(
            confusion, ws.figures / "confusion_test.png", 0.5
        ),
        "scores_test": figures.score_histogram(labels, ct, ws.figures / "scores_test.png"),
        "subgroups_test": figures.subgroup_figure(subgroups, ws.figures / "subgroups_test.png"),
    }
    evaluated_at = datetime.now(UTC).isoformat(timespec="seconds")
    results = {
        "evaluated_at": evaluated_at,
        "run": run_number,
        "forced": forced,
        "reason": reason,
        "bundle": {
            "file": str(bundle_path),
            "sha256": bundle_file_sha256(bundle_path),
            "weights_sha256": weights,
            "model_name": bundle["model_name"],
            "model_version": bundle.get("model_version"),
            "is_demo": bool(bundle["is_demo"]),
            "created_at": bundle["created_at"],
        },
        "n": len(ids),
        "n_tb": int(sum(labels)),
        "n_ntm": len(ids) - int(sum(labels)),
        "ct": {
            "metrics": metrics,
            "auc_delong": delong_ci(labels, ct),
            "confusion": confusion,
            "calibration_bins": bins,
            "youden_threshold": youden,
            "at_youden": at_youden,
        },
        "baselines": baselines,
        "subgroups": subgroups,
        "hypotheses": hypotheses,
        "figures": {k: str(v) for k, v in paths.items()},
    }
    ws.evaluation.mkdir(parents=True, exist_ok=True)
    with (ws.evaluation / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "label", "ct_probability", "ct_logit", "clinical", "metadata"])
        for i, case_id in enumerate(ids):
            writer.writerow(
                [
                    case_id,
                    labels[i],
                    f"{ct[i]:.6f}",
                    f"{logits[i]:.6f}",
                    f"{scores['clinical'][i]:.6f}",
                    f"{scores['metadata'][i]:.6f}",
                ]
            )
    (ws.evaluation / "test_results.json").write_text(
        json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8"
    )

    dev = development_results(ws, log)
    write_report(ws.report, data=_data_summary(ws), dev=dev, test=results)
    locked_test = {
        "auc": auc["value"],
        "auc_ci": auc["ci"],
        **{k: metrics[k]["value"] for k in metrics if k != "auc"},
        "n": len(ids),
        "n_tb": results["n_tb"],
        "n_ntm": results["n_ntm"],
        "threshold": 0.5,
        "evaluated_at": evaluated_at,
        "ci": metrics,
    }
    update_bundle_metrics(bundle_path, locked_test, ws.model_card_md)
    verdict = {
        h: "supported" if hypotheses[h]["supported"] else "NOT supported" for h in ("H1", "H2")
    }
    log(
        f"Locked test set: AUC {auc['value']:.4f} (95% CI {auc['ci'][0]:.3f}-{auc['ci'][1]:.3f}), "
        f"sensitivity {metrics['sensitivity']['value']:.3f}, specificity "
        f"{metrics['specificity']['value']:.3f}. H1 {verdict['H1']}; H2 {verdict['H2']}. "
        f"Report: {ws.report}; "
        f"model card and bundle metrics updated ({bundle_path})."
    )
    return results


# --- compare (FR-10.8) --------------------------------------------------------------------------


def _scores_for(ws: Workspace, name: str, split: str) -> dict[str, tuple[int, float]]:
    if split == "dev":
        sources = {
            "ct": ws.oof_predictions,
            "clinical": ws.baselines / "clinical_oof.csv",
            "metadata": ws.baselines / "metadata_oof.csv",
        }
        if name in sources:
            return read_oof_predictions(sources[name], "probability")
    else:
        columns = {"ct": "ct_probability", "clinical": "clinical", "metadata": "metadata"}
        if name in columns:
            path = ws.evaluation / "test_predictions.csv"
            if not path.is_file():
                raise EvaluationError(f"{path} not found; evaluate the test split first.")
            return read_oof_predictions(path, columns[name])
    return read_oof_predictions(Path(name), "probability")


def compare(
    ws: Workspace, a: str, b: str, *, split: str = "dev", log: Callable[[str], None] = print
) -> dict[str, Any]:
    """DeLong test of two models' AUCs on the same patients (default: CT vs clinical)."""
    first, second = _scores_for(ws, a, split), _scores_for(ws, b, split)
    ids = sorted(set(first) & set(second))
    if any(first[c][0] != second[c][0] for c in ids):
        raise EvaluationError("The two prediction files disagree on labels.")
    labels = [first[c][0] for c in ids]
    result = delong_test(labels, [first[c][1] for c in ids], [second[c][1] for c in ids])
    result.update(a=a, b=b, split=split, created_at=datetime.now(UTC).isoformat(timespec="seconds"))
    ws.baselines.mkdir(parents=True, exist_ok=True)
    out = ws.baselines / f"compare_{Path(a).stem}_vs_{Path(b).stem}_{split}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    detail = (
        f"(95% CI {result['ci'][0]:+.3f} to {result['ci'][1]:+.3f}), DeLong z = "
        f"{result['z']:.2f}, p = {result['p_value']:.4f}"
        if result["p_value"] is not None
        else f"(DeLong test not computable: {result['note']})"
    )
    log(
        f"{a} AUC {result['auc_a']:.4f} vs {b} AUC {result['auc_b']:.4f} on {result['n']} "
        f"{split} patients: difference {result['difference']:+.4f} {detail}. Written to {out}."
    )
    return result
