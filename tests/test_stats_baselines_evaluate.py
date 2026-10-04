"""Section 11.7 statistics, FR-10.8 baselines and comparisons, FR-10.7 evaluation and guard."""

import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from toolkit_helpers import make_workspace, narrow_network

from cavinet_ml.evaluation import evaluate as evaluation
from cavinet_ml.evaluation.baselines import (
    features,
    fit_dev_predict_test,
    run_baseline,
    shortcut_interpretation,
)
from cavinet_ml.evaluation.metrics import (
    calibration_bins,
    expected_calibration_error,
    roc_auc,
    roc_curve,
    threshold_metrics,
    youden_threshold,
)
from cavinet_ml.evaluation.stats import bootstrap, delong_ci, delong_test, metrics_with_ci
from cavinet_ml.model.bundle import load_bundle
from cavinet_ml.training.calibrate import calibrate
from cavinet_ml.training.config import TrainingConfig, with_overrides
from cavinet_ml.training.export import export_bundle
from cavinet_ml.training.trainer import train_fold

QUIET = {"log": lambda message: None}


@pytest.fixture(scope="module", autouse=True)
def _narrow_network():
    with narrow_network():
        yield


RNG = np.random.default_rng(0)
Y = np.array([1] * 30 + [0] * 20)
SCORES = np.round(RNG.normal(Y * 0.8, 1.0), 1)  # rounded, so there are ties
OTHER = np.round(SCORES + RNG.normal(0, 0.8, Y.size), 1)


# --- metrics -------------------------------------------------------


def test_auc_matches_scikit_learn_with_ties():
    assert roc_auc(Y, SCORES) == pytest.approx(roc_auc_score(Y, SCORES))
    assert np.isnan(roc_auc([1, 1], [0.2, 0.3]))


def test_threshold_metrics():
    m = threshold_metrics([1, 1, 1, 0, 0], [0.9, 0.6, 0.4, 0.7, 0.1])
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 1)
    assert m["sensitivity"] == pytest.approx(2 / 3) and m["specificity"] == pytest.approx(1 / 2)
    assert m["ppv"] == pytest.approx(2 / 3) and m["npv"] == pytest.approx(1 / 2)
    assert m["accuracy"] == pytest.approx(3 / 5) and m["f1"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert np.isnan(threshold_metrics([0, 0], [0.1, 0.2])["sensitivity"])


def test_roc_curve_and_youden_threshold():
    fpr, tpr, _ = roc_curve(Y, SCORES)
    assert (fpr[0], tpr[0], fpr[-1], tpr[-1]) == (0, 0, 1, 1)
    area = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    assert area == pytest.approx(roc_auc(Y, SCORES))
    assert youden_threshold([0, 0, 1, 1], [0.1, 0.3, 0.6, 0.8]) == pytest.approx(0.6)


def test_calibration_bins_and_ece():
    labels, probabilities = [0, 1, 1, 1], [0.1, 0.1, 0.9, 0.9]
    bins = calibration_bins(labels, probabilities)
    assert len(bins) == 10 and bins[1]["count"] == 2 and bins[1]["observed"] == 0.5
    assert bins[9]["count"] == 2 and bins[9]["mean_predicted"] == pytest.approx(0.9)
    assert expected_calibration_error(labels, probabilities) == pytest.approx(0.25)
    assert expected_calibration_error([0, 1], [0.0, 1.0]) == pytest.approx(0.0)


# --- bootstrap and DeLong -------------------------------------------------------


def test_bootstrap_is_stratified_reproducible_and_brackets_the_value():
    a = bootstrap(Y, SCORES, {"auc": roc_auc}, n=300)
    b = bootstrap(Y, SCORES, {"auc": roc_auc}, n=300)
    assert a == b
    low, high = a["auc"]["ci"]
    assert low <= a["auc"]["value"] <= high and high - low > 0.05


def test_metrics_with_ci_reports_every_section_11_7_metric():
    result = metrics_with_ci(Y, 1 / (1 + np.exp(-SCORES)), n=200)
    assert set(result) == {
        "auc",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "brier",
        "ece",
    }
    assert all(len(r["ci"]) == 2 for r in result.values())


def _structural(y, s):
    pos, neg = s[y == 1], s[y == 0]
    psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    return psi.mean(axis=1), psi.mean(axis=0)


def test_delong_matches_the_definition():
    v10, v01 = _structural(Y, SCORES)
    slow_var = np.var(v10, ddof=1) / v10.size + np.var(v01, ddof=1) / v01.size
    result = delong_ci(Y, SCORES)
    assert result["auc"] == pytest.approx(roc_auc(Y, SCORES))
    assert result["se"] ** 2 == pytest.approx(slow_var)
    a10, a01 = _structural(Y, SCORES)
    b10, b01 = _structural(Y, OTHER)
    var_a = np.var(a10, ddof=1) / a10.size + np.var(a01, ddof=1) / a01.size
    var_b = np.var(b10, ddof=1) / b10.size + np.var(b01, ddof=1) / b01.size
    cov = np.cov(a10, b10)[0, 1] / a10.size + np.cov(a01, b01)[0, 1] / a01.size
    test = delong_test(Y, SCORES, OTHER)
    assert test["se"] ** 2 == pytest.approx(var_a + var_b - 2 * cov)
    assert test["difference"] == pytest.approx(roc_auc(Y, SCORES) - roc_auc(Y, OTHER))
    assert 0 < test["p_value"] < 1 and test["n"] == 50 and test["n_tb"] == 30


def test_delong_edge_cases():
    same = delong_test(Y, SCORES, SCORES)
    assert same["p_value"] == 1.0 and same["difference"] == 0.0
    better = delong_test(Y, Y + RNG.normal(0, 0.01, Y.size), RNG.normal(0, 1, Y.size))
    assert better["p_value"] < 0.001
    few = delong_test([1, 1, 0], [0.9, 0.8, 0.1], [0.5, 0.4, 0.6])
    assert few["p_value"] is None and "too few" in few["note"]
    assert delong_ci([1, 0], [0.9, 0.1])["ci"] is None
    with pytest.raises(ValueError):
        delong_test([1, 1], [0.1, 0.2], [0.3, 0.4])


# --- baselines -------------------------------------------------------


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    import torch

    torch.set_num_threads(2)
    ws = make_workspace(tmp_path_factory.mktemp("eval"), n_tb=20, n_ntm=10)
    config = with_overrides(
        TrainingConfig(), epochs=1, batch_size=2, num_workers=0, pretrained="none"
    )
    for fold in (0, 1):
        train_fold(
            fold,
            config=config,
            manifest_path=ws.manifest,
            splits_path=ws.splits,
            cache_dir=ws.cache,
            runs_dir=ws.runs,
            device="cpu",
            **QUIET,
        )
    calibrate(runs_dir=ws.runs, splits_path=ws.splits, out=ws.calibration, **QUIET)
    export_bundle(
        runs_dir=ws.runs,
        splits_path=ws.splits,
        manifest_path=ws.manifest,
        cache_dir=ws.cache,
        calibration_path=ws.calibration,
        out=ws.bundle,
        model_card_md=ws.model_card_md,
        **QUIET,
    )
    return ws


def test_features_code_sex_and_missing_values():
    rows = [
        {
            "case_id": "TB_001",
            "age": 40,
            "sex": "male",
            "manufacturer_group": "GE",
            "kernel": "lung",
            "slice_thickness_mm": 1.25,
            **{c: 1 for c in ("chest_pain", "cough")},
        },
        {
            "case_id": "TB_002",
            "age": None,
            "sex": None,
            "manufacturer_group": "",
            "kernel": None,
            "slice_thickness_mm": None,
        },
    ]
    from cavinet_ml.dataset.patient_index import SYMPTOM_COLUMNS

    for row in rows:
        for column in SYMPTOM_COLUMNS:
            row.setdefault(column, None)
    clinical = features(rows, "clinical")
    assert list(clinical.columns)[:2] == ["age", "sex_male"]
    assert clinical.loc["TB_001", "sex_male"] == 1.0 and np.isnan(
        clinical.loc["TB_002", "sex_male"]
    )
    assert np.isnan(clinical.loc["TB_002", "age"]) and np.isnan(clinical.loc["TB_002", "cough"])
    metadata = features(rows, "metadata")
    assert (
        metadata.loc["TB_001", "kernel"] == "LUNG" and metadata.loc["TB_002", "kernel"] == "UNKNOWN"
    )
    assert metadata.loc["TB_002", "manufacturer_group"] == "unknown"
    with pytest.raises(ValueError):
        features(rows, "other")


def test_clinical_baseline_uses_the_same_folds_and_learns_the_age_signal(evaluated):
    ws = evaluated
    summary = run_baseline(
        "clinical", manifest_path=ws.manifest, splits_path=ws.splits, out_dir=ws.baselines, **QUIET
    )
    assert summary["n"] == 24 and len(summary["auc_per_fold"]) == 2
    assert summary["auc"] > 0.6  # TB patients are younger in the test workspace
    lines = (ws.baselines / "clinical_oof.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "case_id,fold,label,probability" and len(lines) == 25
    assert (
        json.loads((ws.baselines / "clinical.json").read_text(encoding="utf-8"))["auc"]
        == summary["auc"]
    )


def test_out_of_fold_predictions_learn_a_real_signal():
    from toolkit_helpers import manifest_rows

    from cavinet_ml.dataset.split import Splits, make_splits
    from cavinet_ml.evaluation.baselines import out_of_fold

    rows = manifest_rows(120, 80, seed=3)
    by_id = {r["case_id"]: r for r in rows}
    test, folds = make_splits(rows)
    splits = Splits(test=test, folds=folds)
    predictions = out_of_fold("clinical", by_id, splits)
    assert set(predictions) == set(splits.dev)  # every development patient, no test patient
    ids = splits.dev
    assert roc_auc([by_id[c]["label"] for c in ids], [predictions[c] for c in ids]) > 0.85


def test_shortcut_check_runs_and_explains_itself(evaluated):
    ws = evaluated
    summary = run_baseline(
        "metadata", manifest_path=ws.manifest, splits_path=ws.splits, out_dir=ws.baselines, **QUIET
    )
    assert "interpretation" in summary and (ws.baselines / "metadata_oof.csv").is_file()


def test_shortcut_interpretation():
    assert shortcut_interpretation(0.8, [0.7, 0.9]).startswith("WARNING")
    assert "weak signal" in shortcut_interpretation(0.6, [0.52, 0.68])
    assert "does not separate" in shortcut_interpretation(0.5, [0.4, 0.6])


def test_baseline_for_the_test_set_is_fitted_on_the_development_set(evaluated):
    from cavinet_ml.dataset.manifest import read_manifest
    from cavinet_ml.dataset.split import load_splits

    ws = evaluated
    rows = {r["case_id"]: r for r in read_manifest(ws.manifest)}
    splits = load_splits(ws.splits)
    predictions, info = fit_dev_predict_test("clinical", rows, splits)
    assert set(predictions) == set(splits.test)
    assert "age" in info["coefficients"] and info["coefficients"]["age"] < 0  # older → NTM


# --- evaluation and the guard -------------------------------------------------------


def test_compare_runs_delong_on_the_development_set(evaluated):
    ws = evaluated
    result = evaluation.compare(ws, "ct", "clinical", **QUIET)
    assert result["n"] == 24 and result["a"] == "ct" and result["split"] == "dev"
    assert (ws.baselines / "compare_ct_vs_clinical_dev.json").is_file()
    with pytest.raises(evaluation.EvaluationError, match="evaluate the test split first"):
        evaluation.compare(ws, "ct", "clinical", split="test", **QUIET)


def test_development_report_before_the_test_set(evaluated):
    ws = evaluated
    evaluation.evaluate_dev(ws, **QUIET)
    text = ws.report.read_text(encoding="utf-8")
    assert "## Cross-validation (development set)" in text and "Not evaluated yet" in text
    assert "SYNTHETIC REHEARSAL" in text and "Clinical-only baseline" in text
    assert (ws.figures / "roc_dev.png").is_file() and (ws.figures / "training_curves.png").is_file()
    assert evaluation.read_log(ws.evaluation_log)[-1]["split"] == "dev"


def test_locked_test_set_is_evaluated_once(evaluated):
    ws = evaluated
    results = evaluation.evaluate_test(ws, bundle_path=ws.bundle, **QUIET)
    assert results["n"] == 6 and results["run"] == 1
    assert set(results["hypotheses"]) == {"H1", "H2"}
    assert results["baselines"]["clinical"]["comparison_with_ct"]["n"] == 6
    assert {r["group"] for r in results["subgroups"]} == {"sex", "age band", "manufacturer"}
    for name in ("roc_test", "reliability_test", "confusion_test", "scores_test", "subgroups_test"):
        assert (ws.figures / f"{name}.png").is_file()
    text = ws.report.read_text(encoding="utf-8")
    assert "## Locked test set" in text and "H1:" in text and "| AUC |" in text
    bundle = load_bundle(ws.bundle)
    assert bundle["metrics"]["locked_test"]["n"] == 6
    assert "Evaluated once on 6 patients" in ws.model_card_md.read_text(encoding="utf-8")
    assert (
        (ws.evaluation / "test_predictions.csv")
        .read_text(encoding="utf-8")
        .startswith("case_id,label,ct_probability")
    )

    with pytest.raises(evaluation.EvaluationRefused, match="already evaluated"):
        evaluation.evaluate_test(ws, bundle_path=ws.bundle, **QUIET)
    with pytest.raises(evaluation.EvaluationError, match="--force needs --reason"):
        evaluation.evaluate_test(ws, bundle_path=ws.bundle, force=True, **QUIET)
    again = evaluation.evaluate_test(ws, bundle_path=ws.bundle, force=True, reason="test", **QUIET)
    assert again["run"] == 2 and again["forced"]
    statuses = [(e["split"], e["status"]) for e in evaluation.read_log(ws.evaluation_log)]
    assert statuses.count(("test", "started")) == 2 and statuses.count(("test", "completed")) == 2
    assert ("test", "refused") in statuses
    compared = evaluation.compare(ws, "ct", "metadata", split="test", **QUIET)
    assert compared["n"] == 6


def test_a_failed_test_run_is_logged_and_counts(tmp_path, evaluated, monkeypatch):
    from cavinet_ml.workspace import Workspace

    ws = evaluated
    copy = Workspace(work=ws.work, splits=ws.splits, docs=tmp_path / "docs")
    monkeypatch.setattr(
        evaluation, "_locked_test", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError):
        evaluation.evaluate_test(copy, bundle_path=ws.bundle, **QUIET)
    log = evaluation.read_log(copy.evaluation_log)
    assert [e["status"] for e in log] == ["started", "failed"] and "boom" in log[-1]["error"]
    assert len(evaluation.previous_test_runs(copy.evaluation_log)) == 1
    with pytest.raises(evaluation.EvaluationError, match="not found"):
        evaluation.evaluate_test(copy, bundle_path=tmp_path / "none.pth", **QUIET)
