"""M-12 symptom-assisted prediction: the fusion maths (FR-12.2), missing values, the fusion file,
`fetch-model` installing it, and `cavinet-ml fusion` (H3, locked test set once)."""

import csv
import json
import math

import numpy as np
import pytest
from toolkit_helpers import make_workspace

from cavinet_ml.cli import main
from cavinet_ml.dataset.manifest import read_manifest
from cavinet_ml.dataset.split import load_splits
from cavinet_ml.evaluation import fusion as training
from cavinet_ml.evaluation.evaluate import EvaluationError, EvaluationRefused, read_log
from cavinet_ml.fetch import install_fusion
from cavinet_ml.fusion import (
    FEATURES,
    FUSION_FILE,
    SYMPTOM_NAMES,
    FusionError,
    FusionModel,
    demo_fusion,
    load_fusion,
    save_fusion,
    symptom_vector,
)
from cavinet_ml.model.bundle import load_bundle, save_bundle

QUIET = {"log": lambda message: None}


def _synthetic(n: int = 300, seed: int = 0):
    """CT logit and fever carry signal; a few values are missing."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    x = np.zeros((n, len(FEATURES)))
    x[:, 0] = rng.normal(labels * 1.5 - 0.75, 1.0)  # ct_logit
    x[:, 1] = rng.normal(50, 12, n)  # age
    x[:, 2] = rng.integers(0, 2, n)  # sex_male
    x[:, 3:] = rng.random((n, len(FEATURES) - 3)) < 0.3
    fever = FEATURES.index("fever")
    x[:, fever] = rng.random(n) < np.where(labels == 1, 0.7, 0.2)
    x[:5, 1] = np.nan
    x[5:8, 2] = np.nan
    x[8:10, fever] = np.nan
    return x, labels


def test_fit_and_apply_match_scikit_learn():
    from sklearn.linear_model import LogisticRegression

    x, labels = _synthetic()
    params = training.fit(x, labels)
    model = FusionModel(params)
    prepared = np.array([model.prepared(dict(zip(FEATURES, row, strict=True))) for row in x])
    reference = LogisticRegression(class_weight="balanced", C=1.0, max_iter=5000)
    reference.fit(prepared, labels)
    ours = np.array(training.predict(params, x))
    assert np.allclose(ours, reference.predict_proba(prepared)[:, 1], atol=1e-9)
    assert params["coefficients"]["ct_logit"] > 0.5
    assert params["coefficients"]["fever"] > 0.5  # learns the symptom signal
    assert set(params["coefficients"]) == set(FEATURES)


def test_imputation_and_scaling_come_from_the_training_patients():
    x, labels = _synthetic()
    params = training.fit(x, labels)
    p = params["preprocessing"]
    known_age = x[~np.isnan(x[:, 1]), 1]
    assert p["median"]["age"] == pytest.approx(float(np.median(known_age)))
    assert p["mode"]["sex_male"] in (0.0, 1.0)
    model = FusionModel(params)
    # A missing age counts as the training median: the same probability as giving it.
    common = {"ct_logit": 0.3, "age": None, "sex": "male", "symptoms": ["cough"]}
    with_median = {**common, "age": p["median"]["age"]}
    assert model.predict(**common) == pytest.approx(model.predict(**with_median))
    # Unknown sex takes the training mode.
    mode_sex = "male" if p["mode"]["sex_male"] == 1.0 else "female"
    assert model.predict(**{**common, "sex": None}) == pytest.approx(
        model.predict(**{**common, "sex": mode_sex})
    )


def test_predict_is_the_logistic_formula():
    params = demo_fusion("abc")
    model = FusionModel(params)
    p = model.predict(ct_logit=1.0, age=65, sex="female", symptoms=["fever", "night_sweats"])
    z = -0.3 + 1.5 * 1.0 + 0.2 * (65 - 50) / 15 + 0.4 + 0.4
    assert p == pytest.approx(1 / (1 + math.exp(-z)))
    # More CT evidence for TB never lowers the combined probability.
    assert model.predict(ct_logit=2.0, age=65, sex="female", symptoms=[]) > model.predict(
        ct_logit=-2.0, age=65, sex="female", symptoms=[]
    )
    assert model.predict(ct_logit=-800.0, age=None, sex=None, symptoms=[]) < 1e-100  # no overflow


def test_symptoms_are_the_twelve_dataset_symptoms():
    assert len(SYMPTOM_NAMES) == 12 and SYMPTOM_NAMES["night_sweats"] == "Night sweats"
    vector = symptom_vector(["cough", "fever"])
    assert sum(vector.values()) == 2 and vector["cough"] == 1.0 and vector["chills"] == 0.0
    assert sum(symptom_vector([]).values()) == 0
    with pytest.raises(FusionError, match="unknown symptoms: sneezing"):
        symptom_vector(["sneezing"])


def test_fusion_file_round_trip_and_validation(tmp_path):
    path = save_fusion(demo_fusion("sha-1"), tmp_path / FUSION_FILE)
    model = load_fusion(path)
    assert model.is_demo and model.version == "demo"
    assert model.matches("sha-1") and not model.matches("sha-2") and not model.matches(None)
    for broken in (
        {**demo_fusion("x"), "format": 99},
        {**demo_fusion("x"), "features": ["ct_logit"]},
        {**demo_fusion("x"), "intercept": "high"},
        {k: v for k, v in demo_fusion("x").items() if k != "coefficients"},
    ):
        (tmp_path / "bad.json").write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(FusionError):
            load_fusion(tmp_path / "bad.json")
    zero_scale = demo_fusion("x")
    zero_scale["preprocessing"]["scale"]["age"] = 0
    with pytest.raises(FusionError, match="scale"):
        save_fusion(zero_scale, tmp_path / "z.json")
    (tmp_path / "text.json").write_text("not json", encoding="utf-8")
    with pytest.raises(FusionError):
        load_fusion(tmp_path / "text.json")
    with pytest.raises(FusionError, match="not found"):
        load_fusion(tmp_path / "missing.json")


def test_fetch_model_installs_the_matching_fusion_file(tmp_path, tiny_bundle_path):
    from cavinet_ml.fetch import bundle_weights_sha256

    demo = tmp_path / "demo" / "cavinet_model.pth"
    demo.parent.mkdir()
    demo.write_bytes(tiny_bundle_path.read_bytes())
    out = install_fusion(demo, **QUIET)
    assert out == demo.with_name(FUSION_FILE) and load_fusion(out).is_demo

    # A "released" (non-demo) model: only a parameter file fitted for its weights is used.
    bundle = load_bundle(tiny_bundle_path)
    bundle["is_demo"] = False
    real = save_bundle(bundle, tmp_path / "real" / "cavinet_model.pth")
    weights = bundle_weights_sha256(load_bundle(real))
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    (params_dir / "broken.json").write_text("{}", encoding="utf-8")
    other = {**demo_fusion("someone-else"), "is_demo": False, "version": "other"}
    save_fusion(other, params_dir / "a-other.json")
    stale = real.with_name(FUSION_FILE)
    save_fusion(other, stale)
    messages: list[str] = []
    assert install_fusion(real, params_dir=params_dir, log=messages.append) is None
    assert not stale.exists() and "CT-only" in messages[-1]

    mine = {**demo_fusion(weights), "is_demo": False, "version": "model-v1-fusion"}
    save_fusion(mine, params_dir / "b-mine.json")
    out = install_fusion(real, params_dir=params_dir, **QUIET)
    assert load_fusion(out).version == "model-v1-fusion"


def test_fetch_model_keeps_the_model_if_the_fusion_cannot_be_installed(
    tmp_path, tiny_bundle_path, monkeypatch
):
    from cavinet_ml import fetch

    out = tmp_path / "cavinet_model.pth"
    out.write_bytes(tiny_bundle_path.read_bytes())
    monkeypatch.setattr(
        fetch, "install_fusion", lambda *a, **k: (_ for _ in ()).throw(OSError("x"))
    )
    messages: list[str] = []
    assert fetch.fetch_model(out, log=messages.append) == "kept"
    assert "not installed (OSError: x)" in messages[-1]


# --- cavinet-ml fusion ----------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    """A toolkit workspace with the CT model's outputs written directly (no training)."""
    ws = make_workspace(tmp_path, n_tb=40, n_ntm=30, folds=3, size=8)
    splits = load_splits(ws.splits)
    rows = {r["case_id"]: r for r in read_manifest(ws.manifest)}
    rng = np.random.default_rng(3)

    def logit(case_id: str) -> float:
        return float(rng.normal(1.0 if rows[case_id]["label"] else -1.0, 1.2))

    with ws.oof_predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "fold", "label", "logit", "probability"])
        for case_id in splits.dev:
            z = logit(case_id)
            writer.writerow(
                [
                    case_id,
                    splits.fold_of(case_id),
                    rows[case_id]["label"],
                    z,
                    1 / (1 + math.exp(-z)),
                ]
            )
    ws.evaluation.mkdir(parents=True, exist_ok=True)
    with (ws.evaluation / "test_predictions.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.writer(h)
        writer.writerow(["case_id", "label", "ct_probability", "ct_logit", "clinical", "metadata"])
        for case_id in splits.test:
            z = logit(case_id)
            writer.writerow(
                [case_id, rows[case_id]["label"], 1 / (1 + math.exp(-z / 1.3)), z, 0.5, 0.5]
            )
    (ws.evaluation / "test_results.json").write_text(
        json.dumps(
            {"bundle": {"weights_sha256": "w" * 64, "model_version": "model-v1", "is_demo": False}}
        ),
        encoding="utf-8",
    )
    ws.report.parent.mkdir(parents=True, exist_ok=True)
    ws.report.write_text(
        "# Evaluation report\n\n## Summary\n\n| Question | Result |\n|---|---|\n"
        "| H1: x | **Supported**. |\n| H2: y | **Supported**. |\n"
        "| H4: the heatmap falls mostly inside the lungs | **Not supported**. |\n\n"
        "## Locked test set\n\nText.\n\n## Methods\n\nMethods.\n",
        encoding="utf-8",
    )
    return ws


def test_fusion_fits_on_development_and_evaluates_the_test_set_once(workspace):
    ws = workspace
    results = training.run_fusion(ws, **QUIET)
    splits = load_splits(ws.splits)
    out = ws.work / "fusion"
    params = load_fusion(out / FUSION_FILE)
    assert params.matches("w" * 64) and params.version == "model-v1-fusion"
    assert not params.is_demo and params.params["metrics"]["h3_supported"] in (True, False)
    assert results["development"]["n"] == len(splits.dev)
    assert results["test"]["n"] == len(splits.test)
    comparison = results["test"]["comparison"]
    assert comparison["n"] == len(splits.test)
    assert results["hypothesis"]["supported"] == (
        comparison["p_value"] is not None
        and comparison["difference"] > 0
        and comparison["p_value"] < 0.05
    )
    with (out / "fusion_oof.csv").open(encoding="utf-8") as handle:
        oof = list(csv.DictReader(handle))
    assert {r["case_id"] for r in oof} == set(splits.dev)
    with (out / "fusion_test_predictions.csv").open(encoding="utf-8") as handle:
        test = list(csv.DictReader(handle))
    assert [r["case_id"] for r in test] == list(splits.test)

    report = ws.report.read_text(encoding="utf-8")
    assert training.SECTION in report and report.index(training.SECTION) < report.index(
        "## Methods"
    )
    lines = report.splitlines()
    rows = [line[:4] for line in lines if line[:3] == "| H" and line[3:4].isdigit()]
    assert rows == ["| H1", "| H2", "| H3", "| H4"]
    assert "| Night sweats |" in report and "| CT model logit |" in report

    log = [e for e in read_log(ws.evaluation_log) if e["split"] == "fusion-test"]
    assert [e["status"] for e in log] == ["started", "completed"]

    # Once only: a second run is refused and logged, and needs --force with --reason.
    with pytest.raises(EvaluationRefused):
        training.run_fusion(ws, **QUIET)
    with pytest.raises(EvaluationError, match="--reason"):
        training.run_fusion(ws, force=True, **QUIET)
    again = training.run_fusion(ws, force=True, reason="test of the guard", **QUIET)
    assert again["run"] == 2 and again["forced"]
    # Re-running replaces the section and the summary row instead of adding new ones.
    report = ws.report.read_text(encoding="utf-8")
    assert report.count(training.SECTION) == 1 and report.count("| H3:") == 1
    statuses = [e["status"] for e in read_log(ws.evaluation_log) if e["split"] == "fusion-test"]
    assert statuses == ["started", "completed", "refused", "started", "completed"]


def test_fusion_needs_the_ct_model_outputs(workspace):
    ws = workspace
    (ws.evaluation / "test_predictions.csv").unlink()
    with pytest.raises(EvaluationError, match="evaluate the test split first"):
        training.run_fusion(ws, **QUIET)
    ws.oof_predictions.unlink()
    with pytest.raises(EvaluationError, match="calibrate"):
        training.run_fusion(ws, **QUIET)


def test_fusion_command(workspace, capsys):
    ws = workspace
    args = ["fusion", "--work", str(ws.work), "--splits", str(ws.splits), "--docs", str(ws.docs)]
    assert main(args) == 0
    assert "H3" in capsys.readouterr().out
    assert main(args) == 3  # refused the second time
    assert "refused" in capsys.readouterr().err


def test_rewritten_report_keeps_the_h3_section(workspace):
    from cavinet_ml.evaluation.evaluate import _keep_heatmap_section

    ws = workspace
    training.run_fusion(ws, **QUIET)
    ws.report.write_text(
        "# Evaluation report\n\n## Summary\n\n| Question | Result |\n|---|---|\n"
        "| H1: x | a |\n| H2: y | b |\n\n## Methods\n\nM.\n",
        encoding="utf-8",
    )
    _keep_heatmap_section(ws)
    report = ws.report.read_text(encoding="utf-8")
    assert training.SECTION in report and "| H3:" in report
