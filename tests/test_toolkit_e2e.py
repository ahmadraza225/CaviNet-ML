"""Phase 6 acceptance (section 15.7):

- the synthetic end-to-end pipeline runs through the CLI: synthetic Kaggle-like zip -> index ->
  preprocess -> split -> train 2 folds × 1 epoch (CPU) -> calibrate -> export -> baseline ->
  shortcut-check -> compare -> evaluate (dev, then test);
- the exported bundle loads in the application (the Phase 5 code the worker runs) and produces a
  result;
- evaluating the test split a second time without --force is refused.

Set CAVINET_E2E_OUT to keep the report, figures and model card (CI uploads them).
"""

import csv
import json
import os
import shutil
import zipfile
from pathlib import Path

import pytest

from cavinet_ml.analysis import Analyser
from cavinet_ml.cli import EXIT_ERROR, EXIT_REFUSED, main
from cavinet_ml.dataset.source import open_source
from cavinet_ml.inference.ensemble import Ensemble
from cavinet_ml.model.bundle import load_bundle

CI_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "ci.yaml"


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    import torch

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    root = tmp_path_factory.mktemp("e2e")
    docs = (
        Path(os.environ["CAVINET_E2E_OUT"]) if os.environ.get("CAVINET_E2E_OUT") else root / "docs"
    )
    if docs.exists():
        shutil.rmtree(docs)
    paths = ["--work", str(root / "work"), "--splits", str(root / "splits.json")]
    with_docs = [*paths, "--docs", str(docs)]
    steps = [
        ["synthetic-dataset", "--out", str(root / "data"), "--cases", "24"],
        [
            "index",
            "--data",
            str(root / "data" / "dicom-dataset.zip"),
            "--workers",
            "2",
            "--work",
            str(root / "work"),
        ],
        ["preprocess", "--work", str(root / "work"), "--workers", "2", "--no-lungmask"],
        ["split", *paths, "--folds", "2"],
        ["train", "--fold", "0", "--config", str(CI_CONFIG), "--device", "cpu", *paths],
        ["train", "--fold", "1", "--config", str(CI_CONFIG), "--device", "cpu", *paths],
        ["calibrate", *paths],
        ["export", *with_docs],
        ["baseline", *paths],
        ["shortcut-check", *paths],
        ["compare", *paths],
        ["evaluate", "--split", "dev", *with_docs],
        ["evaluate", "--split", "test", *with_docs],
    ]
    for step in steps:
        assert main(step) == 0, f"cavinet-ml {' '.join(step)} failed"
    return {"root": root, "work": root / "work", "docs": docs, "with_docs": with_docs}


def test_every_step_leaves_its_outputs(pipeline):
    work, docs = pipeline["work"], pipeline["docs"]
    for path in (
        work / "manifest.csv",
        work / "manifest.json",
        work / "qc" / "qc_report.csv",
        pipeline["root"] / "splits.json",
        work / "runs" / "fold_0" / "best.pt",
        work / "runs" / "fold_1" / "oof.csv",
        work / "calibration.json",
        work / "export" / "cavinet_model.pth",
        work / "export" / "model_card.json",
        work / "baselines" / "clinical_oof.csv",
        work / "baselines" / "metadata_oof.csv",
        work / "baselines" / "compare_ct_vs_clinical_dev.json",
        work / "evaluation" / "test_results.json",
        docs / "EVALUATION_REPORT.md",
        docs / "MODEL_CARD.md",
        docs / "figures" / "roc_test.png",
        docs / "audit" / "evaluation_runs.jsonl",
    ):
        assert path.is_file(), path
    splits = json.loads((pipeline["root"] / "splits.json").read_text(encoding="utf-8"))
    assert splits["n_folds"] == 2 and len(splits["excluded"]) == 3
    report = (docs / "EVALUATION_REPORT.md").read_text(encoding="utf-8")
    assert "SYNTHETIC REHEARSAL" in report and "## Locked test set" in report


def test_exported_bundle_loads_in_the_application_and_produces_a_result(pipeline, tmp_path):
    bundle = load_bundle(pipeline["work"] / "export" / "cavinet_model.pth")
    assert bundle["is_demo"] is True and bundle["metrics"]["locked_test"]["n"] > 0
    assert bundle["architecture"]["block_inplanes"] == [64, 128, 256, 512]  # the full network
    # The worker stores the metrics in a PostgreSQL JSON column, which refuses NaN.
    json.dumps({k: v for k, v in bundle.items() if k != "fold_state_dicts"}, allow_nan=False)
    with open_source(pipeline["root"] / "data" / "dicom-dataset.zip") as source:
        case = next(c for c in source.layout().cases if c.case_id == "TB_001")
        scan = source.extract_case(case, tmp_path / "scan")
    analyser = Analyser(Ensemble(bundle), segmenter=None)  # what the worker calls (M-05)
    result = analyser.infer(analyser.prepare(scan, tmp_path / "previews"))
    assert result.decision["predicted_class"] in ("TB", "NTM")
    assert 0.0 <= result.decision["probability_tb"] <= 1.0
    assert result.model["is_demo"] is True and result.model["folds"] == 2
    assert result.previews.count == 48 and len(result.fold_logits) == 2


def test_thesis_tables_are_written_from_the_outputs(pipeline):
    docs = pipeline["docs"]
    assert main(["tables", *pipeline["with_docs"]]) == 0
    tables = docs / "tables"
    names = sorted(p.name for p in tables.glob("*.csv"))
    assert names == [
        "1_dataset.csv",
        "2_folds.csv",
        "3_cross_validation.csv",
        "4_test_metrics.csv",
        "5_baselines.csv",
        "6_subgroups.csv",
        "7_hypotheses.csv",
    ]
    with (tables / "1_dataset.csv").open(encoding="utf-8") as handle:
        dataset = list(csv.reader(handle))
    patients = dict(zip(dataset[0], dataset[1], strict=True))
    splits = json.loads((pipeline["root"] / "splits.json").read_text(encoding="utf-8"))
    total = sum(int(patients[f"All {name}"]) for name in ("TB", "NTM"))
    assert total == len(splits["test"]) + sum(len(f) for f in splits["folds"])
    with (tables / "2_folds.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.reader(handle))) == 1 + 2  # header + the two CI folds
    summary = (tables / "TABLES.md").read_text(encoding="utf-8")
    assert "SYNTHETIC REHEARSAL" in summary
    for title in ("Table 1.", "Table 4. Locked test set performance", "Table 7. Hypotheses"):
        assert title in summary
    log = (docs / "audit" / "evaluation_runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert sum(json.loads(line)["split"] == "test" for line in log) == 2  # no new evaluation


def test_tables_need_the_test_results(tmp_path, capsys):
    assert main(["tables", "--work", str(tmp_path), "--docs", str(tmp_path / "docs")]) == EXIT_ERROR
    assert "evaluate --split test" in capsys.readouterr().err


def test_verification_set_has_real_zips_and_the_toolkit_probabilities(pipeline, tmp_path):
    out = tmp_path / "verification"
    data = pipeline["root"] / "data" / "dicom-dataset.zip"
    paths = ["--work", str(pipeline["work"]), "--splits", str(pipeline["root"] / "splits.json")]
    command = ["verification-set", "--data", str(data), "--out", str(out), *paths]
    assert main([*command, "--per-class", "1"]) == 0
    with (out / "expected.csv").open(encoding="utf-8") as handle:
        expected = list(csv.DictReader(handle))
    assert sorted(r["label"] for r in expected) == ["NTM", "TB"]
    with (pipeline["work"] / "evaluation" / "test_predictions.csv").open(encoding="utf-8") as h:
        toolkit = {r["case_id"]: float(r["ct_probability"]) for r in csv.DictReader(h)}
    for row in expected:
        assert float(row["toolkit_probability"]) == pytest.approx(toolkit[row["case_id"]], abs=1e-6)
        with zipfile.ZipFile(out / row["zip"]) as archive:
            names = archive.namelist()
        assert names and all(n.startswith(f"{row['case_id']}/") for n in names)
        assert len(names) >= int(row["slices"])
    assert main([*command, "--per-class", "50"]) == EXIT_ERROR  # more than the test set has


def test_heatmap_check_reports_h4_and_resumes(pipeline):
    data = pipeline["root"] / "data" / "dicom-dataset.zip"
    command = ["heatmap-check", "--data", str(data), "--no-lungmask", *pipeline["with_docs"]]
    assert main([*command, "--limit", "1"]) == 0
    shares = pipeline["work"] / "evaluation" / "heatmap_shares.csv"
    assert len(shares.read_text(encoding="utf-8").splitlines()) == 2  # header + one patient
    assert main(command) == 0  # resumes with the others
    splits = json.loads((pipeline["root"] / "splits.json").read_text(encoding="utf-8"))
    with shares.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert sorted(r["case_id"] for r in rows) == sorted(splits["test"])
    assert all(0.0 <= float(r["lung_share"]) <= 1.0 for r in rows)
    summary = json.loads(
        (pipeline["work"] / "evaluation" / "heatmap_check.json").read_text(encoding="utf-8")
    )
    assert summary["n"] == len(rows) and summary["supported"] == (summary["mean"] >= 0.8)
    report = (pipeline["docs"] / "EVALUATION_REPORT.md").read_text(encoding="utf-8")
    assert report.count("## Heatmap explanation (H4)") == 1
    assert report.index("## Heatmap explanation (H4)") < report.index("## Methods")
    # Rewriting the cross-validation report keeps the H4 section.
    assert main(["evaluate", "--split", "dev", *pipeline["with_docs"]]) == 0
    report = (pipeline["docs"] / "EVALUATION_REPORT.md").read_text(encoding="utf-8")
    assert report.count("## Heatmap explanation (H4)") == 1


def test_a_second_test_evaluation_is_refused(pipeline, capsys):
    again = ["evaluate", "--split", "test", *pipeline["with_docs"]]
    assert main(again) == EXIT_REFUSED
    assert "already evaluated" in capsys.readouterr().err
    assert main([*again, "--force"]) == EXIT_ERROR
    log = (
        (pipeline["docs"] / "audit" / "evaluation_runs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    statuses = [json.loads(line)["status"] for line in log if json.loads(line)["split"] == "test"]
    assert statuses == ["started", "completed", "refused"]
