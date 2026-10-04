"""Phase 6 acceptance (section 15.7):

- the synthetic end-to-end pipeline runs through the CLI: synthetic Kaggle-like zip -> index ->
  preprocess -> split -> train 2 folds × 1 epoch (CPU) -> calibrate -> export -> baseline ->
  shortcut-check -> compare -> evaluate (dev, then test);
- the exported bundle loads in the application (the Phase 5 code the worker runs) and produces a
  result;
- evaluating the test split a second time without --force is refused.

Set CAVINET_E2E_OUT to keep the report, figures and model card (CI uploads them).
"""

import json
import os
import shutil
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
    splits = json.loads((pipeline["root"] / "splits.json").read_text())
    assert splits["n_folds"] == 2 and len(splits["excluded"]) == 3
    report = (docs / "EVALUATION_REPORT.md").read_text()
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


def test_a_second_test_evaluation_is_refused(pipeline, capsys):
    again = ["evaluate", "--split", "test", *pipeline["with_docs"]]
    assert main(again) == EXIT_REFUSED
    assert "already evaluated" in capsys.readouterr().err
    assert main([*again, "--force"]) == EXIT_ERROR
    log = (pipeline["docs"] / "audit" / "evaluation_runs.jsonl").read_text().splitlines()
    statuses = [json.loads(line)["status"] for line in log if json.loads(line)["split"] == "test"]
    assert statuses == ["started", "completed", "refused"]
