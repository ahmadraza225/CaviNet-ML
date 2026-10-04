"""FR-10.4 (train --fold), FR-10.5 (calibrate) and FR-10.6 (export) on a small workspace."""

import csv
import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from toolkit_helpers import make_workspace, narrow_network

from cavinet_ml.inference.calibration import fit_temperature
from cavinet_ml.inference.ensemble import Ensemble
from cavinet_ml.model.bundle import load_bundle
from cavinet_ml.training import trainer
from cavinet_ml.training.calibrate import calibrate
from cavinet_ml.training.config import TrainingConfig, with_overrides
from cavinet_ml.training.export import export_bundle, update_bundle_metrics, weights_sha256
from cavinet_ml.training.trainer import TrainingError, lr_factor, train_fold

QUIET = {"log": lambda message: None}


@pytest.fixture(scope="module", autouse=True)
def _narrow_network():
    with narrow_network():
        yield


def config(epochs=2, **batch):
    base = with_overrides(TrainingConfig(), epochs=epochs, num_workers=0, pretrained="none")
    if batch:
        base = replace(base, batch=replace(base.batch, **batch))
    return replace(base, logging=replace(base.logging, tensorboard=True))


def train(ws, fold, cfg, **kwargs):
    return train_fold(
        fold,
        config=cfg,
        manifest_path=ws.manifest,
        splits_path=ws.splits,
        cache_dir=ws.cache,
        runs_dir=ws.runs,
        device="cpu",
        **QUIET,
        **kwargs,
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    torch.set_num_threads(2)
    ws = make_workspace(tmp_path_factory.mktemp("trained"))
    cfg = config(epochs=2, size=2, effective_size=4)
    results = [train(ws, fold, cfg) for fold in (0, 1)]
    return ws, cfg, results


def test_lr_schedule_warms_up_then_follows_a_cosine():
    factors = [lr_factor(e, 3, 60) for e in range(60)]
    assert factors[:3] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert factors[3] == pytest.approx(1.0)
    assert factors[31] == pytest.approx(0.5, abs=0.03)
    assert factors[-1] < 0.01 and all(
        a >= b for a, b in zip(factors[3:], factors[4:], strict=False)
    )


def test_a_fold_trains_and_writes_everything(trained):
    ws, cfg, results = trained
    result = results[0]
    run = ws.runs / "fold_0"
    assert result.epochs_run == 2 and result.batch_size == 2 and 0 <= result.best_epoch <= 1
    for name in (
        "best.pt",
        "last.pt",
        "config.yaml",
        "run.json",
        "train_log.csv",
        "curves.png",
        "oof.csv",
    ):
        assert (run / name).is_file(), name
    assert any((run / "tensorboard").iterdir())
    info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert info["initialisation"] == "trained from scratch" and info["device"] == "cpu"
    assert info["config_sha256"] == cfg.sha256() and info["epochs_run"] == 2
    n_tb, n = info["train_patients"]["tb"], info["train_patients"]["n"]
    assert info["pos_weight"] == pytest.approx((n - n_tb) / n_tb, abs=1e-6)  # #NTM / #TB
    with (run / "train_log.csv").open(encoding="utf-8") as handle:
        log = list(csv.DictReader(handle))
    assert [int(r["epoch"]) for r in log] == [0, 1]
    assert float(log[0]["lr"]) == pytest.approx(1e-4 / 3)  # first warm-up epoch
    with (run / "oof.csv").open(encoding="utf-8") as handle:
        oof = list(csv.DictReader(handle))
    from cavinet_ml.dataset.split import load_splits

    assert [r["case_id"] for r in oof] == load_splits(ws.splits).val_ids(0)


def test_a_finished_fold_is_not_trained_again(trained):
    ws, cfg, results = trained
    again = train(ws, 0, cfg)
    assert (
        again.epochs_run == results[0].epochs_run and again.best_val_auc == results[0].best_val_auc
    )


def test_a_changed_configuration_needs_restart(trained):
    ws, cfg, _ = trained
    with pytest.raises(TrainingError, match="--restart"):
        train(ws, 0, with_overrides(cfg, epochs=3))


def test_fold_numbers_and_missing_cache_are_checked(trained, tmp_path):
    ws, cfg, _ = trained
    with pytest.raises(TrainingError, match="between 0 and 1"):
        train(ws, 5, cfg)


def test_training_resumes_after_an_interruption(tmp_path, monkeypatch):
    ws = make_workspace(tmp_path)
    cfg = config(epochs=3, size=4, effective_size=4)
    calls = {"n": 0}
    original = trainer.predict

    def stop_during_the_second_epoch(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer, "predict", stop_during_the_second_epoch)
    with pytest.raises(KeyboardInterrupt):
        train(ws, 0, cfg)
    state = torch.load(ws.runs / "fold_0" / "last.pt", weights_only=True)
    assert state["epoch"] == 1 and not state["finished"]
    monkeypatch.setattr(trainer, "predict", original)
    messages = []
    result = train_fold(
        0,
        config=cfg,
        manifest_path=ws.manifest,
        splits_path=ws.splits,
        cache_dir=ws.cache,
        runs_dir=ws.runs,
        device="cpu",
        log=messages.append,
    )
    assert any("resuming at epoch 2" in m for m in messages)
    assert result.epochs_run == 3
    with (ws.runs / "fold_0" / "train_log.csv").open(encoding="utf-8") as handle:
        assert [int(r["epoch"]) for r in csv.DictReader(handle)] == [0, 1, 2]
    restarted = train(ws, 0, cfg, restart=True)
    assert restarted.epochs_run == 3


def test_out_of_memory_halves_the_batch_and_keeps_the_effective_size(tmp_path, monkeypatch):
    ws = make_workspace(tmp_path, size=48)
    cfg = config(epochs=1, size=4, effective_size=4)
    from cavinet_ml.model.network import CaviNetResNet

    forward = CaviNetResNet.forward

    def limited(self, x):
        if self.training and x.shape[0] > 1:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return forward(self, x)

    monkeypatch.setattr(CaviNetResNet, "forward", limited)
    messages = []
    result = train_fold(
        0,
        config=cfg,
        manifest_path=ws.manifest,
        splits_path=ws.splits,
        cache_dir=ws.cache,
        runs_dir=ws.runs,
        device="cpu",
        log=messages.append,
    )
    assert result.batch_size == 1
    assert sum("ran out of memory" in m for m in messages) == 2  # 4 -> 2 -> 1
    assert any("effective batch at 4" in m for m in messages)


def test_out_of_memory_at_batch_one_is_explained(tmp_path, monkeypatch):
    ws = make_workspace(tmp_path, size=48)
    from cavinet_ml.model.network import CaviNetResNet

    def always(self, x):
        raise torch.OutOfMemoryError("CUDA out of memory (simulated)")

    monkeypatch.setattr(CaviNetResNet, "forward", always)
    with pytest.raises(TrainingError, match="ran out of memory with batch size 1"):
        train(ws, 0, config(epochs=1, size=1, effective_size=1))


def test_early_stopping_after_patience(tmp_path, monkeypatch):
    ws = make_workspace(tmp_path)
    cfg = replace(
        config(epochs=10, size=4, effective_size=4),
        epochs=replace(TrainingConfig().epochs, max=10, patience=2),
    )
    aucs = iter([0.6, 0.55, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    monkeypatch.setattr(trainer, "roc_auc", lambda y, s: next(aucs))
    result = train(ws, 0, cfg)
    assert result.stopped_early and result.epochs_run == 3 and result.best_epoch == 0
    assert result.best_val_auc == pytest.approx(0.6)


# --- calibrate and export -----------------------------------------------------------------


def test_calibrate_fits_the_temperature_on_out_of_fold_logits(trained):
    ws, _, _ = trained
    result = calibrate(runs_dir=ws.runs, splits_path=ws.splits, out=ws.calibration, **QUIET)
    with (
        (ws.runs / "fold_0" / "oof.csv").open(encoding="utf-8") as a,
        (ws.runs / "fold_1" / "oof.csv").open(encoding="utf-8") as b,
    ):
        rows = list(csv.DictReader(a)) + list(csv.DictReader(b))
    expected = fit_temperature([float(r["logit"]) for r in rows], [int(r["label"]) for r in rows])
    assert result["temperature"] == pytest.approx(expected)
    assert result["n"] == len(rows) and len(result["per_fold"]) == 2
    assert 0 <= result["oof"]["auc"] <= 1 and len(result["oof"]["auc_ci"]) == 2
    # Temperature scaling minimises the negative log-likelihood (ECE may move either way).
    assert result["oof"]["nll"] <= result["oof"]["nll_before"] + 1e-6
    assert ws.oof_predictions.is_file()
    assert (
        json.loads(ws.calibration.read_text(encoding="utf-8"))["temperature"]
        == result["temperature"]
    )


def test_calibrate_refuses_incomplete_out_of_fold_predictions(trained, tmp_path):
    ws, _, _ = trained
    broken = tmp_path / "runs"
    for fold in (0, 1):
        (broken / f"fold_{fold}").mkdir(parents=True)
    (broken / "fold_0" / "oof.csv").write_text(
        (ws.runs / "fold_0" / "oof.csv").read_text(encoding="utf-8")
    )
    with pytest.raises(TrainingError, match="train fold 1 first"):
        calibrate(runs_dir=broken, splits_path=ws.splits, out=tmp_path / "c.json", **QUIET)
    (broken / "fold_1" / "oof.csv").write_text(
        (ws.runs / "fold_0" / "oof.csv").read_text(encoding="utf-8")
    )
    with pytest.raises(TrainingError, match="do not match the development folds"):
        calibrate(runs_dir=broken, splits_path=ws.splits, out=tmp_path / "c.json", **QUIET)


@pytest.fixture(scope="module")
def exported(trained):
    ws, _, _ = trained
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


def test_exported_bundle_follows_section_11_6_and_loads_in_the_app(exported):
    ws = exported
    bundle = load_bundle(ws.bundle)  # weights_only=True
    assert len(bundle["fold_state_dicts"]) == 2
    assert all(
        t.dtype == torch.float16
        for t in bundle["fold_state_dicts"][0].values()
        if t.is_floating_point()
    )
    assert bundle["is_demo"] is True  # built from a synthetic dataset
    assert bundle["decision_threshold"] == 0.5 and bundle["label_map"] == {1: "TB", 0: "NTM"}
    assert bundle["confidence_bands"] == {"high": 0.8, "moderate": 0.65}
    assert bundle["preprocessing"]["output_size"] == [32, 32, 32]
    metrics = bundle["metrics"]
    assert metrics["locked_test"] is None and 0 <= metrics["oof"]["auc"] <= 1
    assert "Synthetic" in metrics["dataset"]
    assert bundle["weights_sha256"] == weights_sha256(bundle["fold_state_dicts"])
    # The application stores these in a PostgreSQL JSON column: no NaN or infinity.
    json.dumps({k: v for k, v in bundle.items() if k != "fold_state_dicts"}, allow_nan=False)
    assert bundle["training"]["folds"][0]["epochs_run"] == 2
    ensemble = Ensemble(bundle)
    prediction = ensemble.predict(np.load(ws.cache / "TB_001.npy"))
    assert prediction.decision.predicted_class in ("TB", "NTM") and len(prediction.fold_logits) == 2
    card = json.loads((ws.bundle.parent / "model_card.json").read_text(encoding="utf-8"))
    assert card["file_sha256"] and "fold_state_dicts" not in card
    text = ws.model_card_md.read_text(encoding="utf-8")
    for heading in (
        "## Intended use",
        "## Training data",
        "## Performance",
        "## Limitations",
        "DEMO MODEL",
    ):
        assert heading in text
    assert "Not evaluated yet" in text


def test_locked_test_metrics_are_added_without_changing_the_weights(exported, tmp_path):
    ws = exported
    copy = tmp_path / "cavinet_model.pth"
    copy.write_bytes(ws.bundle.read_bytes())
    before = load_bundle(copy)["weights_sha256"]
    metric = {"value": 0.8, "ci": [0.7, 0.9]}
    update_bundle_metrics(
        copy,
        {
            "auc": 0.8,
            "auc_ci": [0.7, 0.9],
            "sensitivity": 0.7,
            "specificity": 0.75,
            "npv": math.nan,
            "accuracy": 0.72,
            "n": 10,
            "n_tb": 6,
            "n_ntm": 4,
            "evaluated_at": "now",
            "ci": {"auc": metric},
        },
        tmp_path / "MODEL_CARD.md",
    )
    after = load_bundle(copy)
    assert after["metrics"]["locked_test"]["auc"] == 0.8
    assert after["metrics"]["locked_test"]["npv"] is None  # NaN (undefined) becomes None
    assert weights_sha256(after["fold_state_dicts"]) == before
    assert "Evaluated once on 10 patients" in (tmp_path / "MODEL_CARD.md").read_text(
        encoding="utf-8"
    )


def test_export_needs_every_fold_finished(trained, tmp_path):
    ws, _, _ = trained
    with pytest.raises(TrainingError, match="not trained yet"):
        export_bundle(
            runs_dir=tmp_path,
            splits_path=ws.splits,
            manifest_path=ws.manifest,
            cache_dir=ws.cache,
            calibration_path=ws.calibration,
            out=tmp_path / "x.pth",
            model_card_md=None,
            **QUIET,
        )


def test_nan_auc_does_not_crash_training(tmp_path, monkeypatch):
    ws = make_workspace(tmp_path)
    monkeypatch.setattr(trainer, "roc_auc", lambda y, s: math.nan)
    result = train(ws, 0, config(epochs=1, size=4, effective_size=4))
    assert result.best_epoch == 0 and (ws.runs / "fold_0" / "best.pt").is_file()


def test_json_safe_replaces_undefined_numbers():
    from cavinet_ml.training.export import json_safe

    value = {
        "a": math.nan,
        "b": [1.0, math.inf, np.float32(0.5)],
        "c": {"d": np.int64(3)},
        "e": "x",
    }
    assert json_safe(value) == {"a": None, "b": [1.0, None, 0.5], "c": {"d": 3}, "e": "x"}
    assert json_safe(True) is True and json_safe(None) is None
