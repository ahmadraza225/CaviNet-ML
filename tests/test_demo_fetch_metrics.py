"""Demo model (FR-05.6), make fetch-model, metrics, the lungmask wrapper and the CLI."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from helpers import chest_volume, image_from

from cavinet_ml import fetch as fetch_module
from cavinet_ml.cli import main
from cavinet_ml.demo.synthetic import synthetic_volume
from cavinet_ml.evaluation import brier_score, roc_auc, threshold_metrics
from cavinet_ml.fetch import fetch_model
from cavinet_ml.model.bundle import file_sha256, load_bundle
from cavinet_ml.preprocessing import SegmenterUnavailable
from cavinet_ml.preprocessing.lungmask_segmenter import R231_FILENAME, LungmaskSegmenter


def test_demo_bundle_is_flagged_and_loads_with_weights_only(tiny_bundle_path):
    raw = torch.load(tiny_bundle_path, map_location="cpu", weights_only=True)
    assert raw["is_demo"] is True
    assert raw["model_name"] == "CaviNet demo model (synthetic data)"
    assert len(raw["fold_state_dicts"]) == 5
    assert raw["architecture"]["layers"] == [2, 2, 2, 2]
    assert raw["preprocessing"]["window_hu"] == [-1350.0, 150.0]
    metrics = raw["metrics"]
    assert "Synthetic" in metrics["dataset"]
    assert set(metrics["locked_test"]) >= {"auc", "sensitivity", "specificity", "n"}
    assert raw["temperature"] > 0


def test_synthetic_volumes_are_in_the_model_input_space():
    volume = synthetic_volume(1, np.random.default_rng(0), size=32)
    assert volume.shape == (32, 32, 32) and volume.dtype == np.float16
    assert float(volume.min()) >= 0 and float(volume.max()) <= 1


def fast_demo(out, log=print):
    from cavinet_ml.demo.build import build_demo_bundle

    return build_demo_bundle(out, dev_cases=10, test_cases=6, epochs=1, size=16, log=log)


def test_fetch_builds_the_demo_model_when_no_url_is_set(tmp_path):
    out = tmp_path / "models" / "cavinet_model.pth"
    assert fetch_model(out, build_demo=fast_demo, log=lambda m: None) == "demo"
    assert load_bundle(out)["is_demo"] is True
    card = json.loads((out.parent / "model_card.json").read_text())
    assert card["is_demo"] is True and card["file_sha256"] == file_sha256(out)
    assert "fold_state_dicts" not in card
    assert fetch_model(out, build_demo=fast_demo, log=lambda m: None) == "kept"


def test_fetch_downloads_the_release_asset(tmp_path, tiny_bundle_path):
    out = tmp_path / "cavinet_model.pth"
    url = Path(tiny_bundle_path).as_uri()  # urllib serves file:// like https://
    result = fetch_model(
        out, url=url, sha256=file_sha256(tiny_bundle_path), build_demo=fast_demo, log=lambda m: None
    )
    assert result == "downloaded"
    assert file_sha256(out) == file_sha256(tiny_bundle_path)
    assert not out.with_suffix(".download").exists()


@pytest.mark.parametrize("problem", ["bad checksum", "not a bundle", "unreachable"])
def test_a_failed_download_falls_back_to_the_demo_model(tmp_path, tiny_bundle_path, problem):
    junk = tmp_path / "junk.pth"
    junk.write_bytes(b"not a model")
    url, sha = {
        "bad checksum": (Path(tiny_bundle_path).as_uri(), "0" * 64),
        "not a bundle": (junk.as_uri(), None),
        "unreachable": ((tmp_path / "missing.pth").as_uri(), None),
    }[problem]
    messages = []
    out = tmp_path / "models" / "cavinet_model.pth"
    assert (
        fetch_model(out, url=url, sha256=sha, build_demo=fast_demo, log=messages.append) == "demo"
    )
    assert any("Download failed" in m for m in messages)
    assert not list(out.parent.glob("*.download"))


def test_an_invalid_installed_model_is_replaced_and_force_rebuilds(tmp_path):
    out = tmp_path / "cavinet_model.pth"
    out.write_bytes(b"broken")
    assert fetch_model(out, build_demo=fast_demo, log=lambda m: None) == "demo"
    assert fetch_model(out, force=True, build_demo=fast_demo, log=lambda m: None) == "demo"


def test_cli_fetch_model_and_build_demo(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fetch_module, "fetch_model", lambda out, **kw: calls.append((out, kw)))
    assert main(["fetch-model", "--out", str(tmp_path / "m.pth"), "--url", "https://x/y"]) == 0
    assert calls[0][0] == str(tmp_path / "m.pth") and calls[0][1]["url"] == "https://x/y"

    import cavinet_ml.demo.build as build_module

    original = build_module.build_demo_bundle
    monkeypatch.setattr(
        build_module,
        "build_demo_bundle",
        lambda out, seed: original(
            out, dev_cases=10, test_cases=6, epochs=1, size=16, log=lambda m: None
        ),
    )
    assert main(["build-demo", "--out", str(tmp_path / "demo.pth")]) == 0
    assert (tmp_path / "model_card.json").exists()


def test_metrics():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5  # ties count half
    assert np.isnan(roc_auc([1, 1], [0.2, 0.3]))
    at = threshold_metrics([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1])
    assert (at["sensitivity"], at["specificity"], at["accuracy"]) == (0.5, 0.5, 0.5)
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0


def test_lungmask_segmenter_without_weights_is_unavailable(tmp_path):
    segmenter = LungmaskSegmenter(tmp_path / R231_FILENAME)
    assert not segmenter.available
    with pytest.raises(SegmenterUnavailable):
        segmenter(image_from(chest_volume(depth=4)))
    assert not LungmaskSegmenter(None).available


LUNGMASK_WEIGHTS = os.environ.get("LUNGMASK_WEIGHTS")


@pytest.mark.skipif(
    not (LUNGMASK_WEIGHTS and Path(LUNGMASK_WEIGHTS).is_file()),
    reason="LUNGMASK_WEIGHTS not available (the Docker image and CI full-stack job have them)",
)
def test_real_lungmask_finds_the_lungs_of_a_synthetic_chest():
    volume = chest_volume(depth=24, size=128)
    image = image_from(volume, spacing=(2.8, 2.8, 8.0))
    mask = LungmaskSegmenter(LUNGMASK_WEIGHTS)(image)
    assert mask.shape == volume.shape
    lung_ml = float((mask > 0).sum()) * 2.8 * 2.8 * 8.0 / 1000
    assert lung_ml > 500
