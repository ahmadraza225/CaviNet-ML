"""Ensemble inference (section 11.5), previews (FR-05.3) and the analysis entry point."""

import numpy as np
import pytest
import torch
from helpers import chest_volume, fake_segmenter, write_series
from PIL import Image

from cavinet_ml.analysis import Analyser, PipelineError
from cavinet_ml.inference import Ensemble
from cavinet_ml.inference.calibration import calibrated_probability
from cavinet_ml.model.bundle import load_bundle
from cavinet_ml.previews import preview_indices


def fixed_logit_bundle(tiny_bundle_path, biases, temperature=2.0):
    """Fold models whose output is exactly their final-layer bias."""
    bundle = load_bundle(tiny_bundle_path)
    bundle["fold_state_dicts"] = [
        {
            **state,
            "fc.weight": torch.zeros_like(state["fc.weight"]),
            "fc.bias": torch.tensor([b], dtype=state["fc.bias"].dtype),
        }
        for state, b in zip(bundle["fold_state_dicts"], biases, strict=False)
    ]
    bundle["temperature"] = temperature
    return bundle


def test_ensemble_averages_fold_logits_then_applies_the_temperature(tiny_bundle_path):
    biases = [1.0, 2.0, 3.0, 4.0, 6.0]
    ensemble = Ensemble(fixed_logit_bundle(tiny_bundle_path, biases, temperature=2.0))
    prediction = ensemble.predict(np.zeros((32, 32, 32), np.float16))
    assert prediction.fold_logits == pytest.approx(biases, abs=1e-2)
    assert prediction.logit == pytest.approx(3.2, abs=1e-2)
    expected = calibrated_probability(3.2, 2.0)
    assert prediction.decision.probability_tb == pytest.approx(expected, abs=1e-3)
    assert prediction.decision.predicted_class == "TB"


def test_ensemble_reads_its_settings_from_the_bundle(tiny_bundle_path):
    ensemble = Ensemble(load_bundle(tiny_bundle_path))
    assert ensemble.is_demo is True
    assert len(ensemble.models) == 5
    assert ensemble.preprocessing.output_size == (128, 128, 128)
    assert ensemble.label_map == {1: "TB", 0: "NTM"}


def test_preview_indices_cover_the_lungs_head_to_feet():
    mask = np.zeros((200, 4, 4), np.uint8)
    mask[40:160] = 1
    indices = preview_indices(mask, 48)
    assert len(indices) == 48
    assert indices[0] == 159 and indices[-1] == 40
    assert indices == sorted(indices, reverse=True)
    short = np.zeros((60, 4, 4), np.uint8)
    short[10:20] = 1  # fewer lung slices than previews: use the whole scan
    assert preview_indices(short, 48)[0] == 59 and preview_indices(short, 48)[-1] == 0


def test_analysis_end_to_end(tmp_path, tiny_bundle_path):
    write_series(tmp_path / "dicom", chest_volume(depth=60, size=64), spacing=(5.0, 5.0, 4.0))
    analyser = Analyser(Ensemble(load_bundle(tiny_bundle_path)), fake_segmenter())
    prepared = analyser.prepare(tmp_path / "dicom", tmp_path / "previews")
    result = analyser.infer(prepared)

    assert result.decision["predicted_class"] in {"TB", "NTM"}
    assert 0 <= result.decision["probability_tb"] <= 1
    assert result.decision["band"] in {"High", "Moderate", "Low"}
    assert len(result.fold_logits) == 5
    assert result.model["is_demo"] is True and result.model["folds"] == 5
    assert set(result.seconds) >= {"load", "lung_mask", "previews", "inference", "total"}
    assert result.seconds["total"] == pytest.approx(
        sum(v for k, v in result.seconds.items() if k != "total"), abs=0.01
    )
    # FR-05.3: 48 previews and 3 representative slices, lung-windowed PNGs.
    assert result.previews.count == 48
    files = sorted(p.name for p in (tmp_path / "previews").iterdir())
    assert len(files) == 51 and files[0] == "preview-00.png"
    image = Image.open(tmp_path / "previews" / "preview-24.png")
    assert image.mode == "L" and max(image.size) == 512
    assert result.previews.representative == [
        "representative-1.png",
        "representative-2.png",
        "representative-3.png",
    ]


def test_unreadable_scan_is_a_pipeline_error(tmp_path, tiny_bundle_path):
    (tmp_path / "dicom").mkdir()
    analyser = Analyser(Ensemble(load_bundle(tiny_bundle_path)), None)
    with pytest.raises(PipelineError, match="No DICOM series"):
        analyser.prepare(tmp_path / "dicom", tmp_path / "previews")
