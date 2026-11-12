"""M-11 heatmap explanation: FR-11.1 (Grad-CAM, averaged over folds, mapped onto the previews),
FR-11.2 (overlays, top-3 slices), FR-11.3 (label) and the FR-11.4 lung share."""

import numpy as np
import pytest
import torch
from helpers import chest_volume, fake_segmenter, write_series
from PIL import Image

from cavinet_ml.analysis import Analyser
from cavinet_ml.explain import LABEL, TOP_SLICES, colour, grad_cam, lung_share
from cavinet_ml.inference import Ensemble
from cavinet_ml.model.bundle import load_bundle
from cavinet_ml.model.network import build_model


def small_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = build_model({"block_inplanes": [4, 8, 8, 8]})
    model.eval()
    return model


def test_grad_cam_is_a_fold_average_in_the_input_shape():
    volume = np.random.default_rng(0).random((32, 32, 32)).astype(np.float16)
    models = [small_model(1), small_model(2)]
    cam = grad_cam(models, volume, 1.0)
    assert cam.shape == volume.shape and cam.dtype == np.float32
    assert 0.0 <= float(cam.min()) and float(cam.max()) == pytest.approx(1.0)
    single = [grad_cam([m], volume, 1.0) for m in models]
    mean = (single[0] + single[1]) / 2
    assert np.allclose(cam, mean / mean.max(), atol=1e-5)
    # Explaining the other class gives a different map; no gradients are left behind.
    assert not np.allclose(grad_cam(models, volume, -1.0), cam)
    assert all(p.grad is None for m in models for p in m.parameters())


def test_lung_share_is_the_energy_inside_the_mask():
    cam = np.zeros((4, 4, 4), dtype=np.float32)
    cam[0] = 1.0  # 16 voxels
    cam[1, 0, 0] = 4.0
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[0] = 1
    assert lung_share(cam, mask) == pytest.approx(16 / 20)
    assert lung_share(np.zeros_like(cam), mask) is None
    assert lung_share(cam, None) is None


def test_overlay_colours_are_transparent_below_the_floor():
    rgba = colour(np.array([0.0, 0.3, 0.7, 1.0]))
    assert rgba[0, 3] == 0 and rgba[1, 3] == 0  # weak values stay invisible
    assert 0 < rgba[2, 3] < rgba[3, 3]  # stronger is more opaque
    assert rgba[3, 0] == 255 and rgba[3, 2] == 0  # the strongest is red


def test_explain_writes_an_overlay_per_preview_and_the_top_slices(tmp_path, tiny_bundle_path):
    write_series(tmp_path / "dicom", chest_volume(depth=60, size=64), spacing=(5.0, 5.0, 4.0))
    analyser = Analyser(Ensemble(load_bundle(tiny_bundle_path)), fake_segmenter())
    prepared = analyser.prepare(tmp_path / "dicom", tmp_path / "previews")
    result = analyser.infer(prepared)
    heatmaps = analyser.explain(prepared, result, tmp_path / "previews")

    assert heatmaps.target == result.decision["predicted_class"]
    assert heatmaps.files == [f"heatmap-{n:02d}.png" for n in range(48)]
    preview = Image.open(tmp_path / "previews" / "preview-24.png")
    layer = Image.open(tmp_path / "previews" / "heatmap-24.png")
    assert layer.mode == "RGBA" and layer.size == preview.size  # lines up with the preview
    assert len(heatmaps.top) == TOP_SLICES and len(set(heatmaps.top)) == TOP_SLICES
    shares = heatmaps.slice_share
    assert len(shares) == 48 and sum(shares) == pytest.approx(1.0)
    assert min(shares[i] for i in heatmaps.top) >= max(
        s for i, s in enumerate(shares) if i not in heatmaps.top
    )
    assert heatmaps.top_files == ["heatmap-top-1.png", "heatmap-top-2.png", "heatmap-top-3.png"]
    for name in heatmaps.top_files:
        assert Image.open(tmp_path / "previews" / name).mode == "RGB"
    assert 0.0 <= heatmaps.lung_share <= 1.0
    assert prepared.preprocessed.model_mask.shape == prepared.preprocessed.volume.shape
    assert heatmaps.seconds["heatmap"] > 0
    summary = heatmaps.to_dict()
    assert (
        summary["label"]
        == LABEL
        == ("Regions that most influenced the prediction. This is not a lesion marker.")
    )
    assert summary["top"] == heatmaps.top
