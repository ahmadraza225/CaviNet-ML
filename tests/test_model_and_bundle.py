"""Section 11.2 network and section 11.6 bundle."""

import math
import pickle
from collections import OrderedDict

import numpy as np
import pytest
import torch

from cavinet_ml.demo.build import DEMO_ARCHITECTURE
from cavinet_ml.model import bundle as bundle_module
from cavinet_ml.model.bundle import BundleError, load_bundle, metadata, save_bundle
from cavinet_ml.model.network import DEFAULT_ARCHITECTURE, build_model, count_parameters


def test_network_is_a_3d_resnet18_with_one_input_channel_and_one_logit():
    model = build_model()
    assert DEFAULT_ARCHITECTURE["layers"] == [2, 2, 2, 2]  # ResNet-18
    assert model.backbone.conv1.in_channels == 1
    assert isinstance(model.backbone.conv1, torch.nn.Conv3d)
    assert model.fc.out_features == 1
    assert model.dropout.p == 0.3  # dropout before the final layer
    model.eval()
    with torch.inference_mode():
        out = build_model(DEMO_ARCHITECTURE).eval()(torch.zeros(2, 1, 32, 32, 32))
    assert out.shape == (2,)


def test_full_size_ensemble_weights_are_about_330_mb_in_float16():
    per_fold = count_parameters(build_model())
    total_mb = per_fold * 2 * 5 / 1e6
    assert 300 < total_mb < 360


def make_bundle(folds=2, architecture=DEMO_ARCHITECTURE, **overrides):
    torch.manual_seed(0)
    states = [build_model(architecture).state_dict() for _ in range(folds)]
    bundle = {
        "format_version": 1,
        "model_name": "test",
        "created_at": "2026-10-03T00:00:00+00:00",
        "git_commit": "abc123",
        "is_demo": True,
        "architecture": dict(architecture),
        "fold_state_dicts": states,
        "temperature": 1.5,
        "decision_threshold": 0.5,
        "confidence_bands": {"high": 0.8, "moderate": 0.65},
        "preprocessing": {"output_size": [128, 128, 128], "window_hu": [-1350.0, 150.0]},
        "label_map": {1: "TB", 0: "NTM"},
        "metrics": {"locked_test": {"auc": 0.5}},
        "data_manifest_sha256": "0" * 64,
    }
    bundle.update(overrides)
    return bundle


def test_bundle_round_trip_with_weights_only(tmp_path):
    original = make_bundle()
    path = save_bundle(original, tmp_path / "cavinet_model.pth")
    raw = torch.load(path, map_location="cpu", weights_only=True)  # no arbitrary code
    assert raw["is_demo"] is True and raw["label_map"] == {1: "TB", 0: "NTM"}
    assert raw["temperature"] == 1.5 and raw["decision_threshold"] == 0.5
    assert len(raw["fold_state_dicts"]) == 2
    weights = raw["fold_state_dicts"][0]
    assert all(t.dtype == torch.float16 for t in weights.values() if t.is_floating_point())
    assert any(t.dtype == torch.int64 for t in weights.values())  # BN counters kept as is
    loaded = load_bundle(path)
    assert set(bundle_module.REQUIRED_KEYS) <= set(loaded)


def test_loaded_weights_reproduce_the_model(tmp_path):
    original = make_bundle(folds=1)
    model = build_model(DEMO_ARCHITECTURE).eval()
    model.load_state_dict(original["fold_state_dicts"][0])
    path = save_bundle(original, tmp_path / "m.pth")
    restored = build_model(DEMO_ARCHITECTURE).eval()
    restored.load_state_dict(
        {
            k: v.float() if v.is_floating_point() else v
            for k, v in load_bundle(path)["fold_state_dicts"][0].items()
        }
    )
    x = torch.rand(1, 1, 32, 32, 32)
    with torch.inference_mode():
        assert math.isclose(float(model(x)), float(restored(x)), abs_tol=1e-2)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"format_version": 2}, "unsupported format_version"),
        ({"is_demo": "yes"}, "is_demo must be true or false"),
        ({"temperature": np.float32(1.0)}, "temperature: float32 is not a plain value"),
        ({"metrics": {"auc": object()}}, "metrics.auc: object is not a plain value"),
        ({"fold_state_dicts": []}, "non-empty list"),
    ],
)
def test_invalid_bundles_are_refused(tmp_path, change, message):
    with pytest.raises(BundleError, match=message):
        save_bundle(make_bundle(**change), tmp_path / "bad.pth")


def test_missing_keys_are_refused(tmp_path):
    bundle = make_bundle()
    del bundle["metrics"]
    with pytest.raises(BundleError, match="missing keys: metrics"):
        save_bundle(bundle, tmp_path / "bad.pth")


def test_files_that_are_not_bundles_are_refused(tmp_path):
    (tmp_path / "junk.pth").write_bytes(b"not a model")
    with pytest.raises(BundleError, match="cannot read the model file"):
        load_bundle(tmp_path / "junk.pth")
    torch.save([1, 2, 3], tmp_path / "list.pth")
    with pytest.raises(BundleError, match="does not hold a bundle"):
        load_bundle(tmp_path / "list.pth")


@pytest.mark.filterwarnings("ignore:Detected pickle protocol")
def test_code_inside_a_file_never_runs(tmp_path):
    class Evil:
        def __reduce__(self):
            return (print, ("this must never run",))

    with (tmp_path / "evil.pth").open("wb") as handle:
        pickle.dump({"x": Evil()}, handle)
    with pytest.raises(BundleError):
        load_bundle(tmp_path / "evil.pth")


def test_metadata_has_no_weights_and_counts_parameters(tmp_path):
    path = save_bundle(make_bundle(folds=3), tmp_path / "m.pth")
    info = metadata(load_bundle(path, mmap=True))
    assert "fold_state_dicts" not in info
    assert info["folds"] == 3
    assert info["parameters_per_fold"] == count_parameters(build_model(DEMO_ARCHITECTURE))


def test_state_dicts_are_ordered_dicts_after_saving(tmp_path):
    path = save_bundle(make_bundle(folds=1), tmp_path / "m.pth")
    assert isinstance(load_bundle(path)["fold_state_dicts"][0], OrderedDict)
