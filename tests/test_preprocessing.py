"""Section 11.1 steps 2 to 7, each on its own, then the whole pipeline."""

import math

import numpy as np
import pytest
import SimpleITK as sitk
from helpers import chest_volume, fake_segmenter, image_from

from cavinet_ml.config import DEFAULT_PREPROCESSING, PreprocessingConfig
from cavinet_ml.preprocessing import SegmenterUnavailable, preprocess, steps

CFG = DEFAULT_PREPROCESSING


# --- Step 2: clean ------------------------------------------------------------------------


def test_step2_values_below_minus_1024_are_clipped():
    volume = np.array([[[-3024.0, -2048.0, -1024.0, -1000.0, 0.0, 3000.0]]], dtype=np.float32)
    out = sitk.GetArrayFromImage(steps.clip_hu(image_from(volume), -1024))
    assert out.tolist() == [[[-1024.0, -1024.0, -1024.0, -1000.0, 0.0, 3000.0]]]


# --- Step 3: lung mask and fallback ----------------------------------------------------------


def lungs_mask(image, value=1):
    array = sitk.GetArrayFromImage(image)
    return (array < -800).astype(np.uint8) * value


def big_image():
    # 40 slices × 64² at 0.7 × 0.7 × 1.25 mm is tiny; use larger voxels so lungs exceed 0.5 L.
    return steps.clip_hu(image_from(chest_volume(depth=40), spacing=(5.0, 5.0, 5.0)), -1024)


def test_step3_a_plausible_lung_mask_is_used():
    image = big_image()
    result = steps.lung_mask(image, lungs_mask, min_volume_ml=500, body_threshold_hu=-500)
    assert not result.used_fallback and result.warning is None
    assert result.volume_ml >= 500
    assert result.mask.dtype == np.uint8 and set(np.unique(result.mask)) == {0, 1}


def test_step3_labels_for_left_and_right_lungs_count_as_lung():
    image = big_image()
    result = steps.lung_mask(
        image, lambda i: lungs_mask(i, 2), min_volume_ml=500, body_threshold_hu=-500
    )
    assert set(np.unique(result.mask)) == {0, 1}


@pytest.mark.parametrize(
    ("segmenter", "reason"),
    [
        (
            lambda image: np.zeros(tuple(reversed(image.GetSize())), np.uint8),
            "the lung segmentation found no lungs",
        ),
        (None, "the lung segmentation model is not available"),
    ],
)
def test_step3_falls_back_to_the_body_outline_with_a_warning(segmenter, reason):
    image = big_image()
    result = steps.lung_mask(image, segmenter, min_volume_ml=500, body_threshold_hu=-500)
    assert result.used_fallback
    assert result.warning == (
        f"Fallback crop: {reason}, so the scan was cropped to the body outline."
    )
    array = sitk.GetArrayFromImage(image)
    assert np.all(array[result.mask.astype(bool)] > -500)  # only body voxels


def test_step3_less_than_half_a_litre_falls_back():
    image = big_image()

    def tiny(img):
        mask = np.zeros(tuple(reversed(img.GetSize())), np.uint8)
        mask[20, 30:32, 20:22] = 1  # 4 voxels × 0.125 mL
        return mask

    result = steps.lung_mask(image, tiny, min_volume_ml=500, body_threshold_hu=-500)
    assert result.used_fallback and result.volume_ml == 0.5
    assert "found only 0 mL of lung" in result.warning or "found only 1 mL" in result.warning


def test_step3_unavailable_segmenter_falls_back():
    def unavailable(image):
        raise SegmenterUnavailable("no weights")

    result = steps.lung_mask(big_image(), unavailable, min_volume_ml=500, body_threshold_hu=-500)
    assert result.used_fallback and "not available" in result.warning


def test_step3_no_body_at_all_keeps_the_whole_scan():
    image = image_from(np.full((5, 8, 8), -1024.0, np.float32))
    result = steps.lung_mask(image, None, min_volume_ml=500, body_threshold_hu=-500)
    assert result.mask.all() and "whole scan" in result.warning


def test_step3_mask_must_match_the_image():
    with pytest.raises(ValueError, match="does not match"):
        steps.lung_mask(
            big_image(), lambda image: np.ones((2, 2, 2)), min_volume_ml=1, body_threshold_hu=-500
        )


def test_body_mask_keeps_only_the_largest_region():
    volume = np.full((3, 10, 10), -1000.0, np.float32)
    volume[:, 2:8, 2:8] = 40
    volume[0, 0, 0] = 40  # a speck outside the body
    mask = steps.body_mask(image_from(volume), -500)
    assert mask.sum() == 3 * 36 and mask[0, 0, 0] == 0


@pytest.mark.parametrize(
    ("spacing", "gap", "step"),
    [(1.0, 3.0, 3), (1.25, 3.0, 2), (2.5, 3.0, 1), (5.0, 3.0, 1), (0.5, 3.0, 6)],
)
def test_step3_thin_slices_are_segmented_at_most_3_mm_apart(spacing, gap, step):
    assert steps.slice_step(spacing, gap) == step


def test_step3_skipped_slices_copy_the_nearest_segmented_slice():
    image = image_from(np.zeros((10, 4, 4), np.float32), spacing=(1, 1, 1))
    seen = []

    def segmenter(img):
        seen.append(img.GetDepth())
        mask = np.zeros(tuple(reversed(img.GetSize())), np.uint8)
        mask[:, 0, 0] = np.arange(img.GetDepth()) + 1  # label = sampled slice number
        return mask

    mask = steps.segment_sampled(image, segmenter, 3)
    assert seen == [4]  # slices 0, 3, 6, 9
    assert list(mask[:, 0, 0]) == [1, 1, 2, 2, 2, 3, 3, 3, 4, 4]


# --- Step 4: resample ------------------------------------------------------------------------


def test_step4_resampling_gives_1_5_mm_voxels_and_keeps_geometry():
    image = image_from(chest_volume(depth=40), spacing=(0.7, 0.7, 1.25))
    image.SetOrigin((10.0, -20.0, 5.0))
    out = steps.resample(image, 1.5, interpolator=sitk.sitkLinear, default_value=-1024)
    assert out.GetSpacing() == (1.5, 1.5, 1.5)
    assert out.GetSize() == (round(64 * 0.7 / 1.5), round(64 * 0.7 / 1.5), round(40 * 1.25 / 1.5))
    assert out.GetOrigin() == (10.0, -20.0, 5.0)
    assert out.GetDirection() == image.GetDirection()


def test_step4_the_mask_is_resampled_with_nearest_neighbour():
    image = image_from(chest_volume(depth=20))
    mask = steps.mask_image(lungs_mask(image), image)
    out = steps.resample(mask, 1.5, interpolator=sitk.sitkNearestNeighbor, default_value=0)
    assert set(np.unique(sitk.GetArrayFromImage(out))) <= {0, 1}
    assert out.GetPixelID() == sitk.sitkUInt8


# --- Step 5: crop ----------------------------------------------------------------------------


def test_step5_crop_is_the_lung_box_plus_10_mm():
    mask = np.zeros((60, 60, 60), np.uint8)
    mask[20:30, 25:35, 5:50] = 1
    box = steps.crop_box(mask, spacing_mm=1.5, margin_mm=10.0)
    margin = math.ceil(10 / 1.5)
    assert margin == 7
    assert box == (slice(13, 37), slice(18, 42), slice(0, 57))  # clamped at 0


def test_step5_empty_mask_keeps_everything():
    assert steps.crop_box(np.zeros((4, 5, 6), np.uint8), 1.5, 10) == (
        slice(0, 4),
        slice(0, 5),
        slice(0, 6),
    )


# --- Steps 6 and 7: window and resize ---------------------------------------------------------


def test_step6_lung_window_clips_to_minus_1350_150_and_scales_to_0_1():
    values = np.array([-3000, -1350, -600, 150, 2000], dtype=np.float32)
    out = steps.lung_window(values, -1350, 150)
    assert out.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])
    assert out.min() >= 0 and out.max() <= 1


def test_step7_resize_to_128_cubed_float16():
    out = steps.resize(np.random.default_rng(0).random((37, 51, 66)), (128, 128, 128))
    assert out.shape == (128, 128, 128) and out.dtype == np.float16


def test_step7_resize_is_trilinear():
    ramp = np.linspace(0, 1, 10, dtype=np.float32)[:, None, None] * np.ones((10, 4, 4))
    out = steps.resize(ramp, (20, 4, 4), "float32")
    assert np.all(np.diff(out[:, 0, 0]) >= 0)  # smooth, monotone interpolation
    assert 0 < out[1, 0, 0] < out[2, 0, 0]


# --- The whole pipeline -------------------------------------------------------------------------


def test_pipeline_produces_the_model_input():
    image = image_from(chest_volume(depth=60, size=96), spacing=(3.0, 3.0, 3.0))
    result = preprocess(image, fake_segmenter(), CFG)
    assert result.volume.shape == (128, 128, 128)
    assert result.volume.dtype == np.float16
    assert float(result.volume.min()) >= 0.0 and float(result.volume.max()) <= 1.0
    assert result.resampled_size == (192, 192, 120)  # 96 × 3 / 1.5, 60 × 3 / 1.5
    assert not result.lung.used_fallback and result.warnings == []
    assert set(result.seconds) == {"clean", "lung_mask", "resample", "crop_window_resize"}
    # The crop is tighter than the volume: the lungs plus 7 voxels (10 mm) on each side.
    (z0, z1), (y0, y1), (x0, x1) = result.crop
    assert 0 < y0 < y1 < 192 and 0 < x0 < x1 < 192
    assert float(sitk.GetArrayFromImage(result.hu_image).min()) == -1024.0  # step 2 applied


def test_pipeline_records_the_fallback_warning():
    image = image_from(chest_volume(depth=30, size=64, lungs=False), spacing=(4.0, 4.0, 4.0))
    result = preprocess(image, fake_segmenter(), CFG)
    assert result.lung.used_fallback
    assert result.warnings and result.warnings[0].startswith("Fallback crop:")


def test_config_round_trips_through_plain_values():
    values = DEFAULT_PREPROCESSING.to_dict()
    assert values["window_hu"] == [-1350.0, 150.0] and values["output_size"] == [128, 128, 128]
    assert values["target_spacing_mm"] == 1.5 and values["crop_margin_mm"] == 10.0
    assert PreprocessingConfig.from_dict(values) == DEFAULT_PREPROCESSING
