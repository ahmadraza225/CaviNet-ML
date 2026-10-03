"""Section 11.1, steps 2 to 7. Each step is a small function so it can be tested on its own;
`pipeline.preprocess` chains them. Arrays are indexed [z, y, x] (SimpleITK order)."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

# A lung segmenter takes the HU image and returns a mask array (0 = background).
Segmenter = Callable[[sitk.Image], np.ndarray]


class SegmenterUnavailable(Exception):
    """The lung segmentation model cannot be used (e.g. its weights are missing)."""


# --- Step 2: clean -------------------------------------------------------------------------


def clip_hu(image: sitk.Image, floor: float) -> sitk.Image:
    """Values below `floor` (scanner padding such as -3024 or -2048) become `floor`."""
    return sitk.Threshold(
        image, lower=floor, upper=float(np.finfo(np.float32).max), outsideValue=floor
    )


# --- Step 3: lung mask, with the documented fallback --------------------------------------


@dataclass
class LungMask:
    mask: np.ndarray  # uint8, same shape as the image array; 1 = inside the crop region
    volume_ml: float  # volume found by the lung segmenter (0 if it was not usable)
    used_fallback: bool
    warning: str | None = None


def voxel_volume_ml(image: sitk.Image) -> float:
    sx, sy, sz = image.GetSpacing()
    return sx * sy * sz / 1000.0


def body_mask(image: sitk.Image, threshold_hu: float) -> np.ndarray:
    """The largest connected region above `threshold_hu`: the patient's body."""
    above = sitk.GetArrayViewFromImage(image) > threshold_hu
    labels, count = ndimage.label(above)
    if count == 0:
        return np.zeros(above.shape, dtype=np.uint8)
    sizes = ndimage.sum_labels(above, labels, index=np.arange(1, count + 1))
    return (labels == (int(np.argmax(sizes)) + 1)).astype(np.uint8)


def slice_step(spacing_z_mm: float, max_gap_mm: float) -> int:
    """Segment every Nth slice so that the gap between segmented slices is at most
    `max_gap_mm` (every slice when they are already that far apart)."""
    return max(1, int(math.floor(max_gap_mm / spacing_z_mm + 1e-6)))


def segment_sampled(image: sitk.Image, segmenter: Segmenter, step: int) -> np.ndarray:
    """Run the segmenter on every `step`-th slice; each skipped slice copies the mask of
    the nearest segmented slice."""
    if step == 1:
        return np.asarray(segmenter(image))
    sampled = np.asarray(segmenter(image[:, :, ::step]))
    depth = image.GetDepth()
    nearest = np.minimum(np.round(np.arange(depth) / step).astype(int), sampled.shape[0] - 1)
    return sampled[nearest]


def lung_mask(
    image: sitk.Image,
    segmenter: Segmenter | None,
    *,
    min_volume_ml: float,
    body_threshold_hu: float,
    max_slice_gap_mm: float = 0.0,
) -> LungMask:
    """Lungs from the segmenter (lungmask R231). If it finds nothing or less than
    `min_volume_ml`, or cannot run, crop to the body outline instead and say so."""
    found_ml = 0.0
    reason = "the lung segmentation model is not available"
    if segmenter is not None:
        step = slice_step(image.GetSpacing()[2], max_slice_gap_mm) if max_slice_gap_mm else 1
        try:
            lungs = (segment_sampled(image, segmenter, step) > 0).astype(np.uint8)
        except SegmenterUnavailable:
            lungs = None
        if lungs is not None:
            if lungs.shape != tuple(reversed(image.GetSize())):
                raise ValueError("the lung mask does not match the image size")
            found_ml = float(lungs.sum()) * voxel_volume_ml(image)
            if found_ml >= min_volume_ml:
                return LungMask(lungs, round(found_ml, 1), used_fallback=False)
            reason = (
                "the lung segmentation found no lungs"
                if found_ml == 0
                else f"the lung segmentation found only {found_ml:.0f} mL of lung"
            )

    body = body_mask(image, body_threshold_hu)
    if body.any():
        warning = f"Fallback crop: {reason}, so the scan was cropped to the body outline."
    else:
        body = np.ones(body.shape, dtype=np.uint8)
        warning = f"Fallback crop: {reason} and no body outline was found; the whole scan was used."
    return LungMask(body, round(found_ml, 1), used_fallback=True, warning=warning)


# --- Step 4: resample -----------------------------------------------------------------------


def resample(
    image: sitk.Image, spacing_mm: float, *, interpolator: int, default_value: float
) -> sitk.Image:
    """Resample to isotropic `spacing_mm`, keeping origin, direction and physical extent."""
    old_spacing = image.GetSpacing()
    old_size = image.GetSize()
    new_size = [
        max(1, int(round(size * spacing / spacing_mm)))
        for size, spacing in zip(old_size, old_spacing, strict=True)
    ]
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        image.GetOrigin(),
        (spacing_mm,) * 3,
        image.GetDirection(),
        default_value,
        image.GetPixelID(),
    )


def mask_image(mask: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """A mask array as an image with the reference image's geometry."""
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.CopyInformation(reference)
    return image


# --- Step 5: crop ---------------------------------------------------------------------------


def crop_box(mask: np.ndarray, spacing_mm: float, margin_mm: float) -> tuple[slice, slice, slice]:
    """Bounding box of the mask plus `margin_mm` on every side, clamped to the volume.
    An empty mask keeps the whole volume."""
    if not mask.any():
        return tuple(slice(0, n) for n in mask.shape)  # type: ignore[return-value]
    margin = math.ceil(margin_mm / spacing_mm)
    box = []
    for axis, size in enumerate(mask.shape):
        other = tuple(a for a in range(mask.ndim) if a != axis)
        present = np.flatnonzero(mask.any(axis=other))
        box.append(
            slice(max(0, int(present[0]) - margin), min(size, int(present[-1]) + 1 + margin))
        )
    return tuple(box)  # type: ignore[return-value]


# --- Steps 6 and 7: window and resize -------------------------------------------------------


def lung_window(array: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clip to [low, high] HU and scale to [0, 1]."""
    clipped = np.clip(array.astype(np.float32), low, high)
    return (clipped - low) / (high - low)


def resize(array: np.ndarray, size: tuple[int, int, int], dtype: str = "float16") -> np.ndarray:
    """Trilinear resize to `size` voxels, stored as `dtype`."""
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None, None]
    resized = torch.nn.functional.interpolate(
        tensor, size=tuple(size), mode="trilinear", align_corners=False
    )
    return resized[0, 0].numpy().astype(dtype)
