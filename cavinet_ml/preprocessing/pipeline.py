"""Section 11.1 end to end. The training toolkit and the application call `preprocess` with
the parameters stored in the model bundle, so both always prepare scans identically."""

import time
from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk

from cavinet_ml.config import DEFAULT_PREPROCESSING, PreprocessingConfig
from cavinet_ml.preprocessing import steps
from cavinet_ml.preprocessing.steps import LungMask, Segmenter


@dataclass
class Preprocessed:
    volume: np.ndarray  # model input, output_size voxels, float16, values in [0, 1]
    hu_image: sitk.Image  # step 2 output (HU, original resolution), for previews
    lung: LungMask  # step 3 output (original resolution)
    resampled_size: tuple[int, int, int]  # (x, y, z) voxels at target spacing
    crop: tuple[tuple[int, int], ...]  # [z, y, x] (start, stop) in the resampled volume
    seconds: dict[str, float] = field(default_factory=dict)

    @property
    def warnings(self) -> list[str]:
        return [self.lung.warning] if self.lung.warning else []


def preprocess(
    image: sitk.Image,
    segmenter: Segmenter | None,
    config: PreprocessingConfig = DEFAULT_PREPROCESSING,
) -> Preprocessed:
    """Steps 2 to 7 on a loaded HU image (step 1: cavinet_ml.io.load_series)."""
    seconds: dict[str, float] = {}

    started = time.perf_counter()
    hu = steps.clip_hu(image, config.hu_floor)
    seconds["clean"] = time.perf_counter() - started

    started = time.perf_counter()
    lung = steps.lung_mask(
        hu,
        segmenter,
        min_volume_ml=config.min_lung_volume_ml,
        body_threshold_hu=config.body_threshold_hu,
        max_slice_gap_mm=config.lungmask_max_slice_gap_mm,
    )
    seconds["lung_mask"] = time.perf_counter() - started

    started = time.perf_counter()
    spacing = config.target_spacing_mm
    image_rs = steps.resample(
        hu, spacing, interpolator=sitk.sitkLinear, default_value=config.hu_floor
    )
    mask_rs = steps.resample(
        steps.mask_image(lung.mask, hu),
        spacing,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0,
    )
    seconds["resample"] = time.perf_counter() - started

    started = time.perf_counter()
    array = sitk.GetArrayFromImage(image_rs)
    box = steps.crop_box(sitk.GetArrayViewFromImage(mask_rs), spacing, config.crop_margin_mm)
    cropped = array[box]
    low, high = config.window_hu
    windowed = steps.lung_window(cropped, low, high)
    volume = steps.resize(windowed, config.output_size, config.output_dtype)
    seconds["crop_window_resize"] = time.perf_counter() - started

    return Preprocessed(
        volume=volume,
        hu_image=hu,
        lung=lung,
        resampled_size=tuple(image_rs.GetSize()),  # type: ignore[arg-type]
        crop=tuple((s.start, s.stop) for s in box),
        seconds={name: round(value, 3) for name, value in seconds.items()},
    )
