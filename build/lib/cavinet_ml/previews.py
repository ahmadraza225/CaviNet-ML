"""FR-05.3: 48 evenly spaced axial preview slices (lung window, PNG) for the viewer, and 3
representative slices for the report. Slices come from the cleaned HU volume at its original
resolution, over the lungs' head-to-feet extent (from the step 3 mask)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

from cavinet_ml.config import DEFAULT_PREPROCESSING, PreprocessingConfig


@dataclass
class PreviewSet:
    count: int
    files: list[str]  # file names in viewing order, head to feet
    representative: list[str]
    slice_indices: list[int]  # z index in the LPS volume of each preview


def _extent(mask: np.ndarray, count: int) -> tuple[int, int]:
    """First and last z index containing the mask; the whole scan if that is too short."""
    depth = mask.shape[0]
    present = np.flatnonzero(mask.reshape(depth, -1).any(axis=1))
    if present.size == 0 or (present[-1] - present[0] + 1) < count:
        return 0, depth - 1
    return int(present[0]), int(present[-1])


def preview_indices(mask: np.ndarray, count: int) -> list[int]:
    """`count` evenly spaced z indices over the lungs, head (high z in LPS) to feet."""
    first, last = _extent(mask, count)
    return [int(round(z)) for z in np.linspace(last, first, count)]


def render(
    hu_slice: np.ndarray,
    spacing_xy: tuple[float, float],
    window: tuple[float, float],
    max_size: int,
) -> Image.Image:
    """Lung-window an axial slice to 8-bit grey, scaled to `max_size` with true proportions.
    Displayed radiologically: anterior up, the patient's right on the left."""
    low, high = window
    grey = ((np.clip(hu_slice, low, high) - low) / (high - low) * 255).round().astype(np.uint8)
    image = Image.fromarray(grey, mode="L")
    width_mm = hu_slice.shape[1] * spacing_xy[0]
    height_mm = hu_slice.shape[0] * spacing_xy[1]
    scale = max_size / max(width_mm, height_mm)
    size = (max(1, round(width_mm * scale)), max(1, round(height_mm * scale)))
    return image.resize(size, Image.Resampling.BILINEAR)


def write_previews(
    hu_image: sitk.Image,
    mask: np.ndarray,
    out_dir: Path,
    config: PreprocessingConfig = DEFAULT_PREPROCESSING,
) -> PreviewSet:
    out_dir.mkdir(parents=True, exist_ok=True)
    array = sitk.GetArrayViewFromImage(hu_image)
    spacing = hu_image.GetSpacing()[:2]
    indices = preview_indices(mask, config.preview_count)
    files = []
    for number, z in enumerate(indices):
        name = f"preview-{number:02d}.png"
        render(array[z], spacing, config.window_hu, config.preview_max_size).save(
            out_dir / name, optimize=True
        )
        files.append(name)
    representative = []
    for number, fraction in enumerate(
        np.linspace(0, 1, config.representative_count + 2)[1:-1], start=1
    ):
        z = indices[int(round(fraction * (len(indices) - 1)))]
        name = f"representative-{number}.png"
        render(array[z], spacing, config.window_hu, config.preview_max_size).save(
            out_dir / name, optimize=True
        )
        representative.append(name)
    return PreviewSet(len(files), files, representative, indices)
