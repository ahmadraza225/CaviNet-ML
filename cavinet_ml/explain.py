"""M-11 heatmap explanation (FR-11.1 to FR-11.4).

Grad-CAM on the last convolutional block (layer4) of each fold model, for the predicted class;
each fold's map is upsampled to the model input, scaled to [0, 1] and the folds are averaged.
The averaged map is then mapped back onto the original scan, so it lines up with the 48 preview
slices of the viewer, and drawn as transparent overlays (`heatmap-NN.png`, one per preview). The
3 previews holding the largest share of the heatmap are highlighted and composited for the PDF
(`heatmap-top-K.png`). The share of heatmap energy inside the lung mask (in model space) is the
H4 measure (FR-11.4).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image

from cavinet_ml.config import PreprocessingConfig
from cavinet_ml.preprocessing import steps
from cavinet_ml.preprocessing.pipeline import Preprocessed
from cavinet_ml.previews import PreviewSet, render

LABEL = "Regions that most influenced the prediction. This is not a lesion marker."
TOP_SLICES = 3
H4_TARGET = 0.80  # section 4.3: at least 80% of the heatmap energy inside the lungs
OVERLAY_FLOOR = 0.4  # map values below this are left transparent
OVERLAY_ALPHA = 0.55  # opacity of the strongest regions


@dataclass
class HeatmapSet:
    files: list[str]  # heatmap-NN.png, one transparent overlay per preview-NN.png
    top: list[int]  # preview numbers (0-based) with the largest heatmap share, largest first
    top_files: list[str]  # heatmap-top-K.png: preview + overlay, for the report
    slice_share: list[float]  # share of the heatmap energy shown on each preview slice
    lung_share: float | None  # share of the heatmap energy inside the lung mask (model space)
    target: str  # the class the heatmap explains
    seconds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "files": self.files,
            "top": self.top,
            "top_files": self.top_files,
            "slice_share": [round(v, 4) for v in self.slice_share],
            "lung_share": None if self.lung_share is None else round(self.lung_share, 4),
            "target": self.target,
            "label": LABEL,
        }


def _fold_cam(model: torch.nn.Module, x: torch.Tensor, sign: float) -> torch.Tensor:
    """One fold's Grad-CAM at the resolution of layer4 (before upsampling)."""
    captured: dict[str, torch.Tensor] = {}
    hook = model.backbone.layer4.register_forward_hook(
        lambda _module, _inputs, output: captured.__setitem__("a", output)
    )
    try:
        with torch.enable_grad():
            score = sign * model(x).sum()
            (gradients,) = torch.autograd.grad(score, captured["a"])
    finally:
        hook.remove()
    weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
    return torch.relu((weights * captured["a"].detach()).sum(dim=1, keepdim=True))


def grad_cam(models: list[torch.nn.Module], volume: np.ndarray, sign: float) -> np.ndarray:
    """The fold-averaged Grad-CAM of `volume` (model input), in [0, 1], same shape.
    `sign` is +1 to explain the positive class (TB, a higher logit) and -1 for the other."""
    x = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))[None, None]
    maps = []
    for model in models:
        cam = torch.nn.functional.interpolate(
            _fold_cam(model, x, sign),
            size=tuple(volume.shape),
            mode="trilinear",
            align_corners=False,
        )[0, 0]
        peak = float(cam.max())
        maps.append(cam / peak if peak > 0 else cam)
    mean = torch.stack(maps).mean(dim=0)
    peak = float(mean.max())
    return (mean / peak if peak > 0 else mean).numpy().astype(np.float32)


def lung_share(cam: np.ndarray, mask: np.ndarray | None) -> float | None:
    """Share of the heatmap energy inside the lung mask (FR-11.4); None without a map."""
    total = float(cam.sum())
    if mask is None or total <= 0:
        return None
    return float((cam * (mask > 0)).sum()) / total


def to_original(cam: np.ndarray, preprocessed: Preprocessed, spacing_mm: float) -> np.ndarray:
    """The model-space map placed back into the crop box and resampled onto the original
    scan's grid (z, y, x), so slice z matches the preview of slice z."""
    size_x, size_y, size_z = preprocessed.resampled_size
    (z0, z1), (y0, y1), (x0, x1) = preprocessed.crop
    full = np.zeros((size_z, size_y, size_x), dtype=np.float32)
    full[z0:z1, y0:y1, x0:x1] = steps.resize(cam, (z1 - z0, y1 - y0, x1 - x0), "float32")
    reference = preprocessed.hu_image
    image = sitk.GetImageFromArray(full)
    image.SetOrigin(reference.GetOrigin())
    image.SetSpacing((spacing_mm,) * 3)
    image.SetDirection(reference.GetDirection())
    resampled = sitk.Resample(
        image, reference, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )
    return np.clip(sitk.GetArrayFromImage(resampled), 0.0, 1.0)


def colour(values: np.ndarray) -> np.ndarray:
    """RGBA overlay of a [0, 1] map: blue (weak) to yellow to red (strong); transparent
    below OVERLAY_FLOOR."""
    v = np.clip(values, 0.0, 1.0)
    red = np.clip(2.0 * v, 0.0, 1.0)
    green = np.clip(2.0 - 2.0 * v, 0.0, 1.0) * np.clip(2.0 * v, 0.0, 1.0)
    blue = np.clip(1.0 - 2.0 * v, 0.0, 1.0)
    strength = (v - OVERLAY_FLOOR) / (1 - OVERLAY_FLOOR)
    alpha = np.where(v < OVERLAY_FLOOR, 0.0, OVERLAY_ALPHA * strength)
    rgba = np.stack([red, green, blue, alpha], axis=-1)
    return (rgba * 255).round().astype(np.uint8)


def overlay(cam_slice: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """The RGBA overlay of one axial slice, resized to the preview's `size` (width, height)."""
    image = Image.fromarray(colour(cam_slice), mode="RGBA")
    return image.resize(size, Image.Resampling.BILINEAR)


def write_heatmaps(
    cam: np.ndarray,
    preprocessed: Preprocessed,
    previews: PreviewSet,
    out_dir: Path,
    config: PreprocessingConfig,
    target: str,
) -> HeatmapSet:
    original = to_original(cam, preprocessed, config.target_spacing_mm)
    hu = sitk.GetArrayViewFromImage(preprocessed.hu_image)
    spacing = preprocessed.hu_image.GetSpacing()[:2]
    energy = np.array([float(original[z].sum()) for z in previews.slice_indices])
    total = float(energy.sum())
    share = (energy / total).tolist() if total > 0 else [0.0] * len(energy)
    top = [int(i) for i in np.argsort(-energy, kind="stable")[:TOP_SLICES]] if total > 0 else []
    out_dir.mkdir(parents=True, exist_ok=True)
    files, top_files = [], []
    for number, z in enumerate(previews.slice_indices):
        grey = render(hu[z], spacing, config.window_hu, config.preview_max_size)
        name = f"heatmap-{number:02d}.png"
        layer = overlay(original[z], grey.size)
        layer.save(out_dir / name, optimize=True)
        files.append(name)
        if number in top:
            rank = top.index(number) + 1
            composite = Image.alpha_composite(grey.convert("RGBA"), layer).convert("RGB")
            composite.save(out_dir / f"heatmap-top-{rank}.png", optimize=True)
    top_files = [f"heatmap-top-{rank}.png" for rank in range(1, len(top) + 1)]
    return HeatmapSet(
        files=files,
        top=top,
        top_files=top_files,
        slice_share=share,
        lung_share=lung_share(cam, preprocessed.model_mask),
        target=target,
    )
