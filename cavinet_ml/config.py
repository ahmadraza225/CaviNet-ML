"""Section 11.1 preprocessing parameters. The same values are stored in every model bundle
(section 11.6), so training and the application always preprocess identically."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PreprocessingConfig:
    # Step 2: clip values below this (scanner padding is often -3024 or -2048).
    hu_floor: float = -1024.0
    # Step 3: lungmask model, and the smallest believable lung volume before falling back.
    lungmask_model: str = "R231"
    min_lung_volume_ml: float = 500.0
    # lungmask runs on the original-resolution slices; thin-slice scans are segmented on every
    # Nth slice so that no gap exceeds this (skipped slices copy the nearest segmented one).
    # The mask only sets the crop, and this keeps a 300-slice scan within the 3-minute budget.
    lungmask_max_slice_gap_mm: float = 3.0
    # Fallback crop: voxels above this HU belong to the body.
    body_threshold_hu: float = -500.0
    # Step 4: isotropic voxel size.
    target_spacing_mm: float = 1.5
    # Step 5: margin around the lung bounding box.
    crop_margin_mm: float = 10.0
    # Step 6: lung window, scaled to [0, 1].
    window_hu: tuple[float, float] = (-1350.0, 150.0)
    # Step 7: model input size (voxels) and storage type.
    output_size: tuple[int, int, int] = (128, 128, 128)
    output_dtype: str = "float16"
    # FR-05.3 previews.
    preview_count: int = 48
    representative_count: int = 3
    preview_max_size: int = 512
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain values only (lists, not tuples), so the bundle stays weights_only-safe."""
        values = asdict(self)
        values["window_hu"] = list(self.window_hu)
        values["output_size"] = list(self.output_size)
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "PreprocessingConfig":
        known = {name: values[name] for name in cls.__dataclass_fields__ if name in values}
        if "window_hu" in known:
            known["window_hu"] = tuple(float(v) for v in known["window_hu"])
        if "output_size" in known:
            known["output_size"] = tuple(int(v) for v in known["output_size"])
        return cls(**known)


DEFAULT_PREPROCESSING = PreprocessingConfig()
