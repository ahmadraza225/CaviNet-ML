"""Section 11.1 step 3: the pretrained lungmask U-Net (R231).

The weights are baked into the Docker image at build time (no download at runtime). The
path comes from the caller (the application reads LUNGMASK_WEIGHTS). If the file is missing
the segmenter raises SegmenterUnavailable, and preprocessing falls back to the body crop
with a recorded warning.
"""

import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from cavinet_ml.preprocessing.steps import SegmenterUnavailable

R231_FILENAME = "unet_r231-d5d2fc3d.pth"
R231_SHA256 = "d5d2fc3df432115933491f115b283e94d1f5c841c0e82f6e7bd2d65d3ccde69f"
R231_URL = f"https://github.com/JoHof/lungmask/releases/download/v0.0/{R231_FILENAME}"


class LungmaskSegmenter:
    """Callable segmenter; the model is loaded on first use and then reused."""

    def __init__(self, weights_path: str | os.PathLike | None, batch_size: int = 20) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self.batch_size = batch_size
        self._inferer = None

    @property
    def available(self) -> bool:
        return self.weights_path is not None and self.weights_path.is_file()

    def _load(self):
        if not self.available:
            raise SegmenterUnavailable(f"lungmask weights not found at {self.weights_path}")
        from lungmask import LMInferer  # heavy import, only when needed

        return LMInferer(
            modelpath=str(self.weights_path),
            force_cpu=True,
            batch_size=self.batch_size,
            tqdm_disable=True,
        )

    def __call__(self, image: sitk.Image) -> np.ndarray:
        if self._inferer is None:
            self._inferer = self._load()
        return self._inferer.apply(image)
