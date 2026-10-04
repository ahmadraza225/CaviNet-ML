"""Section 11.1 step 3: the pretrained lungmask U-Net (R231).

The weights are baked into the Docker image at build time (no download at runtime). The
path comes from the caller (the application reads LUNGMASK_WEIGHTS). If the file is missing
the segmenter raises SegmenterUnavailable, and preprocessing falls back to the body crop
with a recorded warning. The training toolkit downloads the weights once with
`download_r231` and may run the segmenter on the GPU.
"""

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from cavinet_ml.preprocessing.steps import SegmenterUnavailable

R231_FILENAME = "unet_r231-d5d2fc3d.pth"
R231_SHA256 = "d5d2fc3df432115933491f115b283e94d1f5c841c0e82f6e7bd2d65d3ccde69f"
R231_URL = f"https://github.com/JoHof/lungmask/releases/download/v0.0/{R231_FILENAME}"


# Slices per forward pass. On CPU small batches are as fast as lungmask's default of 20 and
# need far less memory (measured on 100 slices of 512 x 512: batch 20 took 57 s and 2.9 GB
# extra, batch 2 took 44 s and 0.5 GB). The mask does not depend on it (BatchNorm in eval).
BATCH_SIZE = 2
GPU_BATCH_SIZE = 20  # lungmask's default; used by the training toolkit on a GPU
DEFAULT_WEIGHTS_PATH = Path("~/.cache/cavinet").expanduser() / R231_FILENAME


def download_r231(destination: Path | str = DEFAULT_WEIGHTS_PATH) -> Path:
    """Download the R231 weights once (checked against their SHA-256)."""
    destination = Path(destination).expanduser()
    if destination.is_file() and _sha256(destination) == R231_SHA256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    request = urllib.request.Request(R231_URL, headers={"User-Agent": "cavinet-ml"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    if _sha256(temporary) != R231_SHA256:
        temporary.unlink(missing_ok=True)
        raise SegmenterUnavailable("the downloaded lungmask weights do not match their SHA-256")
    temporary.replace(destination)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LungmaskSegmenter:
    """Callable segmenter; the model is loaded on first use and then reused."""

    def __init__(
        self,
        weights_path: str | os.PathLike | None,
        batch_size: int = BATCH_SIZE,
        *,
        force_cpu: bool = True,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self.batch_size = batch_size
        self.force_cpu = force_cpu  # the application always runs on the CPU
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
            force_cpu=self.force_cpu,
            batch_size=self.batch_size,
            tqdm_disable=True,
        )

    def __call__(self, image: sitk.Image) -> np.ndarray:
        if self._inferer is None:
            self._inferer = self._load()
        return self._inferer.apply(image)
