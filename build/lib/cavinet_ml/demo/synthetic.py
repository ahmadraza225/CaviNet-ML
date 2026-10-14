"""Synthetic training volumes in the model's input space (128³, values in [0, 1] after the
lung window). Class 1 ("TB-like") has thick-walled cavities in the upper lungs; class 0
("NTM-like") has many small nodules lower down. These are toy patterns only."""

from functools import lru_cache

import numpy as np

AIR, LUNG, TISSUE = 0.22, 0.33, 0.93


@lru_cache(maxsize=4)
def _grid(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    return np.meshgrid(axis, axis, axis, indexing="ij")  # z (feet→head), y, x


def _sphere(z, y, x, centre, radius) -> np.ndarray:
    cz, cy, cx = centre
    return (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= radius**2


def synthetic_volume(label: int, rng: np.random.Generator, size: int = 128) -> np.ndarray:
    z, y, x = _grid(size)
    volume = np.full(z.shape, AIR, dtype=np.float32)
    volume[(y / 0.75) ** 2 + (x / 0.92) ** 2 <= 1] = TISSUE
    lungs = []
    for side in (-1, 1):
        cx = side * rng.uniform(0.38, 0.46)
        lung = ((z / 0.85) ** 2 + (y / 0.5) ** 2 + ((x - cx) / 0.3) ** 2) <= 1
        volume[lung] = LUNG
        lungs.append(cx)
    voxel = 2.0 / size
    if label == 1:
        for _ in range(rng.integers(2, 4)):
            centre = (
                rng.uniform(0.2, 0.6),
                rng.uniform(-0.2, 0.2),
                lungs[rng.integers(0, 2)] + rng.uniform(-0.06, 0.06),
            )
            outer = rng.uniform(10, 14) * voxel
            volume[_sphere(z, y, x, centre, outer)] = 0.9
            volume[_sphere(z, y, x, centre, outer - 4 * voxel)] = 0.08
    else:
        for _ in range(rng.integers(16, 26)):
            centre = (
                rng.uniform(-0.65, 0.3),
                rng.uniform(-0.35, 0.35),
                lungs[rng.integers(0, 2)] + rng.uniform(-0.18, 0.18),
            )
            volume[_sphere(z, y, x, centre, rng.uniform(3, 4.5) * voxel)] = 0.85
    volume += rng.normal(0.0, 0.02, volume.shape).astype(np.float32)
    return np.clip(volume, 0.0, 1.0).astype(np.float16)


def synthetic_set(labels, seed: int, size: int = 128) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [synthetic_volume(int(label), rng, size) for label in labels]
