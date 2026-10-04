"""Section 11.4 augmentation (training only), applied to a batch on the training device:

- left-right flip (p = 0.5);
- rotation ±10° about each axis, scaling 0.9-1.1 and translation ±8 voxels, as one affine
  resampling (trilinear); voxels pulled in from outside the volume are air (-1024 HU);
- intensity scale ±10% and Gaussian noise σ = 0.01.

Volumes are [N, 1, D, H, W] with axes [z, y, x] in LPS, so left-right is the last axis.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from cavinet_ml.config import PreprocessingConfig
from cavinet_ml.training.config import AugmentationConfig


def air_value(config: PreprocessingConfig) -> float:
    """-1024 HU after the section 11.1 lung window."""
    low, high = config.window_hu
    return (max(config.hu_floor, low) - low) / (high - low)


@dataclass
class AugmentationParameters:
    flip: torch.Tensor  # [N] bool
    angles: torch.Tensor  # [N, 3] radians about x, y, z
    scale: torch.Tensor  # [N]
    shift: torch.Tensor  # [N, 3] voxels along x, y, z
    intensity: torch.Tensor  # [N] multiplier


def sample_parameters(
    n: int,
    config: AugmentationConfig,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> AugmentationParameters:
    def uniform(*shape: int, low: float, high: float) -> torch.Tensor:
        return torch.rand(*shape, generator=generator, device=device) * (high - low) + low

    angle = math.radians(config.rotate_degrees)
    low, high = config.scale
    return AugmentationParameters(
        flip=torch.rand(n, generator=generator, device=device) < config.flip_left_right_p,
        angles=uniform(n, 3, low=-angle, high=angle),
        scale=uniform(n, low=low, high=high),
        shift=uniform(n, 3, low=-config.translate_voxels, high=config.translate_voxels),
        intensity=uniform(n, low=1 - config.intensity_scale, high=1 + config.intensity_scale),
    )


def _rotation(angles: torch.Tensor) -> torch.Tensor:
    """[N, 3, 3] rotation matrices from angles about x, y and z."""
    cx, cy, cz = torch.cos(angles).unbind(-1)
    sx, sy, sz = torch.sin(angles).unbind(-1)
    one, zero = torch.ones_like(cx), torch.zeros_like(cx)
    rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], -1).view(-1, 3, 3)
    ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], -1).view(-1, 3, 3)
    rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], -1).view(-1, 3, 3)
    return rz @ ry @ rx


def affine_theta(params: AugmentationParameters, size: tuple[int, int, int]) -> torch.Tensor:
    """[N, 3, 4] matrices for F.affine_grid (normalised x, y, z coordinates). The grid maps
    each output voxel to the input position it samples, so the inverse transform is used:
    a scale s > 1 enlarges the anatomy, a positive shift moves it towards +x/+y/+z."""
    depth, height, width = size
    rotation = _rotation(params.angles)
    matrix = rotation.transpose(1, 2) / params.scale[:, None, None]
    voxels_to_norm = torch.tensor(
        [2.0 / width, 2.0 / height, 2.0 / depth], device=params.shift.device
    )
    shift = params.shift * voxels_to_norm
    translation = -(matrix @ shift[:, :, None])
    return torch.cat([matrix, translation], dim=2)


def augment_batch(
    x: torch.Tensor,
    config: AugmentationConfig,
    generator: torch.Generator,
    fill_value: float,
    params: AugmentationParameters | None = None,
) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError("expected a batch of shape [N, 1, D, H, W]")
    x = x.float()
    n = x.shape[0]
    params = params or sample_parameters(n, config, generator, x.device)
    x = torch.where(params.flip[:, None, None, None, None], x.flip(-1), x)
    theta = affine_theta(params, tuple(x.shape[2:]))
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    # grid_sample pads with 0; shifting by the fill value makes the padding air instead.
    x = F.grid_sample(
        x - fill_value, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    x = (x + fill_value) * params.intensity[:, None, None, None, None]
    if config.gaussian_noise_std > 0:
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        x = x + noise * config.gaussian_noise_std
    return x
