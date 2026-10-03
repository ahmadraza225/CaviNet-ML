"""Section 11.2: MONAI 3D ResNet-18, 1 input channel, 1 output logit (TB vs NTM), dropout
0.3 before the final layer. The defaults match MedicalNet's ResNet-18 (7×7×7 stride-2 first
convolution, shortcut type A) so its pretrained weights can initialise training."""

from typing import Any

import torch
from monai.networks.nets import ResNet
from torch import nn

DEFAULT_ARCHITECTURE: dict[str, Any] = {
    "name": "monai_resnet18",
    "block": "basic",
    "layers": [2, 2, 2, 2],
    "block_inplanes": [64, 128, 256, 512],
    "spatial_dims": 3,
    "n_input_channels": 1,
    "conv1_t_size": 7,
    "conv1_t_stride": 2,
    "shortcut_type": "A",
    "bias_downsample": True,
    "dropout": 0.3,
    "num_outputs": 1,
}


class CaviNetResNet(nn.Module):
    """ResNet backbone → global average pool → dropout → linear → one logit (TB)."""

    def __init__(self, architecture: dict[str, Any]) -> None:
        super().__init__()
        self.backbone = ResNet(
            block=architecture["block"],
            layers=list(architecture["layers"]),
            block_inplanes=list(architecture["block_inplanes"]),
            spatial_dims=architecture["spatial_dims"],
            n_input_channels=architecture["n_input_channels"],
            conv1_t_size=architecture["conv1_t_size"],
            conv1_t_stride=architecture["conv1_t_stride"],
            shortcut_type=architecture["shortcut_type"],
            bias_downsample=architecture["bias_downsample"],
            feed_forward=False,
        )
        features = architecture["block_inplanes"][-1]
        self.dropout = nn.Dropout(architecture["dropout"])
        self.fc = nn.Linear(features, architecture["num_outputs"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.backbone(x))).squeeze(-1)


def build_model(architecture: dict[str, Any] | None = None) -> CaviNetResNet:
    return CaviNetResNet({**DEFAULT_ARCHITECTURE, **(architecture or {})})


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
