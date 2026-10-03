import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from cavinet_ml.demo.build import build_demo_bundle  # noqa: E402


@pytest.fixture(scope="session")
def tiny_bundle_path(tmp_path_factory) -> Path:
    """A quickly trained demo-style bundle (32³ synthetic volumes, one epoch)."""
    torch.set_num_threads(2)
    path = tmp_path_factory.mktemp("bundle") / "cavinet_model.pth"
    build_demo_bundle(path, dev_cases=10, test_cases=6, epochs=1, size=32, log=lambda m: None)
    return path
