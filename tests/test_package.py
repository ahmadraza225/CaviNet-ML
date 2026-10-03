import importlib

import pytest

SUBPACKAGES = ["io", "preprocessing", "model", "inference", "training", "evaluation"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_importable(name):
    module = importlib.import_module(f"cavinet_ml.{name}")
    assert module.__doc__
