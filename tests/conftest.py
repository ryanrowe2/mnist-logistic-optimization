"""Offline, synthetic fixtures. Never cite these as MNIST performance results."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="module")
def fixture_mnist():
    # Full canonical SHAPE, but the contents are deliberately artificial.
    pixels = np.zeros((70000, 784), dtype=np.uint8)
    labels = np.arange(70000, dtype=np.uint8) % 10
    pixels[:, 0] = labels * 20
    pixels[:, 1] = 255 - labels * 20
    return pixels, labels
