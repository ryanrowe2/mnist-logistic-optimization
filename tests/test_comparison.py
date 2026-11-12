"""Tests for a real three-method comparison with explicitly synthetic fixture."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mnist_optimization.experiments import Configuration


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts/compare_optimizers.py"
    spec = importlib.util.spec_from_file_location("compare_optimizers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_three_methods_same_split_and_real_computed_outputs(fixture_mnist, tmp_path):
    script = load_script()
    summary = script.compare(
        *fixture_mnist,
        config=Configuration(train_per_class=4, test_per_class=3, iterations=12, record_every=4),
        step_sizes={"gd": 1e-4, "linf": 1e-8, "coordinate": 1e-4},
        output=tmp_path,
        source="synthetic_test_fixture",
    )
    assert set(summary["methods"]) == {"gd", "linf", "coordinate"}
    assert summary["source"] == "synthetic_test_fixture"
    indices = []
    for method in ("gd", "linf", "coordinate"):
        with np.load(tmp_path / method / "numeric_results.npz") as result:
            indices.append(result["train_indices"].copy())
    for item in indices[1:]:
        np.testing.assert_array_equal(item, indices[0])
    assert (tmp_path / "comparison.png").stat().st_size > 1000
    assert (tmp_path / "comparison.json").stat().st_size > 100


def test_comparison_rejects_missing_or_invalid_step_sizes(fixture_mnist, tmp_path):
    script = load_script()
    cfg = Configuration(train_per_class=3, test_per_class=2, iterations=1)
    with pytest.raises(ValueError):
        script.compare(*fixture_mnist, config=cfg, step_sizes={"gd": 0.1}, output=tmp_path, source="fixture")
    with pytest.raises(ValueError):
        script.compare(
            *fixture_mnist,
            config=cfg,
            step_sizes={"gd": 0.1, "linf": np.nan, "coordinate": 0.1},
            output=tmp_path,
            source="fixture",
        )
