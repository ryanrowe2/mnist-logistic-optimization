import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from mnist_optimization.data import BinarySplit
from mnist_optimization.experiments import Configuration, run_prepared


def small_split():
    train = np.array([[2.0, 1.0], [1.0, 0.0], [-2.0, 1.0], [-1.0, 0.0]])
    test = np.array([[1.5, 1.0], [-1.5, 1.0]])
    return BinarySplit(
        train, np.array([1, 1, -1, -1]), test, np.array([1, -1]),
        np.arange(4), np.arange(4, 6), (4, 9), "historical", "raw_0_255", 42, None,
    )


def test_end_to_end_results_json_npz_and_figures(tmp_path):
    config = Configuration(train_per_class=2, test_per_class=1, seed=42, iterations=12, record_every=5, step_size=0.01)
    report = run_prepared(small_split(), config, output=tmp_path, source="synthetic_test_fixture")
    loaded = json.loads((tmp_path / "metrics.json").read_text())
    assert loaded["evaluation"] == report["evaluation"]
    assert loaded["data"]["canonical_mnist_assertion"] is False
    assert loaded["data"]["train_test_overlap"] == 0
    with np.load(tmp_path / "numeric_results.npz") as values:
        assert values["objective"].shape == (4,)  # 0, 5, 10, 12
        assert values["iteration"].tolist() == [0, 5, 10, 12]
        assert values["weights"].shape == (2,)
    assert (tmp_path / "figures/convergence.png").stat().st_size > 1000
    assert (tmp_path / "figures/confusion.png").stat().st_size > 1000


def test_cli_help_and_invalid_arguments():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    help_run = subprocess.run(
        [sys.executable, "-m", "mnist_optimization.experiments", "--help"],
        cwd=root, env=env, capture_output=True, text=True, check=False,
    )
    assert help_run.returncode == 0
    assert "--protocol" in help_run.stdout
    wrong = subprocess.run(
        [sys.executable, "-m", "mnist_optimization.experiments", "--protocol", "WRONG"],
        cwd=root, env=env, capture_output=True, text=True, check=False,
    )
    assert wrong.returncode != 0


def test_local_fullshape_cli_with_synthetic_fixture(fixture_mnist, tmp_path):
    """Exercise the full CLI without OpenML; reports MUST label this unverified."""
    from mnist_optimization.experiments import main

    fake = tmp_path / "fixture.npz"
    np.savez_compressed(fake, X=fixture_mnist[0], y=fixture_mnist[1])
    output = tmp_path / "command_results"
    code = main([
        "--local-npz", str(fake), "--protocol", "historical", "--digits", "4", "9",
        "--train-per-class", "5", "--test-per-class", "4", "--iterations", "6",
        "--record-every", "2", "--output", str(output),
    ])
    assert code == 0
    report = json.loads((output / "metrics.json").read_text())
    assert report["source"] == "user_supplied_unverified_array"
    assert report["data"]["canonical_mnist_assertion"] is False
    assert report["data"]["train_count"] == 10
    assert report["data"]["test_count"] == 8
    assert (output / "figures/weights.png").exists()


def test_coordinate_visual_and_update_counts(tmp_path):
    from mnist_optimization.data import BinarySplit

    X = np.zeros((4, 784), dtype=float)
    X[:, 0] = [1, 2, -2, -1]
    X[:, 1] = [2, 3, -3, -2]
    split = BinarySplit(
        X, np.array([1, 1, -1, -1]), X.copy(), np.array([1, 1, -1, -1]),
        np.arange(4), np.arange(4, 8), (4, 9), "historical", "raw_0_255", 42, None,
    )
    cfg = Configuration(
        train_per_class=2, test_per_class=2, seed=42, method="coordinate",
        iterations=12, record_every=2, step_size=0.01,
    )
    report = run_prepared(split, cfg, output=tmp_path, source="synthetic_test_fixture")
    assert report["optimization"]["coordinate_update_total"] == 12
    assert (tmp_path / "figures/coordinate_updates.png").stat().st_size > 1000


def test_run_dataset_dispatch(fixture_mnist, tmp_path):
    from mnist_optimization.experiments import run_dataset

    cfg = Configuration(train_per_class=3, test_per_class=2, iterations=2, record_every=1)
    report = run_dataset(*fixture_mnist, cfg, output=tmp_path)
    assert report["data"]["train_count"] == 6
    assert report["source"] == "user_supplied_unverified_array"


def test_run_prepared_rejects_mismatched_metadata(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="sample counts"):
        run_prepared(small_split(), Configuration(seed=42, iterations=2), output=tmp_path, source="synthetic")


def test_prepared_duplicate_test_index_rejected(tmp_path):
    import pytest
    from mnist_optimization.data import BinarySplit

    s = small_split()
    bad = BinarySplit(
        s.X_train, s.y_train, s.X_test, s.y_test, s.train_indices,
        np.array([0, 7]), s.digit_pair, s.protocol, s.scaling, s.seed, s.bias_index,
    )
    with pytest.raises(ValueError, match="disjoint"):
        run_prepared(
            bad, Configuration(train_per_class=2, test_per_class=1, seed=42, iterations=2),
            output=tmp_path, source="synthetic",
        )


def test_no_historical_score_is_created_from_synthetic_fixture(fixture_mnist, tmp_path):
    from mnist_optimization.experiments import run_dataset

    report = run_dataset(
        *fixture_mnist,
        Configuration(train_per_class=4, test_per_class=3, iterations=3),
        output=tmp_path, source="synthetic_test_fixture",
    )
    assert report["data"]["canonical_mnist_assertion"] is False
    assert report["source"] != "openml_dataset_554_v1"
    assert report["evaluation"]["count"] == 6


def test_historical_cli_defaults_to_per_update_recording():
    from mnist_optimization.experiments import Configuration, parse_arguments

    options = parse_arguments(["--protocol", "historical"])
    assert options.record_every is None  # resolved by main: 1 for historical
    assert Configuration().record_every == 1
