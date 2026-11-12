"""Adversarial regression tests for the independent 2026-11-11 repository audit."""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from mnist_optimization.artifacts import validate_inventory
from mnist_optimization.data import prepare_binary, validate_mnist
from mnist_optimization.experiments import Configuration, main, run_dataset, run_prepared
from mnist_optimization.logistic import add_intercept, logistic_gradient, logistic_loss, predict, scores
from mnist_optimization.optimizers import descent_direction
from mnist_optimization.provenance import OPENML_LOADER_SOURCE
from mnist_optimization.tuning import train_validation_indices, tune_and_evaluate


def tiny_split():
    from mnist_optimization.data import BinarySplit

    X = np.array([[2.0, 1.0], [1.0, 0.0], [-2.0, 1.0], [-1.0, 0.0]])
    T = np.array([[1.5, 0.0], [-1.5, 0.0]])
    return BinarySplit(
        X,
        np.array([1, 1, -1, -1]),
        T,
        np.array([1, -1]),
        np.arange(4),
        np.arange(4, 6),
        (4, 9),
        "historical",
        "raw_0_255",
        42,
        None,
    )


def tiny_config(**kwargs):
    return Configuration(train_per_class=2, test_per_class=1, seed=42, iterations=3, record_every=1, **kwargs)


def load_comparison_script():
    path = Path(__file__).resolve().parents[1] / "scripts/compare_optimizers.py"
    spec = importlib.util.spec_from_file_location("comparison_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reserved_openml_label_cannot_certify_prepared_synthetic(tmp_path):
    with pytest.raises(ValueError, match="provenance"):
        run_prepared(tiny_split(), tiny_config(), output=tmp_path / "report", source=OPENML_LOADER_SOURCE)
    assert not (tmp_path / "report").exists()


def test_reserved_openml_label_cannot_certify_supplied_npz_shape_data(fixture_mnist, tmp_path):
    with pytest.raises(ValueError, match="reserved"):
        run_dataset(
            *fixture_mnist,
            Configuration(train_per_class=2, test_per_class=1, iterations=1),
            output=tmp_path / "report",
            source=OPENML_LOADER_SOURCE,
        )
    assert not (tmp_path / "report").exists()


def test_success_manifest_and_immutability(tmp_path):
    folder = tmp_path / "run"
    run_prepared(tiny_split(), tiny_config(), output=folder, source="synthetic")
    assert validate_inventory(folder)
    with pytest.raises(FileExistsError):
        run_prepared(tiny_split(), tiny_config(method="coordinate"), output=folder, source="synthetic")
    assert validate_inventory(folder)
    assert not (folder / "figures/coordinate_updates.png").exists()
    (folder / "metrics.json").write_text("corrupted")
    assert not validate_inventory(folder)


def test_crash_rolls_back_entire_run(monkeypatch, tmp_path):
    import mnist_optimization.experiments as module

    def crash(*args, **kwargs):
        raise RuntimeError("figure renderer failed")

    monkeypatch.setattr(module, "save_figures", crash)
    target = tmp_path / "crashed"
    with pytest.raises(RuntimeError, match="renderer"):
        run_prepared(tiny_split(), tiny_config(), output=target, source="synthetic")
    assert not target.exists()
    assert list(tmp_path.glob(".crashed-incomplete-*")) == []


def test_complex_data_never_silently_discards_imaginary_parts(fixture_mnist):
    with pytest.raises(ValueError, match="real numeric"):
        logistic_loss(np.zeros(2), [[1 + 2j, 3 - 7j]], [1])
    for fn in [logistic_gradient, logistic_loss]:
        with pytest.raises(ValueError, match="real numeric"):
            fn(np.array([1 + 5j]), [[1]], [1])
    with pytest.raises(ValueError, match="real numeric"):
        scores([[1 + 1j]], [1])
    with pytest.raises(ValueError, match="real numeric"):
        add_intercept([[1 + 1j]])
    with pytest.raises(ValueError, match="real numeric"):
        descent_direction(np.array([1 + 3j]), "gd")
    X, y = fixture_mnist
    with pytest.raises(ValueError, match="numeric"):
        validate_mnist(X.astype(np.complex64) + 1j, y)
    with pytest.raises(ValueError, match="numeric"):
        prepare_binary(X.astype(np.complex64) + 1j, y)


def test_extreme_dot_product_overflow_explicit():
    for fn in (logistic_loss, logistic_gradient):
        with pytest.raises(FloatingPointError, match="overflow"):
            fn([1e308], [[1e308]], [1])
    with pytest.raises(FloatingPointError, match="overflow"):
        scores([[1e308]], [1e308])


def test_historical_sign_tie_is_marked_unclassified():
    assert predict([[0.0], [1.0]], [0.0], historical_sign=True).tolist() == [0, 0]
    from mnist_optimization.evaluation import metrics

    info = metrics([-1, 1], [0, -1])
    assert info["errors"] == 2
    assert info["unclassified_ties"] == 1
    assert info["confusion_rows_true_cols_pred_minus_plus_ties_if_present"] == [[0, 0, 1], [1, 0, 0]]


def test_repeated_inputs_produce_identical_metrics_bytes(tmp_path):
    locations = [tmp_path / "one", tmp_path / "two"]
    for folder in locations:
        run_prepared(tiny_split(), tiny_config(), output=folder, source="synthetic")
    assert (locations[0] / "metrics.json").read_bytes() == (locations[1] / "metrics.json").read_bytes()


@pytest.mark.parametrize("bad", ["--iterations", "--step-size", "--digits", "--l2"])
def test_invalid_configuration_rejected_before_data_fetch(monkeypatch, bad, tmp_path):
    import mnist_optimization.experiments as module

    def fail_network(**kwargs):
        raise AssertionError("NETWORK SHOULD NEVER BE CONTACTED")

    monkeypatch.setattr(module, "load_mnist", fail_network)
    args = {
        "--iterations": ["--iterations", "-1"],
        "--step-size": ["--step-size", "0"],
        "--digits": ["--digits", "4", "4"],
        "--l2": ["--l2", "-2"],
    }[bad]
    with pytest.raises(ValueError):
        main(args + ["--output", str(tmp_path / "never")])


def test_wrapper_scripts_forward_extra_arguments(monkeypatch):
    import mnist_optimization.experiments as module

    collected = []

    def fake_main(argv):
        collected.append(argv)
        return 0

    monkeypatch.setattr(module, "main", fake_main)
    root = Path(__file__).resolve().parents[1]
    for name in ("reproduce_original.py", "run_experiments.py"):
        monkeypatch.setattr(sys, "argv", [name, "--local-npz", "foo.npz", "--seed", "99", "--help"])
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(root / "scripts" / name), run_name="__main__")
        assert exc.value.code == 0
        assert collected[-1][-5:] == ["--local-npz", "foo.npz", "--seed", "99", "--help"]


def test_compare_failure_rolls_back(monkeypatch, fixture_mnist, tmp_path):
    module = load_comparison_script()
    original = module.run_dataset
    counter = 0

    def crash_at_second(*args, **kwargs):
        nonlocal counter
        counter += 1
        if counter == 2:
            raise RuntimeError("second optimizer failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "run_dataset", crash_at_second)
    output = tmp_path / "failed_comparison"
    with pytest.raises(RuntimeError, match="second"):
        module.compare(
            *fixture_mnist,
            config=Configuration(train_per_class=2, test_per_class=2, iterations=2),
            step_sizes={"gd": 1e-4, "linf": 1e-8, "coordinate": 1e-4},
            output=output,
            source="synthetic",
        )
    assert not output.exists()


def test_stratified_selection_deterministic_and_disjoint():
    labels = np.array([-1] * 15 + [1] * 15)
    one = train_validation_indices(labels, seed=7)
    two = train_validation_indices(labels, seed=7)
    for a, b in zip(one, two, strict=True):
        np.testing.assert_array_equal(a, b)
    train, validation = one
    assert len(np.intersect1d(train, validation)) == 0
    assert len(train) + len(validation) == len(labels)
    assert np.sum(labels[validation] == -1) == np.sum(labels[validation] == 1)


def test_train_only_tuning_results_are_real_and_separate(fixture_mnist, tmp_path):
    grid = {"gd": [1e-4, 5e-5], "linf": [1e-8], "coordinate": [1e-4]}
    output = tmp_path / "tuning"
    report = tune_and_evaluate(
        *fixture_mnist,
        config=Configuration(train_per_class=8, test_per_class=3, iterations=3),
        step_grids=grid,
        output=output,
        source="fixture",
    )
    assert report["selection_training_n"] + report["validation_n"] == 16
    assert report["test_n"] == 6
    assert set(report["final_refit_and_test_once_per_method"]) == set(grid)
    assert validate_inventory(output)
    for method in grid:
        assert validate_inventory(output / method)
    with np.load(output / "train_validation_rows.npz") as data:
        assert len(np.intersect1d(data["validation_source_indices"], data["training_source_indices"])) == 0


def test_tuning_rejects_invalid_grid_before_preparation(tmp_path):
    with pytest.raises(ValueError, match="grid"):
        tune_and_evaluate(
            np.array([]),
            np.array([]),
            config=Configuration(),
            step_grids={"gd": [0], "linf": [0.1], "coordinate": [0.1]},
            output=tmp_path / "no",
            source="synthetic",
        )
    assert not (tmp_path / "no").exists()


def test_train_only_selection_is_invariant_to_test_pixel_changes(fixture_mnist, tmp_path):
    """Change only official test pixels: validation selection must remain identical."""
    X, y = fixture_mnist
    altered = X.copy()
    altered[60000:] = 255 - altered[60000:]
    grid = {"gd": [0.05, 0.1], "linf": [0.001], "coordinate": [0.5]}
    config = Configuration(protocol="official", train_per_class=6, test_per_class=3, iterations=2, step_size=0.1)
    old = tune_and_evaluate(X, y, config=config, step_grids=grid, output=tmp_path / "original", source="synthetic")
    new = tune_and_evaluate(
        altered, y, config=config, step_grids=grid, output=tmp_path / "changed_test", source="synthetic"
    )
    assert old["selection"] == new["selection"]
    assert old["dataset_sha256"] != new["dataset_sha256"]
