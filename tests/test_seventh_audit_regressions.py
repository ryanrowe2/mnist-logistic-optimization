"""Regression tests for the five v1.6.0 independent-audit findings."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from mnist_optimization.artifacts import completion_inventory, validate_inventory
from mnist_optimization.data import prepare_binary
from mnist_optimization.experiments import Configuration, run_prepared, validate_configuration
from mnist_optimization.tuning import train_validation_indices, tune_and_evaluate

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"seventh_{name}", ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("test_count", [1, 10, 100, 500])
def test_official_training_unchanged_as_evaluation_size_changes(fixture_mnist, test_count):
    X, y = fixture_mnist
    base = prepare_binary(X, y, protocol="official", train_per_class=120, test_per_class=50, seed=2024)
    altered = prepare_binary(X, y, protocol="official", train_per_class=120, test_per_class=test_count, seed=2024)
    np.testing.assert_array_equal(base.train_indices, altered.train_indices)
    np.testing.assert_array_equal(base.X_train, altered.X_train)
    np.testing.assert_array_equal(base.y_train, altered.y_train)
    assert np.all(base.train_indices < 60000)
    assert np.all(altered.test_indices >= 60000)
    assert np.intersect1d(base.train_indices, altered.test_indices).size == 0


def test_official_random_sampling_repeatable_and_independent_of_test_values(fixture_mnist):
    X, y = fixture_mnist
    first = prepare_binary(X, y, protocol="official", train_per_class=50, test_per_class=12, seed=2024)
    changed = X.copy()
    changed[60000:, 3] = 123
    second = prepare_binary(changed, y, protocol="official", train_per_class=50, test_per_class=120, seed=2024)
    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(first.X_train, second.X_train)
    repeated = prepare_binary(X, y, protocol="official", train_per_class=50, test_per_class=12, seed=2024)
    np.testing.assert_array_equal(first.test_indices, repeated.test_indices)


@pytest.mark.parametrize("value", [np.float16(0.25), np.float32(0.125), np.float64(0.75)])
def test_config_normalizes_supported_numpy_scalar_values(value):
    cfg = Configuration(step_size=value, l2=value)
    validate_configuration(cfg)
    assert type(cfg.step_size) is float
    assert type(cfg.l2) is float
    assert json.loads(json.dumps(asdict(cfg)))["l2"] == float(value)


def test_numpy_config_parameters_run_and_serialize(tmp_path):
    from mnist_optimization.data import BinarySplit

    split = BinarySplit(
        np.array([[2.0, 1.0], [1.0, 0.0], [-2.0, 1.0], [-1.0, 0.0]]),
        np.array([1, 1, -1, -1]),
        np.array([[1.5, 1.0], [-1.5, 1.0]]),
        np.array([1, -1]),
        np.arange(4),
        np.arange(4, 6),
        (4, 9),
        "historical",
        "raw_0_255",
        42,
        None,
    )
    cfg = Configuration(
        train_per_class=2, test_per_class=1, seed=42, iterations=2, step_size=np.float32(0.01), l2=np.float16(0.1)
    )
    report = run_prepared(split, cfg, output=tmp_path / "run", source="synthetic_fixture")
    assert type(report["config"]["step_size"]) is float
    assert json.loads((tmp_path / "run/metrics.json").read_text())["config"]["l2"] == float(cfg.l2)


def test_tuning_accepts_float32_fraction_and_candidate_steps(fixture_mnist, tmp_path):
    X, y = fixture_mnist
    cfg = Configuration(
        protocol="official", train_per_class=5, test_per_class=2, iterations=1, record_every=1, l2=np.float32(0.1)
    )
    steps = {name: [np.float32(0.01)] for name in ("gd", "linf", "coordinate")}
    report = tune_and_evaluate(
        X, y, config=cfg, step_grids=steps, output=tmp_path / "tuning", validation_fraction=np.float32(0.25)
    )
    assert type(report["validation_fraction"]) is float
    assert all(type(report["selection"][m]["selected_step"]) is float for m in steps)
    assert (tmp_path / "tuning/selection.json").is_file()
    training, validation = train_validation_indices(np.array([-1, -1, 1, 1]), seed=1, fraction=np.float32(0.25))
    assert len(training) == len(validation) == 2


def test_comparison_accepts_float32_steps(fixture_mnist, tmp_path):
    module = _script("compare_optimizers")
    steps = {name: np.float32(0.01) for name in ("gd", "linf", "coordinate")}
    report = module.compare(
        *fixture_mnist,
        config=Configuration(train_per_class=3, test_per_class=2, iterations=1, record_every=1),
        step_sizes=steps,
        output=tmp_path / "comparison",
        source="fixture",
    )
    assert all(type(report["methods"][m]["step_size"]) is float for m in steps)
    assert (tmp_path / "comparison/comparison.json").is_file()


def test_result_inventory_rejects_symlinked_ancestor(tmp_path):
    real = tmp_path / "actual" / "result"
    real.mkdir(parents=True)
    (real / "metrics.json").write_text('{"example": 1}', encoding="utf-8")
    completion_inventory(real)
    assert validate_inventory(real)
    alias = tmp_path / "shortcut"
    try:
        alias.symlink_to(real.parent, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert not validate_inventory(alias / "result")
    with pytest.raises(ValueError, match="ancestors"):
        completion_inventory(alias / "another")


@pytest.mark.parametrize("script_name", ["compare_optimizers", "tune_optimizers"])
def test_cli_reports_loader_observation_without_identity_authentication(script_name, monkeypatch, capsys):
    module = _script(script_name)
    captured = {}
    monkeypatch.setattr(module, "load_mnist", lambda **_kw: (np.zeros((1, 1)), np.zeros(1)))
    if script_name == "compare_optimizers":
        monkeypatch.setattr(module, "compare", lambda *args, **kwargs: captured.update(kwargs) or {"ok": True})
        argv = ["--gd-step", "0.1", "--linf-step", "0.1", "--coordinate-step", "0.1"]
    else:
        monkeypatch.setattr(
            module,
            "tune_and_evaluate",
            lambda *args, **kwargs: captured.update(kwargs) or {"final_refit_and_test_once_per_method": {}},
        )
        argv = ["--iterations", "1"]
    assert module.main(argv) == 0
    assert captured["source"] == "unverified_array_input"
    assert "acquisition_observation" not in captured
    printed = json.loads(capsys.readouterr().out)
    assert printed["cli_only_acquisition_observation"] == "openml_loader_invoked_by_cli_unverified"


def test_release_excludes_older_audit_logs_and_retains_primary_source():
    inventory = json.loads((ROOT / "release-files.json").read_text(encoding="utf-8"))["files"]
    assert "docs/engineering-history.md" in inventory
    assert "docs/sources/2024-original-report.pdf" in inventory
    assert "docs/sixth-audit-remediation.md" not in inventory
    assert not any("remediation" in name or "re-audit" in name for name in inventory if name.startswith("docs/"))
