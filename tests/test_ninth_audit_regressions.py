"""Independent regressions for v1.9 provenance and stationary-coordinate defects."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mnist_optimization.optimizers import descent_direction, optimize


SOURCE = Path(__file__).resolve().parents[1]
PACKAGE = SOURCE / "src" / "mnist_optimization"


def load_provenance(package_dir: Path):
    spec = importlib.util.spec_from_file_location("isolated_mnist_provenance", package_dir / "provenance.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_style_package(root: Path) -> Path:
    folder = root / "mnist_optimization"
    folder.mkdir(parents=True)
    for source in PACKAGE.glob("*.py"):
        shutil.copy2(source, folder / source.name)
    return folder


def test_installed_code_hash_is_independent_of_installation_path(tmp_path):
    first = load_provenance(install_style_package(tmp_path / "siteA"))
    second = load_provenance(install_style_package(tmp_path / "siteB"))
    assert first.code_fingerprint() == second.code_fingerprint()
    assert first.environment_fingerprint()["source_fingerprint_scope"] == "installed_package_only"
    assert second.environment_fingerprint()["source_fingerprint_scope"] == "installed_package_only"
    assert first.environment_fingerprint()["git_revision"] is None
    assert second.environment_fingerprint()["git_revision"] is None
    # Changed source in one install must affect its digest.
    (tmp_path / "siteB" / "mnist_optimization" / "optimizers.py").write_text("# mutated\n")
    assert first.code_fingerprint() != second.code_fingerprint()


def checkout_style_module(parent: Path):
    root = parent / "mnist-logistic-optimization"
    folder = root / "src" / "mnist_optimization"
    folder.mkdir(parents=True)
    for source in PACKAGE.glob("*.py"):
        shutil.copy2(source, folder / source.name)
    (root / "scripts").mkdir()
    (root / "scripts" / "smoke.py").write_text("pass\n")
    (root / "pyproject.toml").write_text('[project]\nname="mnist-logistic-optimization"\n')
    return root, load_provenance(folder)


def git(*args, cwd: Path):
    return subprocess.run(["git", *args], cwd=cwd, check=True, text=True,
                          capture_output=True).stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="git executable unavailable")
def test_foreign_parent_git_commit_is_not_attributed_to_project(tmp_path):
    enclosing = tmp_path / "unrelated"
    enclosing.mkdir()
    git("init", "-q", cwd=enclosing)
    (enclosing / "outside.txt").write_text("foreign content")
    git("add", "outside.txt", cwd=enclosing)
    git("-c", "user.name=QA", "-c", "user.email=qa@example.com", "commit", "-qm", "foreign", cwd=enclosing)
    unrelated_sha = git("rev-parse", "HEAD", cwd=enclosing)
    _, module = checkout_style_module(enclosing)
    evidence = module.environment_fingerprint()
    assert evidence["git_revision"] is None
    assert evidence["source_fingerprint_scope"] == "package_scripts_pyproject"
    assert unrelated_sha is not None


@pytest.mark.skipif(shutil.which("git") is None, reason="git executable unavailable")
def test_own_repository_commit_is_reported(tmp_path):
    root, module = checkout_style_module(tmp_path)
    git("init", "-q", cwd=root)
    git("add", ".", cwd=root)
    git("-c", "user.name=QA", "-c", "user.email=qa@example.com", "commit", "-qm", "project", cwd=root)
    own_sha = git("rev-parse", "HEAD", cwd=root)
    assert module.environment_fingerprint()["git_revision"] == own_sha


def test_source_hash_captures_entry_point_changes(tmp_path):
    root, module = checkout_style_module(tmp_path)
    original = module.code_fingerprint()
    (root / "scripts" / "smoke.py").write_text("print('changed')\n")
    assert original != module.code_fingerprint()


def test_stationary_coordinate_gradient_does_not_claim_updates():
    direction, index = descent_direction(np.zeros(2), "coordinate")
    assert index is None
    np.testing.assert_array_equal(direction, [0.0, 0.0])
    result = optimize(np.zeros((4, 2)), [-1, 1, -1, 1], method="coordinate",
                      iterations=7, record_every=1, step_size=0.1)
    assert result.update_counts.tolist() == [0, 0]
    assert result.selection_counts.tolist() == [0, 0]
    np.testing.assert_array_equal(result.weights, [0, 0])
    assert np.all(result.losses == result.losses[0])


def test_rounded_noop_step_counts_selection_but_not_effective_update():
    result = optimize(np.ones((4, 1)), -np.ones(4), method="coordinate", iterations=5,
                      initial_weights=np.array([1e16]), step_size=0.1, record_every=1)
    assert result.selection_counts.tolist() == [5]
    assert result.update_counts.tolist() == [0]
    np.testing.assert_array_equal(result.weights, [1e16])


def test_nonzero_coordinate_steps_count_both_selection_and_update():
    result = optimize(np.ones((4, 1)), -np.ones(4), method="coordinate", iterations=5,
                      step_size=0.1, record_every=1)
    assert result.selection_counts.tolist() == [5]
    assert result.update_counts.tolist() == [5]


def test_stationary_coordinate_counts_are_truthful_in_saved_artifacts(tmp_path):
    """Saved metrics, NPZ and heatmap all describe effective updates."""
    import json

    from mnist_optimization.data import BinarySplit
    from mnist_optimization.experiments import Configuration, run_prepared

    X = np.zeros((4, 784), dtype=float)
    y = np.array([-1, -1, 1, 1])
    split = BinarySplit(
        X, y, X.copy(), y.copy(), np.arange(4), np.arange(4, 8),
        (4, 9), "historical", "raw_0_255", 42, None,
    )
    config = Configuration(train_per_class=2, test_per_class=2, seed=42,
                           method="coordinate", iterations=7, step_size=0.1)
    report = run_prepared(split, config, output=tmp_path / "stationary", source="synthetic_test_fixture")
    assert report["optimization"]["coordinate_selection_total"] == 0
    assert report["optimization"]["coordinate_update_total"] == 0
    stored = json.loads((tmp_path / "stationary" / "metrics.json").read_text())
    assert stored["optimization"] == report["optimization"]
    with np.load(tmp_path / "stationary" / "numeric_results.npz") as data:
        assert data["selection_counts"].sum() == 0
        assert data["update_counts"].sum() == 0
    assert (tmp_path / "stationary" / "figures" / "coordinate_updates.png").stat().st_size > 1000
