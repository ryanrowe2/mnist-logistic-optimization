"""Regressions for the independent v1.11.0 scientific and metadata audit.

All full-shape MNIST-like fixtures here are deliberately synthetic; no live
MNIST accuracy or source authenticity is inferred from their output.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mnist_optimization.data import prepare_binary
from mnist_optimization.experiments import Configuration, run_prepared, validate_split

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "mnist_optimization"


def _load_copied_provenance(tmp_path: Path, pyproject: str):
    package = tmp_path / "src" / "mnist_optimization"
    package.mkdir(parents=True)
    for source in PACKAGE.glob("*.py"):
        shutil.copy2(source, package / source.name)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "local.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("isolated_twelfth_provenance", package / "provenance.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _own_git_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run([
        "git", "-c", "user.name=QA", "-c", "user.email=qa@example.com",
        "commit", "-qm", "snapshot",
    ], cwd=path, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="git executable unavailable")
@pytest.mark.parametrize("pyproject", [
    '[project]\nname = "unrelated-project"\n',
    '[project]\nname = "another-app"\n[tool.setuptools]\npackages = ["mnist_optimization"]\n',
    '[project]\n# name = "mnist-logistic-optimization"\nname = "unrelated-project"\n',
    '[project\nname = "mnist-logistic-optimization"\n',
])
def test_foreign_or_invalid_pyproject_never_claims_own_git_sha(tmp_path, pyproject):
    module = _load_copied_provenance(tmp_path, pyproject)
    foreign_sha = _own_git_repo(tmp_path)
    evidence = module.environment_fingerprint()
    assert evidence["git_revision"] is None
    assert evidence["source_fingerprint_scope"] == "installed_package_only"
    assert len(foreign_sha) == 40
    # Unrelated project scripts and metadata must not enter the package hash.
    original = module.code_fingerprint()
    (tmp_path / "scripts" / "local.py").write_text("print('changed')\n", encoding="utf-8")
    assert module.code_fingerprint() == original


@pytest.mark.skipif(shutil.which("git") is None, reason="git executable unavailable")
def test_owned_checkout_retains_accurate_revision_and_complete_source_scope(tmp_path):
    module = _load_copied_provenance(tmp_path, '[project]\nname = "mnist-logistic-optimization"\n')
    expected_revision = _own_git_repo(tmp_path)
    before = module.code_fingerprint()
    assert module.environment_fingerprint()["git_revision"] == expected_revision
    assert module.environment_fingerprint()["source_fingerprint_scope"] == "package_scripts_pyproject"
    (tmp_path / "scripts" / "local.py").write_text("print('changed')\n", encoding="utf-8")
    assert module.code_fingerprint() != before


@pytest.mark.parametrize("counts", [(12, 7), (20, 4)])
@pytest.mark.parametrize("seed", [0, 42])
def test_official_reversing_pair_changes_only_label_sign(fixture_mnist, counts, seed):
    X, y = fixture_mnist
    a = prepare_binary(X, y, pair=(4, 9), protocol="official",
                       train_per_class=counts[0], test_per_class=counts[1], seed=seed)
    b = prepare_binary(X, y, pair=(9, 4), protocol="official",
                       train_per_class=counts[0], test_per_class=counts[1], seed=seed)
    np.testing.assert_array_equal(a.train_indices, b.train_indices)
    np.testing.assert_array_equal(a.test_indices, b.test_indices)
    np.testing.assert_array_equal(a.X_train, b.X_train)
    np.testing.assert_array_equal(a.X_test, b.X_test)
    np.testing.assert_array_equal(a.y_train, -b.y_train)
    np.testing.assert_array_equal(a.y_test, -b.y_test)
    assert not np.intersect1d(a.train_indices, a.test_indices).size
    assert np.all(a.train_indices < 60000)
    assert np.all(b.test_indices >= 60000)


def test_official_train_selection_is_independent_of_test_count_for_both_pair_orders(fixture_mnist):
    X, y = fixture_mnist
    training_indices = []
    for pair in ((4, 9), (9, 4)):
        for test_count in (4, 20):
            split = prepare_binary(X, y, protocol="official", pair=pair,
                                   train_per_class=12, test_per_class=test_count, seed=17)
            training_indices.append(split.train_indices)
    for sampled in training_indices[1:]:
        np.testing.assert_array_equal(sampled, training_indices[0])


@pytest.mark.parametrize("protocol", ["historical", "official"])
@pytest.mark.parametrize("target", ["train_indices", "test_indices"])
def test_prepared_split_rejects_multidimensional_source_indices_before_writing(
    fixture_mnist, tmp_path, protocol, target,
):
    cfg = Configuration(protocol=protocol, train_per_class=3,
                        test_per_class=2, iterations=1, record_every=1)
    split = prepare_binary(*fixture_mnist, protocol=protocol,
                           train_per_class=3, test_per_class=2, seed=cfg.seed)
    malformed = replace(split, **{target: getattr(split, target).reshape(-1, 1)})
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_split(malformed, cfg)
    destination = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="one-dimensional"):
        run_prepared(malformed, cfg, output=destination)
    assert not destination.exists()
    validate_split(split, cfg)


def test_maintained_docs_explain_identity_keyed_sampling():
    docs = (ROOT / "docs" / "evaluation-methodology.md").read_text(encoding="utf-8")
    assert "SeedSequence([seed, 0, digit])" in docs
    assert "SeedSequence([seed, 1, digit])" in docs
    assert "Reversing the requested pair" in docs
