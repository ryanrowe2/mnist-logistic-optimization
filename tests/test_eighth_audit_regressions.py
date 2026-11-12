"""Regressions for v1.7 independent findings: safe staging and acquisition scope."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from mnist_optimization.artifacts import check_output_available, stage_output
from mnist_optimization.experiments import Configuration
from mnist_optimization.tuning import tune_and_evaluate

ROOT = Path(__file__).resolve().parents[1]


def compare_script():
    spec = importlib.util.spec_from_file_location("eighth_compare", ROOT / "scripts/compare_optimizers.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable on this filesystem")


def test_staging_rejects_symlinked_output_root_without_writing_target(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    make_symlink(alias, real)
    with pytest.raises(ValueError, match="symlinks"):
        check_output_available(alias)
    with pytest.raises(ValueError, match="symlinks"):
        with stage_output(alias) as stage:
            (stage / "proof.txt").write_text("should not appear", encoding="utf-8")
    assert list(real.iterdir()) == []
    assert not (real / "_SUCCESS.json").exists()


def test_staging_rejects_symlinked_ancestor_without_creating_destination(tmp_path):
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    alias = tmp_path / "alias_parent"
    make_symlink(alias, real_parent)
    with pytest.raises(ValueError, match="symlinks"):
        with stage_output(alias / "new_output") as stage:
            (stage / "proof.txt").write_text("unexpected", encoding="utf-8")
    assert not (real_parent / "new_output").exists()
    assert list(real_parent.iterdir()) == []


def test_staging_rejects_dangling_symlink(tmp_path):
    alias = tmp_path / "dangling"
    make_symlink(alias, tmp_path / "does-not-exist")
    with pytest.raises(ValueError, match="symlinks"):
        with stage_output(alias):
            pass
    assert alias.is_symlink()
    assert not (tmp_path / "does-not-exist").exists()


def test_staging_rejects_parent_traversal(tmp_path):
    with pytest.raises(ValueError, match="traversal"):
        with stage_output(tmp_path / "placeholder" / ".." / "result"):
            pass
    assert not (tmp_path / "result").exists()


def test_staging_rechecks_after_user_code_before_commit(tmp_path):
    output = tmp_path / "destination"
    target = tmp_path / "unrelated"
    target.mkdir()
    with pytest.raises(ValueError, match="symlinks"):
        with stage_output(output) as stage:
            (stage / "proof.txt").write_text("test-only", encoding="utf-8")
            make_symlink(output, target)
    assert output.is_symlink()
    assert list(target.iterdir()) == []
    assert not list(tmp_path.glob(".destination-incomplete-*"))


def test_observation_cannot_be_supplied_through_public_array_apis(tmp_path):
    bogus = "openml_loader_invoked_by_cli_unverified"
    config = Configuration(train_per_class=2, test_per_class=1, iterations=1)
    X, y = np.empty((0, 784)), np.empty(0)
    with pytest.raises(TypeError, match="acquisition_observation"):
        tune_and_evaluate(X, y, config=config, output=tmp_path / "tune", step_grids={}, acquisition_observation=bogus)
    with pytest.raises(TypeError, match="acquisition_observation"):
        compare_script().compare(
            X,
            y,
            config=config,
            step_sizes={},
            output=tmp_path / "compare",
            source="synthetic",
            acquisition_observation=bogus,
        )
    assert not (tmp_path / "tune").exists()
    assert not (tmp_path / "compare").exists()


def test_array_api_reports_fixed_acquisition_state(fixture_mnist, tmp_path):
    cfg = Configuration(protocol="official", train_per_class=3, test_per_class=2, iterations=1, record_every=1)
    grids = {name: [0.0001] for name in ("gd", "linf", "coordinate")}
    report = tune_and_evaluate(
        *fixture_mnist, config=cfg, step_grids=grids, output=tmp_path / "tune", source="synthetic_fixture"
    )
    assert report["acquisition_observation"] == "not_observed_by_array_api"
    assert not report["acquisition_observation_is_authenticated"]
    saved = json.loads((tmp_path / "tune/selection.json").read_text(encoding="utf-8"))
    assert saved["acquisition_observation"] == "not_observed_by_array_api"
    cmp = compare_script().compare(
        *fixture_mnist,
        config=cfg,
        step_sizes={key: 0.0001 for key in grids},
        output=tmp_path / "comparison",
        source="synthetic_fixture",
    )
    assert cmp["acquisition_observation"] == "not_observed_by_array_api"
    assert (
        json.loads((tmp_path / "comparison/comparison.json").read_text())["acquisition_observation"]
        == "not_observed_by_array_api"
    )


def test_methodology_documents_identity_keyed_streams():
    docs = (ROOT / "docs/evaluation-methodology.md").read_text(encoding="utf-8")
    assert "SeedSequence([seed, 0, digit])" in docs
    assert "SeedSequence([seed, 1, digit])" in docs
    assert "[seed, 2, low_digit, high_digit]" in docs
    assert "[seed, 3, low_digit, high_digit]" in docs
    assert "Reversing the requested pair" in docs
    assert "test sample counts does not change training" in docs
