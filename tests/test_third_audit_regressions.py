"""Adversarial tests for the independent v1.2.0 audit's six reproducible defects."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from mnist_optimization import __version__
from mnist_optimization.artifacts import completion_inventory, validate_inventory
from mnist_optimization.evaluation import wilson_interval
from mnist_optimization.experiments import Configuration, run_dataset
from mnist_optimization.figures import save_figures
from mnist_optimization.optimizers import OptimizationResult
from mnist_optimization.provenance import OPENML_LOADER_SOURCE

ROOT = Path(__file__).resolve().parents[1]


def test_forged_loader_certification_api_has_been_removed(fixture_mnist, tmp_path):
    import mnist_optimization.provenance as provenance

    assert not hasattr(provenance, "_issue_loader_evidence")
    assert not hasattr(provenance, "is_attested")
    assert not hasattr(provenance, "LoaderEvidence")
    with pytest.raises(ValueError, match="reserved"):
        run_dataset(
            *fixture_mnist,
            Configuration(train_per_class=2, test_per_class=2, iterations=2),
            output=tmp_path / "forged",
            source=OPENML_LOADER_SOURCE,
        )
    assert not (tmp_path / "forged").exists()
    result = run_dataset(
        *fixture_mnist,
        Configuration(train_per_class=2, test_per_class=2, iterations=2),
        output=tmp_path / "caller_data",
        source="synthetic_fixture",
    )
    assert result["data"]["canonical_mnist_assertion"] is False
    assert result["data"]["independent_dataset_identity_verified"] is False


def make_completed(folder: Path) -> None:
    (folder / "nested").mkdir(parents=True)
    (folder / "nested" / "data.txt").write_text("real data", encoding="utf-8")
    completion_inventory(folder)
    assert validate_inventory(folder)


def test_nested_completion_filename_cannot_escape_hash_inventory(tmp_path):
    directory = tmp_path / "results"
    make_completed(directory)
    (directory / "nested" / "_SUCCESS.json").write_text('{"fake":1}')
    assert not validate_inventory(directory)
    (directory / "nested" / "_SUCCESS.json").unlink()
    assert validate_inventory(directory)
    (directory / "nested" / "_SUCCESS.json").write_text("nested manifest before build")
    (directory / "_SUCCESS.json").unlink()
    report = completion_inventory(directory)
    assert "nested/_SUCCESS.json" in report["file_sha256"]
    assert validate_inventory(directory)
    (directory / "nested" / "_SUCCESS.json").write_text("tampered after build")
    assert not validate_inventory(directory)


def test_inventory_rejects_symlinked_directory_file_and_empty_dir(tmp_path):
    target = tmp_path / "safe"
    make_completed(target)
    try:
        (target / "escape").symlink_to(tmp_path, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on filesystem")
    assert not validate_inventory(target)
    (target / "escape").unlink()
    assert validate_inventory(target)
    (target / "data_link").symlink_to(target / "nested" / "data.txt")
    assert not validate_inventory(target)
    (target / "data_link").unlink()
    assert validate_inventory(target)
    (target / "unexpected").mkdir()
    assert not validate_inventory(target)


def test_package_version_matches_distribution_definition():
    config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"$', config)
    assert match is not None
    assert __version__ == match.group(1)
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", __version__)


def test_wilson_interval_handles_largest_representable_confidence_below_one():
    almost_one = np.nextafter(1.0, 0.0)
    for successes in (0, 5, 10):
        result = wilson_interval(successes, 10, confidence=float(almost_one))
        assert result["confidence"] == almost_one
        assert 0 <= result["lower"] <= result["upper"] <= 1
    with pytest.raises(ValueError):
        wilson_interval(5, 10, confidence=1.0)


@pytest.mark.parametrize(
    "values", [np.ones(784), -np.ones(784), np.zeros(784), np.concatenate((np.array([-3.0]), np.ones(783)))]
)
def test_coefficient_color_scale_is_symmetric_about_zero(monkeypatch, tmp_path, values):
    from matplotlib.axes import Axes

    observed = []
    old = Axes.imshow

    def inspect(self, *args, **kwargs):
        if kwargs.get("cmap") == "coolwarm":
            observed.append((kwargs.get("vmin"), kwargs.get("vmax")))
        return old(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", inspect)
    result = OptimizationResult(
        weights=values,
        iterations=np.array([0, 1]),
        losses=np.array([0.693, 0.68]),
        update_counts=np.zeros(784, dtype=int),
        method="gd",
    )
    save_figures(
        result,
        np.zeros((2, 784)),
        np.array([-1, 1]),
        np.array([-1, 1]),
        tmp_path / "fig",
        pair=(4, 9),
        provenance="synthetic controlled",
    )
    assert len(observed) == 1
    vmin, vmax = observed[0]
    assert np.isfinite(vmin) and np.isfinite(vmax) and vmin < 0 < vmax
    assert vmin == -vmax
    if np.any(values):
        assert vmax == np.max(np.abs(values))
