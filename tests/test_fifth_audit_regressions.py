"""Regressions for six independent v1.4.0 audit findings (N01--N06)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mnist_optimization.data import prepare_binary, validate_mnist
from mnist_optimization.logistic import (
    logistic_gradient,
    logistic_loss,
    validate_real_parameter,
)
from mnist_optimization.optimizers import optimize

ROOT = Path(__file__).resolve().parents[1]


def _verify_directory():
    spec = importlib.util.spec_from_file_location("verify_release_fifth", ROOT / "scripts/verify_release.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_directory


def test_zero_regularization_does_not_calculate_overflowing_norm():
    X = np.zeros((2, 1))
    y = np.array([-1, 1])
    w = np.array([1e200])
    assert logistic_loss(w, X, y, l2=0.0) == pytest.approx(np.log(2.0))
    np.testing.assert_array_equal(logistic_gradient(w, X, y, l2=0.0), [0.0])
    assert logistic_loss(w, X, y, l2=0.0, bias_index=0) == pytest.approx(np.log(2.0))
    with pytest.raises(FloatingPointError):
        logistic_loss(w, X, y, l2=1.0)
    # The regularized objective remains finite in this controlled case:
    # l2=1e-200 and w=1e200 imply l2*w**2 = 1e200.
    assert logistic_loss(w, X, y, l2=1e-200) == pytest.approx(0.5e200)


def test_inexact_integer_scalar_is_rejected_before_lossy_coercion():
    bad = 2**53 + 1
    for candidate in [bad, np.uint64(bad), np.int64(-bad)]:
        if int(candidate) < 0:
            candidate = abs(candidate)
        with pytest.raises(ValueError, match="exactly"):
            validate_real_parameter(candidate, "step", positive=True)
        with pytest.raises(ValueError, match="exactly"):
            optimize([[1.0], [-1.0]], [1, -1], iterations=1, step_size=candidate)
        with pytest.raises(ValueError, match="exactly"):
            logistic_loss([0.0], [[1.0]], [1], l2=candidate)
    assert validate_real_parameter(2**53 + 2, "step") == float(2**53 + 2)
    assert validate_real_parameter(np.float64(0.25), "step") == 0.25


def test_extended_precision_scalar_conversion_is_checked():
    if np.dtype(np.longdouble).itemsize <= np.dtype(np.float64).itemsize:
        pytest.skip("platform has no extended-precision long double")
    precise = np.longdouble("0.12345678912345678912345")
    with pytest.raises(ValueError, match="exactly"):
        validate_real_parameter(precise, "step", positive=True)
    exact = np.longdouble(0.25)
    assert validate_real_parameter(exact, "step", positive=True) == 0.25


def test_prepare_binary_rejects_fractional_pixels_rejected_by_validate_mnist(fixture_mnist):
    X, y = fixture_mnist
    invalid = X.astype(np.float32)
    invalid[0, 0] = 0.5
    with pytest.raises(ValueError, match="integer intensities"):
        validate_mnist(invalid, y)
    with pytest.raises(ValueError, match="integer intensities"):
        prepare_binary(invalid, y, train_per_class=1, test_per_class=1)


@pytest.mark.parametrize("value", [np.nan, 256.0, -1.0])
def test_prepare_binary_rejects_other_invalid_pixels(fixture_mnist, value):
    X, y = fixture_mnist
    invalid = X.astype(np.float64)
    invalid[0, 0] = value
    with pytest.raises(ValueError, match="pixels|intensities"):
        prepare_binary(invalid, y, train_per_class=1, test_per_class=1)


def test_prepare_binary_canonical_types_match_loader(fixture_mnist):
    X, y = fixture_mnist
    a = prepare_binary(X.astype(np.float32), y.astype(str), train_per_class=1, test_per_class=1)
    b = prepare_binary(X, y, train_per_class=1, test_per_class=1)
    np.testing.assert_array_equal(a.X_train, b.X_train)
    np.testing.assert_array_equal(a.y_test, b.y_test)


def test_release_verifier_rejects_symlinked_root(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "MANIFEST.sha256").write_text("", encoding="utf-8")
    linked = tmp_path / "project-link"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symlink"):
        _verify_directory()(linked)


def test_source_line_length_matches_ruff_configuration():
    for directory in ("src", "scripts", "tests"):
        for path in (ROOT / directory).rglob("*.py"):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                assert len(line) <= 120, f"E501 candidate: {path}:{number} ({len(line)})"


def test_single_current_version_banner_and_referenced_register():
    reproducibility = (ROOT / "docs/reproducibility.md").read_text(encoding="utf-8")
    assert "release-status.md" in reproducibility
    assert "Current release note (v1.3.0)" not in reproducibility
    assert (ROOT / "docs/engineering-history.md").is_file()
