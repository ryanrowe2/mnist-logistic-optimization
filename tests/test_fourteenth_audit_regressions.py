"""Adversarial numerical, artifact-safety and lock-completeness regressions."""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from mnist_optimization.data import BinarySplit
from mnist_optimization.experiments import Configuration, validate_split
from mnist_optimization.logistic import logistic_gradient, logistic_loss, scores

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_wheel(folder: Path, name: str, version: str = "1.0", requires=()):
    """Create a minimal genuine installable wheel, including PEP 427 RECORD hashes."""
    folder.mkdir(parents=True, exist_ok=True)
    filename = folder / f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
    dist = f"{name.replace('-', '_')}-{version}.dist-info"
    files = {
        f"{name.replace('-', '_')}/__init__.py": b"version = 'fixture'\n",
        f"{dist}/METADATA": (f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
                              + "".join(f"Requires-Dist: {req}\n" for req in requires) + "\n").encode(),
        f"{dist}/WHEEL": b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    rows = []
    for path, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        rows.append([path, f"sha256={digest}", str(len(data))])
    rows.append([f"{dist}/RECORD", "", ""])
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    files[f"{dist}/RECORD"] = buf.getvalue().encode()
    with zipfile.ZipFile(filename, "w") as archive:
        for path, data in files.items():
            archive.writestr(path, data)
    return filename


def minimal_project(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "pyproject.toml").write_text('''[project]
name = "fixture-model"
version = "0.1"
dependencies = ["fake-core>=1"]
[project.optional-dependencies]
dev = []
''')
    (root / "requirements-validation.txt").write_text("")
    return root


def complete_fake_wheelhouse(tmp_path):
    wheelhouse = tmp_path / "wheels"
    for name, version in (("fake-core", "1.2.3"), ("packaging", "25.0"),
                          ("wheel", "0.45.1"), ("ruff", "0.8.0")):
        fake_wheel(wheelhouse, name, version)
    return wheelhouse


def test_a01_list_digit_pair_matches_canonical_split():
    config = Configuration(digits=[4, 9], train_per_class=1, test_per_class=1)
    assert config.digits == (4, 9)
    split = BinarySplit(np.zeros((2, 2)), np.array([-1, 1]), np.zeros((2, 2)),
                        np.array([-1, 1]), np.array([3, 4]), np.array([5, 6]),
                        (4, 9), "historical", "raw_0_255", 2024, None)
    validate_split(split, config)


def test_a02_mean_of_representable_extreme_losses():
    X = np.full((4, 1), 1e308)
    assert logistic_loss([1.0], X, [-1] * 4) == pytest.approx(1e308)


def test_a03_extreme_representable_gradient():
    X = np.full((4, 1), 1e308)
    assert logistic_gradient([1.0], X, [-1] * 4)[0] == pytest.approx(1e308)


def test_a04_representable_regularization():
    value = logistic_loss([1.2e154, 1.2e154], np.zeros((2, 2)), [1, -1], l2=1)
    assert value == pytest.approx(1.44e308)


def test_a05_finite_cancelling_products():
    X = np.array([[1e308, 1e308], [1e308, 1e308]])
    assert np.array_equal(scores(X, [2., -2.]), [0, 0])
    assert logistic_loss([2., -2.], X, [1, -1]) == pytest.approx(np.log(2))
    assert np.allclose(logistic_gradient([2., -2.], X, [1, -1]), 0)
    with pytest.raises(FloatingPointError):
        scores(X, [2., 2.])


def test_a06_lock_output_symlink_rejected_without_touching_canary(tmp_path):
    lock = script("create_platform_lock")
    wheelhouse = complete_fake_wheelhouse(tmp_path)
    canary = tmp_path / "canary"
    canary.write_text("SAFE")
    link = tmp_path / "lock.txt"
    link.symlink_to(canary)
    with pytest.raises(ValueError, match="symlink"):
        lock.create_lock(wheelhouse, link, project_root=minimal_project(tmp_path))
    assert canary.read_text() == "SAFE"


def test_a07_external_wheel_symlink_rejected(tmp_path):
    lock = script("create_platform_lock")
    outside = fake_wheel(tmp_path, "fake-core")
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    (wheelhouse / outside.name).symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        lock.create_lock(wheelhouse, tmp_path / "lock.txt", project_root=minimal_project(tmp_path))


def test_a08_reject_missing_required_wheels(tmp_path):
    lock = script("create_platform_lock")
    wheelhouse = tmp_path / "wheels"
    fake_wheel(wheelhouse, "irrelevant")
    with pytest.raises(ValueError, match="missing required dependency"):
        lock.create_lock(wheelhouse, tmp_path / "lock.txt", project_root=minimal_project(tmp_path))
    assert not (tmp_path / "lock.txt").exists()


def test_a09_reject_fake_wheel_even_with_expected_name(tmp_path):
    lock = script("create_platform_lock")
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    (wheelhouse / "fake_core-1.2.3-py3-none-any.whl").write_bytes(b"fake")
    with pytest.raises(ValueError, match="malformed wheel"):
        lock.create_lock(wheelhouse, tmp_path / "lock.txt", project_root=minimal_project(tmp_path))


def test_wheelhouse_allows_canonical_directory_records(tmp_path):
    wheelhouse = complete_fake_wheelhouse(tmp_path)
    with zipfile.ZipFile(wheelhouse / "fake_core-1.2.3-py3-none-any.whl", "a") as archive:
        archive.writestr("fake_core/", b"")
    output = script("create_platform_lock").create_lock(
        wheelhouse, tmp_path / "lock.txt", project_root=minimal_project(tmp_path)
    )
    assert "fake-core==1.2.3" in output.read_text()


def test_lock_graph_conflicts_and_installable_fixture(tmp_path):
    lock = script("create_platform_lock")
    wheelhouse = complete_fake_wheelhouse(tmp_path)
    project = minimal_project(tmp_path)
    target = lock.create_lock(wheelhouse, tmp_path / "lock.txt", project_root=project)
    assert "fake-core==1.2.3" in target.read_text()
    install = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                              "--require-hashes", "--find-links", str(wheelhouse), "--target",
                              str(tmp_path / "installed"), "-r", str(target)],
                             capture_output=True, text=True, timeout=45)
    assert install.returncode == 0, install.stderr
    fake_wheel(wheelhouse, "fake-core", "2.0")
    with pytest.raises(ValueError, match="conflicting"):
        lock.create_lock(wheelhouse, tmp_path / "invalid.txt", project_root=project)


@pytest.mark.parametrize("entry", ["results/synthetic_validation.json", "figures/synthetic_descent_comparison.png"])
def test_a10_synthetic_outputs_do_not_follow_symlinks(tmp_path, entry):
    project = tmp_path / "fixture"
    (project / "scripts").mkdir(parents=True)
    (project / "figures").mkdir()
    (project / "results").mkdir()
    (project / "scripts/generate_synthetic_validation.py").write_bytes(
        (ROOT / "scripts/generate_synthetic_validation.py").read_bytes())
    canary = tmp_path / "canary.txt"
    canary.write_text("PRESERVE")
    (project / entry).symlink_to(canary)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, str(project / "scripts/generate_synthetic_validation.py")],
                          env=env, capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0
    assert "symlink" in proc.stderr
    assert canary.read_text() == "PRESERVE"
