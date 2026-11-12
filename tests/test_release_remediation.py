"""Regression tests for the 13-finding independent release audit."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from mnist_optimization.artifacts import validate_inventory
from mnist_optimization.data import BinarySplit, prepare_binary, validate_mnist
from mnist_optimization.evaluation import metrics, wilson_interval
from mnist_optimization.experiments import Configuration, run_prepared
from mnist_optimization.figures import save_figures
from mnist_optimization.logistic import logistic_gradient, logistic_loss, validate_data
from mnist_optimization.optimizers import OptimizationResult, descent_direction, optimize
from mnist_optimization.tuning import validate_grids

ROOT = Path(__file__).resolve().parents[1]


def import_script(name):
    location = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, location)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_linf_rejects_gradient_sum_overflow_without_returning_inf():
    with pytest.raises(FloatingPointError, match="direction overflow"):
        descent_direction(np.array([1e308, 1e308]), "linf")


@pytest.mark.parametrize("bad", [True, np.bool_(True), 1j, "0.1", np.nan, np.inf])
def test_public_optimizer_rejects_invalid_parameter_types(bad):
    X, y = [[1.0], [-1.0]], [1, -1]
    for field in ("step_size", "l2"):
        with pytest.raises(ValueError):
            optimize(X, y, iterations=1, **{field: bad})
    for fun in (logistic_loss, logistic_gradient):
        with pytest.raises(ValueError):
            fun([0.0], X, y, l2=bad)


def test_grid_bool_is_invalid():
    with pytest.raises(ValueError):
        validate_grids({"gd": [True], "linf": [0.1], "coordinate": [0.1]})
    with pytest.raises(ValueError):
        import_script("compare_optimizers").validate_steps({"gd": True, "linf": 0.1, "coordinate": 0.1})


@pytest.mark.parametrize("bad", [[True, True], [1 + 0j, -1 + 0j], ["1", "-1"]])
def test_training_labels_cannot_be_lossily_coerced(bad):
    with pytest.raises(ValueError):
        validate_data([[1.0], [0.0]], bad)


def test_digit_labels_and_arguments_are_strict(fixture_mnist):
    X, labels = fixture_mnist
    for bad_label in (labels.astype(bool), labels.astype(complex), labels.astype(object)):
        with pytest.raises(ValueError):
            validate_mnist(X, bad_label)
    for option in ({"pair": (True, 9)}, {"train_per_class": True}, {"test_per_class": False}, {"seed": True}):
        with pytest.raises(ValueError):
            prepare_binary(X, labels, **option)


def test_bad_official_prepared_indices_rejected(tmp_path):
    split = BinarySplit(
        np.ones((2, 785)),
        np.array([-1, 1]),
        np.ones((2, 785)),
        np.array([-1, 1]),
        np.array([10, 11]),
        np.array([20, 21]),
        (4, 9),
        "official",
        "fixed_divide_255",
        42,
        784,
    )
    config = Configuration(
        protocol="official", train_per_class=1, test_per_class=1, seed=42, iterations=1, step_size=0.05
    )
    with pytest.raises(ValueError, match="boundaries"):
        run_prepared(split, config, output=tmp_path / "bad", source="fixture")
    assert not (tmp_path / "bad").exists()


def test_bad_official_prepared_intercept_rejected(tmp_path):
    X = np.zeros((2, 785))
    X[:, -1] = [1, 0]
    split = BinarySplit(
        X,
        np.array([-1, 1]),
        np.ones((2, 785)),
        np.array([-1, 1]),
        np.array([100, 101]),
        np.array([60000, 60001]),
        (4, 9),
        "official",
        "fixed_divide_255",
        42,
        784,
    )
    with pytest.raises(ValueError, match="intercepts"):
        run_prepared(
            split,
            Configuration(protocol="official", train_per_class=1, test_per_class=1, seed=42, iterations=1),
            output=tmp_path / "bad",
            source="fixture",
        )


def test_prepared_data_does_not_certify_original_selection(tmp_path):
    train = np.array([[1.0, 0.0], [-1.0, 0.0]])
    split = BinarySplit(
        train,
        np.array([1, -1]),
        train.copy(),
        np.array([1, -1]),
        np.array([100, 101]),
        np.array([102, 103]),
        (4, 9),
        "historical",
        "raw_0_255",
        42,
        None,
    )
    report = run_prepared(
        split,
        Configuration(train_per_class=1, test_per_class=1, seed=42, iterations=1),
        output=tmp_path / "prepared",
        source="fixture",
    )
    assert report["data"]["protocol"] == "user_supplied_prepared_unverified"
    assert report["data"]["requested_protocol"] == "historical"
    assert report["data"]["selection_protocol_validated"] is False


def test_invalid_inventory_is_not_an_exception(tmp_path):
    folder = tmp_path / "report"
    folder.mkdir()
    path = folder / "_SUCCESS.json"
    for invalid in ("NOT JSON", "[]", "null", "{}", '{"file_sha256": []}'):
        path.write_text(invalid)
        assert validate_inventory(folder) is False


def test_fixed_image_scale_for_both_pixel_protocols(monkeypatch, tmp_path):
    from matplotlib.axes import Axes

    vmax_values = []
    old_imshow = Axes.imshow

    def track(self, *args, **kwargs):
        if kwargs.get("cmap") == "gray":
            vmax_values.append(kwargs.get("vmax"))
        return old_imshow(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", track)
    for columns, expected in ((784, 255), (785, 1)):
        X = np.zeros((2, columns))
        if columns == 785:
            X[:, -1] = 1
        result = OptimizationResult(
            weights=np.zeros(columns),
            iterations=np.array([0, 1]),
            losses=np.array([0.69, 0.68]),
            update_counts=np.zeros(columns, dtype=int),
            method="gd",
        )
        before = len(vmax_values)
        save_figures(
            result,
            X,
            np.array([-1, 1]),
            np.array([-1, 1]),
            tmp_path / str(columns),
            pair=(4, 9),
            provenance="synthetic",
        )
        assert vmax_values[before:] == [expected, expected]


def test_wilson_confidence_intervals_correct_and_boundary_safe():
    high = wilson_interval(10, 10)
    low = wilson_interval(0, 10)
    assert high["lower"] == pytest.approx(0.7224672, abs=1e-6)
    assert high["upper"] == pytest.approx(1.0)
    assert low["lower"] == pytest.approx(0.0)
    assert low["upper"] == pytest.approx(1 - high["lower"], abs=1e-6)
    m = metrics([-1, -1, 1, 1], [-1, -1, -1, 1])
    assert m["accuracy_wilson_95_interval"]["lower"] < 0.75 < m["accuracy_wilson_95_interval"]["upper"]
    assert m["recall_wilson_95_by_class"]["negative"]["upper"] == 1
    for s, n in ((-1, 4), (5, 3), (True, 3)):
        with pytest.raises(ValueError):
            wilson_interval(s, n)


def test_actual_release_zip_extracts_builds_installs_and_detects_tampering(tmp_path):
    builder = import_script("build_release")
    verifier = import_script("verify_release")
    zip_path = builder.build_archive(tmp_path / "release.zip", root=ROOT)
    verifier.verify_archive(zip_path)
    with ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        archive.extractall(tmp_path / "extracted")
    project = tmp_path / "extracted" / "mnist-logistic-optimization"
    assert (project / "pyproject.toml").is_file()
    assert (project / "src/mnist_optimization/logistic.py").is_file()
    assert "mnt/data" not in "\n".join(n for n in ZipFile(zip_path).namelist())
    verifier.verify_directory(project)
    if sys.platform.startswith("linux") and subprocess.run(["which", "sha256sum"], capture_output=True).returncode == 0:
        result = subprocess.run(
            ["sha256sum", "-c", "MANIFEST.sha256"], cwd=project, capture_output=True, text=True, timeout=20
        )
        assert result.returncode == 0, result.stderr
    readme = project / "README.md"
    original_readme = readme.read_bytes()
    readme.write_text("MODIFIED AFTER RELEASE", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        verifier.verify_directory(project)
    readme.write_bytes(original_readme)
    verifier.verify_directory(project)
    wheels = tmp_path / "wheels"
    command = [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "-w", str(wheels)]
    wheel = subprocess.run(command, cwd=project, capture_output=True, text=True, timeout=90)
    assert wheel.returncode == 0, wheel.stderr[-2000:]
    artifacts = list(wheels.glob("*.whl"))
    assert len(artifacts) == 1
    installed = tmp_path / "installed"
    installation = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(installed), str(artifacts[0])],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert installation.returncode == 0, installation.stderr[-2000:]
    env = dict(os.environ, PYTHONPATH=str(installed))
    smoke = subprocess.run(
        [sys.executable, "-m", "mnist_optimization.experiments", "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert smoke.returncode == 0 and "--protocol" in smoke.stdout
    executable = installed / "bin" / "mnist-optimize"
    if executable.exists():
        direct = subprocess.run(
            [str(executable), "--help"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
        )
        assert direct.returncode == 0 and "--protocol" in direct.stdout
    assert list(project.rglob("pyproject.toml")) == [project / "pyproject.toml"]


def test_complete_live_real_mnist_script_exposes_entrypoint():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_real_mnist.py"), "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--output-root" in result.stdout


def test_reference_lock_generation_rejects_non_wheel_bytes(tmp_path):
    generator = import_script("create_platform_lock")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / "fake_core-1.2.3-py3-none-any.whl").write_bytes(b"not a ZIP wheel")
    with pytest.raises(ValueError, match="malformed wheel"):
        generator.create_lock(wheels, tmp_path / "platform.lock")
    assert not (tmp_path / "platform.lock").exists()
