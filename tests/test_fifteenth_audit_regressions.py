"""Consolidated v1.15 numerical and wheel-security adversarial regressions."""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from mnist_optimization.logistic import logistic_gradient, logistic_loss, predict, scores

ROOT = Path(__file__).resolve().parents[1]


def locker():
    spec = importlib.util.spec_from_file_location("create_platform_lock", ROOT / "scripts/create_platform_lock.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def wheel(folder: Path, name: str, *, version: str = "1.0", build: str = "",
          requires=(), internal_tag: str = "py3-none-any", additional: dict[str, bytes] | None = None):
    folder.mkdir(exist_ok=True, parents=True)
    safe_name = name.replace("-", "_")
    stem = f"{safe_name}-{version}"
    filename = folder / f"{stem}-{build + '-' if build else ''}py3-none-any.whl"
    dist = f"{stem}.dist-info"
    files = {
        f"{safe_name}/__init__.py": b"fixture = True\n",
        f"{dist}/METADATA": (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            + "".join(f"Requires-Dist: {value}\n" for value in requires) + "\n"
        ).encode(),
        f"{dist}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: regression\nRoot-Is-Purelib: true\n"
            f"Tag: {internal_tag}\n"
        ).encode(),
    }
    files.update(additional or {})
    rows = []
    for path, payload in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
        rows.append((path, "sha256=" + digest, str(len(payload))))
    rows.append((f"{dist}/RECORD", "", ""))
    text = io.StringIO()
    csv.writer(text, lineterminator="\n").writerows(rows)
    files[f"{dist}/RECORD"] = text.getvalue().encode()
    with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path, payload in files.items():
            z.writestr(path, payload)
    return filename


def project(folder: Path, deps: list[str], build: bool = False):
    folder.mkdir(exist_ok=True, parents=True)
    top = ('[build-system]\nrequires = ["setuptools>=68", "wheel"]\n'
           'build-backend = "setuptools.build_meta"\n') if build else ""
    quoted = ", ".join(repr(d) for d in deps)
    (folder / "pyproject.toml").write_text(top + f'''[project]
name = "local-test-model"
version = "1.0"
dependencies = [{quoted}]
[project.optional-dependencies]
dev = []
''')
    return folder


def test_n01_cancellation_preserves_finite_score_loss_and_label():
    X = [[1e16, 1.0, -1e16]]
    w = [1.0, 1.0, 1.0]
    np.testing.assert_array_equal(scores(X, w), [1.0])
    assert logistic_loss(w, X, [1]) == pytest.approx(np.logaddexp(0, -1))
    assert predict(X, w).tolist() == [1]
    np.testing.assert_array_equal(scores([[1e16, -1e16]], [1, 1]), [0.0])


def test_n02_exact_nonzero_gradient_under_finite_cancellation():
    X = [[1e16], [1], [-1e16]]
    assert logistic_gradient([0], X, [1, 1, 1])[0] == pytest.approx(-1 / 6)
    # A second, rearranged case checks vectorized coordinate-wise reduction.
    X2 = [[1e16, 2.0], [3.0, 4.0], [-1e16, -2.0]]
    np.testing.assert_allclose(logistic_gradient([0, 0], X2, [1, 1, 1]), [-0.5, -2 / 3])


def test_n01_representable_cancelling_float64_products():
    X = [[1e308, 1e308], [1e308, 1e308]]
    assert scores(X, [2, -2]).tolist() == [0, 0]
    with pytest.raises(FloatingPointError):
        scores(X, [2, 2])


def test_n03_reject_conflicting_wheel_variant_dependencies(tmp_path):
    root, store = project(tmp_path / "proj", ["fake-core>=1"]), tmp_path / "wheels"
    wheel(store, "fake-core", build="1", requires=["first-dep>=1"])
    wheel(store, "fake-core", build="2", requires=["second-dep>=1"])
    with pytest.raises(ValueError, match="conflicting wheel dependency metadata"):
        locker().create_lock(store, tmp_path / "lock.txt", project_root=root)
    assert not (tmp_path / "lock.txt").exists()


def test_n04_reject_unavailable_requested_extra(tmp_path):
    root, store = project(tmp_path / "proj", ["fake-core[extra1]>=1"]), tmp_path / "wheels"
    wheel(store, "fake-core", requires=['missing-extra>=1; extra == "extra1"'])
    with pytest.raises(ValueError, match="missing required dependency missing-extra"):
        locker().create_lock(store, tmp_path / "lock.txt", project_root=root)


def test_n04_transitive_extras_are_followed(tmp_path):
    root, store = project(tmp_path / "proj", ["fake-core>=1"]), tmp_path / "wheels"
    wheel(store, "fake-core", requires=["nested[feature]>=1"])
    wheel(store, "nested", requires=['missing-extra>=1; extra == "feature"'])
    with pytest.raises(ValueError, match="missing required dependency missing-extra"):
        locker().create_lock(store, tmp_path / "lock.txt", project_root=root)


def test_n05_reject_mismatched_internal_wheel_tag(tmp_path):
    root, store = project(tmp_path / "proj", ["fake-core>=1"]), tmp_path / "wheels"
    wheel(store, "fake-core", internal_tag="cp999-cp999-linux_aarch64")
    with pytest.raises(ValueError, match="internal tags"):
        locker().create_lock(store, tmp_path / "lock.txt", project_root=root)


@pytest.mark.parametrize("alias", ["fake_core/./bad.py", "fake_core//bad.py", "fake_core/../bad.py"])
def test_n06_reject_noncanonical_wheel_paths(tmp_path, alias):
    root, store = project(tmp_path / "proj", ["fake-core>=1"]), tmp_path / "wheels"
    wheel(store, "fake-core", additional={alias: b"unsafe"})
    with pytest.raises(ValueError, match="noncanonical|unsafe"):
        locker().create_lock(store, tmp_path / "lock.txt", project_root=root)


def test_n07_build_backend_requirements_in_lock(tmp_path):
    root, store = project(tmp_path / "proj", ["fake-core>=1"], build=True), tmp_path / "wheels"
    for name in ("fake-core", "wheel", "packaging", "ruff", "setuptools"):
        wheel(store, name, version=("80" if name == "setuptools" else "0.8.6" if name == "ruff" else "1"))
    text = locker().create_lock(store, tmp_path / "lock.txt", project_root=root).read_text()
    assert "setuptools==80" in text
    (store / "setuptools-80-py3-none-any.whl").unlink()
    with pytest.raises(ValueError, match="missing required dependency setuptools"):
        locker().create_lock(store, tmp_path / "no-setuptools.txt", project_root=root)


def test_n09_declared_wheel_member_expansion_is_bounded(tmp_path, monkeypatch):
    root, store = project(tmp_path / "proj", ["fake-core>=1"]), tmp_path / "wheels"
    wheel(store, "fake-core", additional={"fake_core/large.bin": b"x" * 1024})
    mod = locker()
    monkeypatch.setattr(mod, "MAX_MEMBER_BYTES", 512)
    with pytest.raises(ValueError, match="limit|large"):
        mod.create_lock(store, tmp_path / "lock.txt", project_root=root)


def test_n03_matching_build_variants_can_share_a_lock(tmp_path):
    root, store = project(tmp_path / "proj", ["fake-core>=1"]), tmp_path / "wheels"
    wheel(store, "fake-core", build="1")
    wheel(store, "fake-core", build="2")
    for name, version in (("wheel", "1"), ("packaging", "25"), ("ruff", "0.8.6")):
        wheel(store, name, version=version)
    lock = locker().create_lock(store, tmp_path / "lock.txt", project_root=root).read_text()
    assert lock.count("--hash=sha256:") >= 2
    assert lock.count("fake-core==1") == 1
