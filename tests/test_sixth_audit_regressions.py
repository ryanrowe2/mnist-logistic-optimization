"""Independent regressions for seven v1.5.0 findings and ZIP resource limits."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pytest

from mnist_optimization.experiments import Configuration, validate_configuration
from mnist_optimization.logistic import add_intercept, predict, scores
from mnist_optimization.tuning import train_validation_indices

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "mnist-logistic-optimization"


def script(name: str):
    spec = importlib.util.spec_from_file_location(f"sixth_{name}", ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tamper_archive(source: Path, target: Path, *, add: dict[str, bytes] | None = None,
                   remove: set[str] | None = None) -> None:
    """Build a self-consistent but not allowlisted ZIP to exercise independent verification."""
    with ZipFile(source) as archive:
        content = {info.filename[len(PROJECT) + 1:]: archive.read(info)
                   for info in archive.infolist()}
    content.update(add or {})
    for name in remove or set():
        content.pop(name)
    content["MANIFEST.sha256"] = ("\n".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}"
        for name, data in sorted(content.items()) if name != "MANIFEST.sha256"
    ) + "\n").encode()
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in sorted(content.items()):
            info = ZipInfo(f"{PROJECT}/{name}", (2026, 11, 11, 12, 0, 0))
            info.create_system = 3
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compress_type=ZIP_DEFLATED)


def test_manifest_alone_does_not_authorize_new_secret_or_missing_test(tmp_path):
    builder, verifier = script("build_release"), script("verify_release")
    original = builder.build_archive(tmp_path / "original.zip", ROOT)
    verifier.verify_archive(original)
    secret = tmp_path / "rogue.zip"
    tamper_archive(original, secret, add={"scripts/.env": b"PASSWORD=TEST_ONLY"})
    with pytest.raises(ValueError, match="allowlist"):
        verifier.verify_archive(secret)
    missing = tmp_path / "missing.zip"
    tamper_archive(original, missing, remove={"tests/test_gradient.py"})
    with pytest.raises(ValueError, match="allowlist"):
        verifier.verify_archive(missing)


def test_release_list_is_shared_and_includes_itself():
    builder, verifier = script("build_release"), script("verify_release")
    contents = json.loads((ROOT / "release-files.json").read_text())
    assert contents["schema"] == 1
    assert set(builder.RELEASE_FILES) == verifier.expected_release_files() == set(contents["files"])
    assert {"release-files.json", "docs/release-status.md", "tests/test_gradient.py"} <= set(contents["files"])
    assert "scripts/.env" not in contents["files"]


@pytest.mark.parametrize("parameter", [2**53 + 1, np.uint64(2**53 + 1)])
@pytest.mark.parametrize("field", ["step_size", "l2"])
def test_config_rejects_values_optimizer_cannot_represent(field, parameter):
    with pytest.raises(ValueError, match="exactly"):
        validate_configuration(Configuration(**{field: parameter}))


def test_extended_precision_configuration_is_checked():
    if np.dtype(np.longdouble).itemsize <= np.dtype(np.float64).itemsize:
        pytest.skip("no extended precision")
    with pytest.raises(ValueError, match="exactly"):
        validate_configuration(Configuration(step_size=np.longdouble("0.123456789123456789")))


def test_symlinked_parent_rejected_before_traversal(tmp_path):
    verifier = script("verify_release")
    real = tmp_path / "real-parent" / PROJECT
    real.mkdir(parents=True)
    linked = tmp_path / "alias-parent"
    try:
        linked.symlink_to(real.parent, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available")
    with pytest.raises(ValueError, match="ancestor"):
        verifier.verify_directory(linked / PROJECT)


def test_gitignore_covers_generated_mnist_outputs(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git executable unavailable")
    (tmp_path / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for path in ("results/full_real_validation/full_real_mnist_summary.json",
                 "results/arbitrary/output/metrics.json", "real_mnist_validation/results.json"):
        result = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=tmp_path)
        assert result.returncode == 0, path
    result = subprocess.run(["git", "check-ignore", "--no-index", "-q",
                             "results/synthetic_validation.json"], cwd=tmp_path)
    assert result.returncode == 1


@pytest.mark.parametrize("shape", [(0, 2), (3, 0)])
def test_numerical_helpers_reject_empty_samples_and_features(shape):
    X = np.empty(shape)
    w = np.ones(shape[1])
    with pytest.raises(ValueError, match="nonempty"):
        scores(X, w)
    with pytest.raises(ValueError, match="nonempty"):
        predict(X, w)
    with pytest.raises(ValueError, match="nonempty"):
        add_intercept(X)


@pytest.mark.parametrize("bad", ["0.2", True, np.bool_(True), None, 1j, np.nan, np.inf, -0.1, 1.0])
def test_fraction_invalid_inputs_raise_valueerror(bad):
    with pytest.raises(ValueError):
        train_validation_indices(np.array([-1, -1, 1, 1]), seed=2024, fraction=bad)


def test_fraction_strict_validation_accepts_regular_input():
    tr, va = train_validation_indices(np.array([-1, -1, 1, 1]), seed=2024, fraction=0.2)
    assert len(tr) == len(va) == 2


def test_single_maintained_release_status():
    current = ROOT / "docs/release-status.md"
    assert current.is_file()
    text = current.read_text(encoding="utf-8")
    assert "1.8.0" in text
    for other in ("reproducibility.md", "limitations.md", "engineering-history.md"):
        historical = (ROOT / "docs" / other).read_text(encoding="utf-8")
        assert "release-status.md" in historical


def test_archive_ratio_limit_rejects_compressible_unlisted_content(tmp_path):
    verifier = script("verify_release")
    huge = tmp_path / "compressed.zip"
    info = ZipInfo(f"{PROJECT}/README.md", (2026, 11, 11, 12, 0, 0))
    info.create_system = 3
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    with ZipFile(huge, "w") as archive:
        archive.writestr(info, b"A" * 2_000_000, compress_type=ZIP_DEFLATED)
    with pytest.raises(ValueError, match="expansion limits"):
        verifier.verify_archive(huge)
