"""Independent adversarial regressions for the comprehensive v1.12.0 audit."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mnist_optimization.data import prepare_binary
from mnist_optimization.experiments import Configuration, validate_split
from mnist_optimization.logistic import real_matrix
from mnist_optimization.provenance import hash_arrays, source_label
from mnist_optimization.tuning import train_validation_indices

ROOT = Path(__file__).resolve().parents[1]


def import_script(name: str):
    location = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"audit13_{name}", location)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_copy(folder: Path):
    package = folder / "src" / "mnist_optimization"
    package.mkdir(parents=True)
    shutil.copytree(ROOT / "src" / "mnist_optimization", package, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (folder / "scripts").mkdir()
    (folder / "scripts" / "smoke.py").write_text("pass\n", encoding="utf-8")
    (folder / "pyproject.toml").write_text('[project]\nname="mnist-logistic-optimization"\n', encoding="utf-8")
    spec = importlib.util.spec_from_file_location("audit13_provenance", package / "provenance.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root: Path, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def test_tuning_is_invariant_to_label_reversal():
    y = np.r_[np.full(300, -1), np.full(300, 1)]
    train, val = train_validation_indices(y, seed=2025, fraction=.2)
    reversed_train, reversed_val = train_validation_indices(-y, seed=2025, fraction=.2)
    np.testing.assert_array_equal(train, reversed_train)
    np.testing.assert_array_equal(val, reversed_val)
    assert len(val) == 120
    assert not np.intersect1d(train, val).size


@pytest.mark.parametrize("invalid_seed", [True, np.bool_(False), -1, .5, np.int64(9),
                                           np.random.default_rng(10), np.random.SeedSequence(10)])
def test_tuning_rejects_mutable_invalid_or_ambiguous_seed(invalid_seed):
    with pytest.raises(ValueError, match="seed"):
        train_validation_indices(np.array([-1, -1, 1, 1]), seed=invalid_seed)


@pytest.mark.parametrize("labels", [np.array([-1+0j, -1+0j, 1+0j, 1+0j]),
                                    np.array([-1, -1, 1, 1], dtype=object),
                                    np.array([True, True, True, False])])
def test_tuning_rejects_nonreal_or_noncanonical_labels(labels):
    with pytest.raises(ValueError, match="labels"):
        train_validation_indices(labels, seed=42)


def test_list_pair_normalizes_to_tuple_for_both_protocols(fixture_mnist):
    for protocol in ("historical", "official"):
        split = prepare_binary(*fixture_mnist, pair=[4, 9], protocol=protocol,
                               train_per_class=3, test_per_class=2, seed=2024)
        assert split.digit_pair == (4, 9)
        assert type(split.digit_pair) is tuple
        validate_split(split, Configuration(protocol=protocol, train_per_class=3,
                                            test_per_class=2, iterations=1))


def test_longdouble_matrix_never_silently_rounds():
    if np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps:
        pytest.skip("no wider numpy floating type available")
    precise = np.nextafter(np.longdouble(1), np.longdouble(2))
    with pytest.raises(ValueError, match="extended-precision"):
        real_matrix(np.array([[precise]], dtype=np.longdouble))
    np.testing.assert_array_equal(real_matrix(np.array([[np.longdouble(1)]], dtype=np.longdouble)), [[1.]])


def test_longdouble_hash_is_padding_independent():
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip("long-double padding not used on this architecture")
    original = np.array([np.longdouble(1.)])
    altered = original.copy()
    # Change only padding, not any of the 80 significant x87 float bits.
    if np.dtype(np.longdouble).itemsize == 16 and np.little_endian:
        altered.view(np.uint8)[10:16] = [1, 2, 3, 4, 5, 6]
        assert original[0] == altered[0]
    assert hash_arrays(original) == hash_arrays(altered)
    if np.finfo(np.longdouble).eps < np.finfo(np.float64).eps:
        precise = np.nextafter(np.longdouble(1), np.longdouble(2))
        with pytest.raises(ValueError, match="exactly representable"):
            hash_arrays(np.array([precise], dtype=np.longdouble))


def test_array_source_labels_cannot_claim_transport():
    assert source_label("user_supplied_unverified_npz") == "user_supplied_unverified_array"
    assert source_label("user_supplied_unverified_prepared_split") == "user_supplied_unverified_array"
    assert source_label("user_supplied_unverified_npz", prepared=True) == "user_supplied_unverified_prepared_split"


@pytest.mark.skipif(shutil.which("git") is None, reason="git executable not available")
def test_untracked_project_source_has_no_certified_git_revision(tmp_path):
    module = source_copy(tmp_path)
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "pyproject.toml")
    git(tmp_path, "-c", "user.name=QA", "-c", "user.email=audit@example.com", "commit", "-qm", "meta")
    evidence = module.environment_fingerprint()
    assert evidence["git_revision"] is None
    assert evidence["git_revision_state"] == "untracked_source"
    assert evidence["git_executed_sources_tracked"] is False


@pytest.mark.skipif(shutil.which("git") is None, reason="git executable not available")
def test_dirty_project_source_has_no_certified_git_revision(tmp_path):
    module = source_copy(tmp_path)
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "pyproject.toml", "scripts", "src")
    git(tmp_path, "-c", "user.name=QA", "-c", "user.email=audit@example.com", "commit", "-qm", "source")
    clean = module.environment_fingerprint()
    assert clean["git_revision"] == git(tmp_path, "rev-parse", "HEAD")
    assert clean["git_revision_state"] == "clean"
    modified = tmp_path / "src/mnist_optimization/logistic.py"
    modified.write_text(modified.read_text() + "\n# deliberately dirty\n")
    dirty = module.environment_fingerprint()
    assert dirty["git_revision"] is None
    assert dirty["git_revision_state"] == "dirty_source"
    assert dirty["source_sha256"] != clean["source_sha256"]


def test_builder_ignores_preexisting_stage_symlink_and_cannot_overwrite_canary(tmp_path):
    builder = import_script("build_release")
    output = tmp_path / "release.zip"
    canary = tmp_path / "important.txt"
    canary.write_bytes(b"UNCHANGED-EXTERNAL-FILE")
    old_stage = output.with_suffix(output.suffix + ".incomplete")
    try:
        old_stage.symlink_to(canary)
    except (OSError, NotImplementedError):
        pytest.skip("filesystem does not support symlinks")
    builder.build_archive(output, ROOT)
    assert canary.read_bytes() == b"UNCHANGED-EXTERNAL-FILE"
    assert old_stage.is_symlink()
    assert output.is_file() and output.read_bytes()[:2] == b"PK"


def test_builder_rejects_manifest_symlink_without_overwriting_canary(tmp_path):
    builder = import_script("build_release")
    folder = tmp_path / "copy"
    folder.mkdir()
    # Copy only allowlisted files. Keep the original checkout untouched.
    for relative in builder.RELEASE_FILES:
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    canary = tmp_path / "important.txt"
    canary.write_bytes(b"UNCHANGED-EXTERNAL-FILE")
    try:
        (folder / "MANIFEST.sha256").symlink_to(canary)
    except (OSError, NotImplementedError):
        pytest.skip("filesystem does not support symlinks")
    with pytest.raises(ValueError, match="symlink"):
        builder.build_archive(tmp_path / "release.zip", folder)
    assert canary.read_bytes() == b"UNCHANGED-EXTERNAL-FILE"


def test_builder_rejects_link_as_final_output_without_overwriting_target(tmp_path):
    builder = import_script("build_release")
    output = tmp_path / "release.zip"
    target = tmp_path / "important.txt"
    target.write_text("UNCHANGED")
    try:
        output.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("filesystem does not support symlinks")
    with pytest.raises(ValueError, match="symlink"):
        builder.build_archive(output, ROOT)
    assert target.read_text() == "UNCHANGED"


def test_mnist_preflight_rejects_occupied_output_before_network(tmp_path, monkeypatch):
    verifier = import_script("verify_real_mnist")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "prior.json").write_text("preserve")
    observed = []

    def network_sentinel(**kwargs):
        observed.append(True)
        raise RuntimeError("network unexpectedly invoked")

    monkeypatch.setattr(verifier, "load_mnist", network_sentinel)
    with pytest.raises(FileExistsError):
        verifier.main(["--output-root", str(occupied)])
    assert not observed
    assert (occupied / "prior.json").read_text() == "preserve"


def test_configured_ruff_b905_calls_are_explicit():
    import ast
    for file in [*(ROOT / "src").rglob("*.py"), *(ROOT / "tests").rglob("*.py"),
                 *(ROOT / "scripts").rglob("*.py")]:
        code = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for node in ast.walk(code):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "zip"
                    and len(node.args) >= 2):
                assert any(kw.arg == "strict" for kw in node.keywords), f"{file}:{node.lineno}"
