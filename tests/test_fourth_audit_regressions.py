"""Adversarial regressions for the v1.3.0 fresh audit (F01-F09, E05)."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import numpy as np
import pytest

from mnist_optimization.artifacts import validate_inventory
from mnist_optimization.experiments import Configuration, run_dataset
from mnist_optimization.logistic import real_matrix
from mnist_optimization.provenance import (
    OPENML_LOADER_SOURCE,
    code_fingerprint,
    hash_arrays,
)
from mnist_optimization.tuning import tune_and_evaluate

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location("audit_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_object_memory_addresses_are_not_hashed(fixture_mnist, tmp_path):
    X, y = fixture_mnist
    obj_labels = np.array([str(int(v)) + "" for v in y], dtype=object)
    with pytest.raises(ValueError, match="Object"):
        hash_arrays(obj_labels)
    config = Configuration(train_per_class=2, test_per_class=2, iterations=1)
    first = run_dataset(X, obj_labels, config, source="synthetic", output=tmp_path / "obj")
    second = run_dataset(X, y, config, source="synthetic", output=tmp_path / "int")
    assert first["data"]["dataset_sha256"] == second["data"]["dataset_sha256"]
    assert first["data"]["prepared_split_sha256"] == second["data"]["prepared_split_sha256"]
    assert validate_inventory(tmp_path / "obj")


def test_numeric_hash_is_portable_across_byte_order():
    values = np.array([0, -5, 2**40], dtype=np.int64)
    swapped = values.astype(">i8")
    assert hash_arrays(values) == hash_arrays(swapped)
    unicode_labels = np.array(["4", "9"], dtype="<U1")
    assert hash_arrays(unicode_labels) == hash_arrays(unicode_labels.astype(">U1"))


def test_public_tuning_and_comparison_cannot_claim_loader(fixture_mnist, tmp_path):
    compare = script("compare_optimizers").compare
    config = Configuration(train_per_class=2, test_per_class=2, iterations=1)
    with pytest.raises(ValueError, match="reserved"):
        compare(
            *fixture_mnist,
            config=config,
            step_sizes={"gd": 0.01, "linf": 1e-7, "coordinate": 0.001},
            output=tmp_path / "forged_comparison",
            source=OPENML_LOADER_SOURCE,
        )
    with pytest.raises(ValueError, match="reserved"):
        tune_and_evaluate(
            *fixture_mnist,
            config=config,
            step_grids={"gd": [0.01], "linf": [1e-7], "coordinate": [0.001]},
            output=tmp_path / "forged_tuning",
            source=OPENML_LOADER_SOURCE,
        )
    for name in ("forged_comparison", "forged_tuning"):
        assert not (tmp_path / name).exists()


def test_zip_alias_and_symlink_metadata_rejected(tmp_path):
    builder, verifier = script("build_release"), script("verify_release")
    archive_path = builder.build_archive(tmp_path / "good.zip", ROOT)
    verifier.verify_archive(archive_path)
    with ZipFile(archive_path) as archive:
        infos = [(info, archive.read(info)) for info in archive.infolist()]
    alias = tmp_path / "alias.zip"
    with ZipFile(alias, "w") as z:
        for info, data in infos:
            z.writestr(info, data)
        z.writestr("mnist-logistic-optimization/./README.md", (ROOT / "README.md").read_bytes())
    with pytest.raises(ValueError, match="noncanonical|unlisted|aliased"):
        verifier.verify_archive(alias)
    symlink = tmp_path / "symlink.zip"
    with ZipFile(symlink, "w") as z:
        for info, data in infos:
            if info.filename.endswith("/README.md"):
                info = ZipInfo(info.filename, date_time=info.date_time)
                info.create_system = 3
                info.compress_type = 8
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, data)
    with pytest.raises(ValueError, match="nonregular"):
        verifier.verify_archive(symlink)


def test_release_allowlist_excludes_untracked_secrets_and_unknown_python(tmp_path):
    builder = script("build_release")
    secret = ROOT / "scripts" / ".env"
    unknown = ROOT / "src" / "mnist_optimization" / "unexpected_experiment.py"
    try:
        secret.write_text("FAKE_DO_NOT_PUBLISH=test", encoding="utf-8")
        unknown.write_text("secret = 'FAKE_DO_NOT_PUBLISH'\n", encoding="utf-8")
        assert secret not in builder.included_files(ROOT)
        assert unknown not in builder.included_files(ROOT)
        output = builder.build_archive(tmp_path / "release.zip", ROOT)
        with ZipFile(output) as archive:
            assert not any(".env" in name or "unexpected_experiment" in name for name in archive.namelist())
            assert "mnist-logistic-optimization/scripts/.env" not in archive.namelist()
    finally:
        secret.unlink(missing_ok=True)
        unknown.unlink(missing_ok=True)


def test_script_changes_affect_source_fingerprint(tmp_path, monkeypatch):
    import mnist_optimization.provenance as p

    project = tmp_path / "project"
    package = project / "src" / "mnist_optimization"
    scripts = project / "scripts"
    package.mkdir(parents=True)
    scripts.mkdir()
    (package / "core.py").write_text("x=1", encoding="utf-8")
    (project / "pyproject.toml").write_text('[project]\nname = "mnist-logistic-optimization"\n', encoding="utf-8")
    script_file = scripts / "run.py"
    script_file.write_text("print(1)", encoding="utf-8")
    monkeypatch.setattr(p, "__file__", str(package / "provenance.py"))
    before = code_fingerprint()
    script_file.write_text("print(2)", encoding="utf-8")
    assert code_fingerprint() != before


def test_release_directory_rejects_unexpected_empty_directory(tmp_path):
    builder, verifier = script("build_release"), script("verify_release")
    archive = builder.build_archive(tmp_path / "release.zip", ROOT)
    with ZipFile(archive) as z:
        z.extractall(tmp_path / "extract")
    root = tmp_path / "extract" / "mnist-logistic-optimization"
    verifier.verify_directory(root)
    (root / "extra_empty").mkdir()
    with pytest.raises(ValueError, match="directory"):
        verifier.verify_directory(root)


@pytest.mark.parametrize("integer", [2**53 + 1, 2**54 + 1, -(2**53 + 1)])
def test_large_inexact_integer_rejected(integer):
    with pytest.raises(ValueError, match="exactly"):
        real_matrix(np.array([integer], dtype=np.int64))


@pytest.mark.parametrize("integer", [2**53, 2**54, -(2**54)])
def test_large_exact_integer_accepted(integer):
    assert real_matrix(np.array([integer], dtype=np.int64))[0] == integer


def test_current_reproduction_checklist_is_consistent():
    text = (ROOT / "docs" / "limitations.md").read_text(encoding="utf-8")
    assert "source identifier is `openml_dataset_554_v1`" not in text
    assert "openml_loader_observed_unverified" in text


def test_standalone_scripts_end_to_end_on_synthetic_npz(tmp_path, fixture_mnist):
    """Execute all four CLIs via subprocess against one synthetic data file.

    These observations are NOT MNIST classification metrics.
    """
    X, y = fixture_mnist
    path = tmp_path / "fixture.npz"
    np.savez_compressed(path, X=X, y=y)
    args = [
        ("reproduce_original.py", ["--train-per-class", "2", "--test-per-class", "2", "--iterations", "2"]),
        ("run_experiments.py", ["--train-per-class", "2", "--test-per-class", "2", "--iterations", "2"]),
        (
            "compare_optimizers.py",
            [
                "--train-per-class",
                "2",
                "--test-per-class",
                "2",
                "--iterations",
                "2",
                "--gd-step",
                "0.0001",
                "--linf-step",
                "0.00000001",
                "--coordinate-step",
                "0.0001",
            ],
        ),
        (
            "tune_optimizers.py",
            [
                "--train-per-class",
                "4",
                "--test-per-class",
                "2",
                "--iterations",
                "2",
                "--gd-grid",
                "0.1",
                "--linf-grid",
                "0.0001",
                "--coordinate-grid",
                "0.5",
            ],
        ),
    ]
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    for i, (name, flags) in enumerate(args):
        output = tmp_path / f"output_{i}"
        command = [
            sys.executable,
            str(ROOT / "scripts" / name),
            "--local-npz",
            str(path),
            "--output",
            str(output),
            *flags,
        ]
        result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=65)
        assert result.returncode == 0, (name, result.stdout[-1000:], result.stderr[-1500:])
        assert validate_inventory(output)
        if name == "tune_optimizers.py":
            payload = json.loads((output / "selection.json").read_text(encoding="utf-8"))
        elif name == "compare_optimizers.py":
            payload = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
        else:
            payload = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        assert payload["source"] == "user_supplied_unverified_array"
