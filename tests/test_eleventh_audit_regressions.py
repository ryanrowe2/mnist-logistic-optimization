"""Regression evidence for v1.11.0 independent integrity findings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mnist_optimization.artifacts import completion_inventory, validate_inventory
from mnist_optimization.data import BinarySplit
from mnist_optimization.evaluation import metrics
from mnist_optimization.experiments import Configuration, run_prepared
from mnist_optimization.provenance import source_label


def _small_split() -> BinarySplit:
    train = np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]])
    test = np.array([[2.0, 0.0], [-2.0, 0.0]])
    return BinarySplit(
        train,
        np.array([1, 1, -1, -1]),
        test,
        np.array([1, -1]),
        np.arange(4),
        np.arange(4, 6),
        (4, 9),
        "historical",
        "raw_0_255",
        42,
        None,
    )


@pytest.mark.parametrize(
    "claimed_source",
    [
        "Verified OpenML MNIST data_id=554 (canonical upstream identity confirmed)",
        "iNsPeCtEd oPeNmL dAtA",
        "CANONICAL source verified",
        "AUTHENTICATED dataset",
    ],
)
def test_caller_note_cannot_be_authoritative_source(claimed_source, tmp_path):
    cfg = Configuration(train_per_class=2, test_per_class=1, iterations=2, seed=42)
    target = tmp_path / "results"
    report = run_prepared(_small_split(), cfg, output=target, source=claimed_source)
    assert report["source"] == "user_supplied_unverified_prepared_split"
    assert report["user_supplied_source_note_unverified"] == claimed_source
    assert report["data"]["canonical_mnist_assertion"] is False
    for plot in ("convergence.png", "confusion.png"):
        assert (target / "figures" / plot).is_file()
    assert validate_inventory(target)


@pytest.mark.parametrize("label", ["fixture", "synthetic_test_fixture"])
def test_known_unverified_categories_remain_safe(label):
    assert source_label(label) == label


def test_file_transport_cannot_be_claimed_by_array_api():
    assert source_label("user_supplied_unverified_npz") == "user_supplied_unverified_array"
    assert source_label("user_supplied_unverified_prepared_split") == "user_supplied_unverified_array"
    assert source_label("fixture", prepared=True) == "user_supplied_unverified_prepared_split"


def test_unknown_source_text_is_never_used_as_figure_provenance(tmp_path, monkeypatch):
    seen = []
    from mnist_optimization import experiments

    real_save_figures = experiments.save_figures

    def capture(*args, **kwargs):
        seen.append(kwargs["provenance"])
        return real_save_figures(*args, **kwargs)

    monkeypatch.setattr(experiments, "save_figures", capture)
    cfg = Configuration(train_per_class=2, test_per_class=1, iterations=1, seed=42)
    run_prepared(_small_split(), cfg, source="OpenML: I certify this dataset", output=tmp_path / "data")
    assert seen == ["user_supplied_unverified_prepared_split"]


@pytest.mark.parametrize(
    "alter",
    [
        lambda text: text.replace('"status": "complete"', '"status": "incomplete", "status": "complete"'),
        lambda text: text.replace('"schema_version": 2', '"schema_version": 1, "schema_version": 2'),
        lambda text: text.replace('"directories": []', '"directories": [], "directories": []'),
    ],
)
def test_duplicate_json_keys_in_completion_marker_fail_closed(tmp_path, alter):
    root = tmp_path / "complete"
    root.mkdir()
    (root / "report.json").write_text('{"ok": true}', encoding="utf-8")
    completion_inventory(root)
    marker = root / "_SUCCESS.json"
    original = marker.read_text(encoding="utf-8")
    mutated = alter(original)
    assert mutated != original
    marker.write_text(mutated, encoding="utf-8")
    assert validate_inventory(root) is False


def test_duplicate_nested_inventory_hash_key_rejected(tmp_path):
    root = tmp_path / "complete"
    root.mkdir()
    (root / "report.json").write_text("truth", encoding="utf-8")
    completion_inventory(root)
    marker = root / "_SUCCESS.json"
    text = marker.read_text(encoding="utf-8")
    digest = json.loads(text)["file_sha256"]["report.json"]
    text = text.replace(f'"report.json": "{digest}"', f'"report.json": "{digest}", "report.json": "{digest}"')
    marker.write_text(text, encoding="utf-8")
    assert validate_inventory(root) is False


def test_unexpected_completion_schema_key_and_bool_version_rejected(tmp_path):
    root = tmp_path / "complete"
    root.mkdir()
    (root / "example.txt").write_text("valid", encoding="utf-8")
    completion_inventory(root)
    marker = root / "_SUCCESS.json"
    baseline = json.loads(marker.read_text(encoding="utf-8"))
    for changed in ({**baseline, "unknown": 1}, {**baseline, "schema_version": True}):
        marker.write_text(json.dumps(changed), encoding="utf-8")
        assert not validate_inventory(root)


def test_training_prediction_evidence_matches_report(tmp_path):
    cfg = Configuration(train_per_class=2, test_per_class=1, iterations=8, seed=42)
    report = run_prepared(_small_split(), cfg, output=tmp_path / "observations")
    with np.load(tmp_path / "observations" / "numeric_results.npz", allow_pickle=False) as values:
        assert {"y_train", "y_pred_train", "y_test", "y_pred_test"} <= set(values.files)
        assert metrics(values["y_train"], values["y_pred_train"]) == report["training"]
        assert metrics(values["y_test"], values["y_pred_test"]) == report["evaluation"]
    assert validate_inventory(tmp_path / "observations")


def test_source_code_matches_ruffs_known_import_requirements():
    root = Path(__file__).resolve().parents[1]
    assert "from collections.abc import Iterator" in (root / "src/mnist_optimization/artifacts.py").read_text()
    assert "from zipfile import ZIP_DEFLATED, ZipFile" in (root / "scripts/verify_release.py").read_text()
