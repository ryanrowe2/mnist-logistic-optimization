"""Regression evidence for v1.9.0 independent audit findings.

Synthetic arrays test contracts only; they are not MNIST results.
"""
from __future__ import annotations

import json
import numpy as np
import pytest

from mnist_optimization.artifacts import validate_inventory
from mnist_optimization.data import BinarySplit, validate_mnist
from mnist_optimization.evaluation import wilson_interval
from mnist_optimization.experiments import Configuration, _execute, run_dataset, run_prepared
from mnist_optimization.provenance import OPENML_LOADER_SOURCE, hash_arrays


def small_split() -> BinarySplit:
    return BinarySplit(
        X_train=np.array([[2., 1.], [1., 0.], [-2., 1.], [-1., 0.]]),
        y_train=np.array([1, 1, -1, -1]),
        X_test=np.array([[1.5, 1.], [-1.5, 1.]]),
        y_test=np.array([1, -1]),
        train_indices=np.arange(4), test_indices=np.arange(4, 6),
        digit_pair=(4, 9), protocol="historical", scaling="raw_0_255", seed=42,
        bias_index=None,
    )


@pytest.mark.parametrize("parameters", [
    {"step_size": np.int64(1)},
    {"step_size": np.uint32(1)},
    {"step_size": np.int32(1)},
    {"l2": np.int32(0)},
    {"l2": np.uint64(1)},
])
def test_supported_numpy_integer_configuration_roundtrips_to_json(tmp_path, parameters):
    config = Configuration(
        train_per_class=2, test_per_class=1, seed=42, iterations=2,
        record_every=1, **parameters,
    )
    assert type(config.step_size) is float
    assert type(config.l2) is float
    destination = tmp_path / "experiment"
    report = run_prepared(small_split(), config, output=destination, source="synthetic_fixture")
    assert validate_inventory(destination)
    saved = json.loads((destination / "metrics.json").read_text())
    assert saved["config"] == json.loads(json.dumps(report["config"]))
    assert type(saved["config"]["step_size"]) is float
    assert type(saved["config"]["l2"]) is float


def test_numpy_integer_outside_float64_exact_range_rejected(tmp_path):
    config = Configuration(step_size=np.uint64(2**53 + 1), iterations=1,
                           train_per_class=2, test_per_class=1, seed=42)
    with pytest.raises(ValueError, match="exactly"):
        run_prepared(small_split(), config, output=tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_low_level_runner_cannot_issue_openml_observation_or_fake_digest(tmp_path):
    config = Configuration(train_per_class=2, test_per_class=1, seed=42, iterations=1)
    with pytest.raises(ValueError, match="reserved"):
        _execute(small_split(), config, output=tmp_path / "openml", source=OPENML_LOADER_SOURCE)
    with pytest.raises(TypeError, match="dataset_sha256"):
        _execute(small_split(), config, output=tmp_path / "hash", source="synthetic",
                 dataset_sha256="f" * 64)
    with pytest.raises(TypeError, match="selection_validated"):
        _execute(small_split(), config, output=tmp_path / "flag", source="synthetic",
                 selection_validated=True)
    assert list(tmp_path.iterdir()) == []


def test_private_runner_cannot_claim_full_dataset_for_unrelated_split(fixture_mnist, tmp_path):
    config = Configuration(train_per_class=2, test_per_class=1, seed=42, iterations=1)
    target = tmp_path / "forged"
    with pytest.raises(ValueError, match="supplied full dataset"):
        _execute(small_split(), config, output=target, source="synthetic",
                 full_dataset=fixture_mnist)
    assert not target.exists()


def test_full_dataset_selection_claim_is_reconstructed_and_unverified(fixture_mnist, tmp_path):
    config = Configuration(train_per_class=2, test_per_class=1, iterations=2)
    features, labels = fixture_mnist
    report = run_dataset(features, labels, config, output=tmp_path / "full", source="synthetic_fixture")
    assert report["data"]["selection_protocol_validated"] is True
    assert report["data"]["dataset_sha256"] == hash_arrays(*validate_mnist(features, labels))
    assert report["data"]["canonical_mnist_assertion"] is False
    assert report["data"]["independent_dataset_identity_verified"] is False
    assert "acquisition route" in report["data"]["provenance_note"]
    assert OPENML_LOADER_SOURCE not in json.dumps(report)
    assert validate_inventory(tmp_path / "full")


@pytest.mark.parametrize("confidence", [np.float16(0.95), np.float32(0.95), np.float64(0.95)])
def test_wilson_numpy_float_confidence_returns_finite_json_scalars(confidence):
    result = wilson_interval(5, 10, confidence=confidence)
    assert result["confidence"] == float(confidence)
    assert 0 <= result["lower"] <= result["upper"] <= 1
    assert all(type(value) is float for key, value in result.items() if key != "method")
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bad", [np.bool_(True), np.float32(np.nan), np.float32(1),
                                  np.float16(0), np.complex64(0.95)])
def test_wilson_numpy_invalid_confidence_is_rejected(bad):
    with pytest.raises(ValueError, match="confidence"):
        wilson_interval(5, 10, confidence=bad)
