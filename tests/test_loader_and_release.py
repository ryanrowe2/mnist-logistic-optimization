"""Offline OpenML-contract tests and opt-in live-data release gate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mnist_optimization.artifacts import validate_inventory
from mnist_optimization.data import load_mnist
from mnist_optimization.experiments import Configuration, main, run_dataset
from mnist_optimization.provenance import OPENML_LOADER_SOURCE, hash_arrays


def test_loader_uses_exact_openml_id_554_but_never_certifies_array_identity(monkeypatch, fixture_mnist, tmp_path):
    import sklearn.datasets

    captured = {}

    def fake_fetch(**kwargs):
        captured.update(kwargs)
        return fixture_mnist

    monkeypatch.setattr(sklearn.datasets, "fetch_openml", fake_fetch)
    X, y = load_mnist(data_home=tmp_path / "cache")
    assert captured == {
        "data_id": 554,
        "return_X_y": True,
        "as_frame": False,
        "parser": "auto",
        "data_home": str(tmp_path / "cache"),
    }
    assert hash_arrays(X, y) == hash_arrays(fixture_mnist[0], fixture_mnist[1].astype(np.int64))
    # Even a mocked response from the documented transport is not a canonical hash certificate.
    report = run_dataset(
        X,
        y,
        Configuration(train_per_class=2, test_per_class=2, iterations=2),
        source="synthetic_mock_fetch_response",
        output=tmp_path / "mocked_result",
    )
    assert report["data"]["canonical_mnist_assertion"] is False
    assert report["data"]["independent_dataset_identity_verified"] is False
    assert validate_inventory(tmp_path / "mocked_result")
    with pytest.raises(ValueError, match="reserved"):
        run_dataset(
            X,
            y,
            Configuration(train_per_class=2, test_per_class=2, iterations=2),
            source=OPENML_LOADER_SOURCE,
            output=tmp_path / "forgery",
        )
    # CLI acquisition can record the route, but never claims independent upstream verification.
    assert (
        main(
            [
                "--train-per-class",
                "2",
                "--test-per-class",
                "2",
                "--iterations",
                "2",
                "--data-home",
                str(tmp_path / "cache"),
                "--output",
                str(tmp_path / "cli_result"),
            ]
        )
        == 0
    )
    observed = json.loads((tmp_path / "cli_result" / "metrics.json").read_text())
    assert observed["source"] == "unverified_array_input"
    assert observed["data"]["selection_protocol_validated"] is True
    assert observed["data"]["canonical_mnist_assertion"] is False


def test_loader_refuses_wrong_shape_and_failed_download(monkeypatch):
    import sklearn.datasets

    monkeypatch.setattr(sklearn.datasets, "fetch_openml", lambda **kwargs: (np.ones((1, 784)), [1]))
    with pytest.raises(ValueError, match="70000"):
        load_mnist()

    def failed(**kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(sklearn.datasets, "fetch_openml", failed)
    with pytest.raises(RuntimeError, match="Unable to fetch OpenML"):
        load_mnist()


@pytest.mark.parametrize(
    "script", ["reproduce_original.py", "run_experiments.py", "compare_optimizers.py", "tune_optimizers.py"]
)
def test_all_script_help_commands(script):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / script), "--help"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--local-npz" in result.stdout


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("MNIST_LIVE_TEST") != "1",
    reason="Opt in with MNIST_LIVE_TEST=1 (network and OpenML download required)",
)
def test_live_openml_smoke_and_source_identity(tmp_path):
    # Real-data smoke exercises the genuine loader via the CLI, not caller-issued tokens.
    from mnist_optimization.experiments import main

    assert (
        main(
            [
                "--protocol",
                "official",
                "--train-per-class",
                "16",
                "--test-per-class",
                "8",
                "--iterations",
                "5",
                "--step-size",
                "0.1",
                "--record-every",
                "1",
                "--output",
                str(tmp_path / "real_data"),
            ]
        )
        == 0
    )
    report = json.loads((tmp_path / "real_data" / "metrics.json").read_text())
    assert report["data"]["canonical_mnist_assertion"] is False
    assert report["source"] == "unverified_array_input"
    assert report["data"]["selection_protocol_validated"] is True
    assert report["evaluation"]["count"] == 16
    assert validate_inventory(tmp_path / "real_data")
    assert (tmp_path / "real_data" / "figures" / "examples.png").exists()
