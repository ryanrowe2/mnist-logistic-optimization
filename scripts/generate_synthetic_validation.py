"""Reproduce a real, visibly SYNTHETIC optimizer demonstration (not MNIST)."""

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mnist_optimization.logistic import logistic_gradient, logistic_loss
from mnist_optimization.optimizers import optimize


def _guard(path: Path) -> None:
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError(f"symlinked synthetic output is forbidden: {path}")


def _publish(path: Path, content: bytes) -> None:
    _guard(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _guard(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".mnist-synthetic-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        _guard(path)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    figpath = root / "figures/synthetic_descent_comparison.png"
    resultpath = root / "results/synthetic_validation.json"
    _guard(figpath)
    _guard(resultpath)
    X = np.array([
        [3.0, 1.0], [1.0, 2.0], [-1.0, -3.0],
        [-3.0, -1.0], [1.0, -0.25], [-1.0, 0.5],
    ])
    y = np.array([1, 1, -1, -1, 1, -1])
    methods = ("gd", "linf", "coordinate")
    rows = {}
    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    for method in methods:
        result = optimize(X, y, method=method, step_size=0.02, iterations=250, record_every=1)
        rows[method] = {
            "initial_objective": float(result.losses[0]),
            "final_objective": float(result.losses[-1]),
            "iterations": 250,
            "step_size": 0.02,
            "final_weights": result.weights.tolist(),
            "coordinate_updates": result.update_counts.tolist(),
            "loss_decreased": bool(result.losses[-1] < result.losses[0]),
        }
        if not rows[method]["loss_decreased"]:
            raise AssertionError(f"expected loss reduction for {method}")
        ax.plot(result.iterations, result.losses, lw=1.75, label=method)
    ax.set(
        xlabel="Completed updates", ylabel="Mean logistic loss",
        title="Synthetic 2-feature logistic loss (NOT MNIST)",
    )
    ax.grid(alpha=0.25)
    ax.legend(title="NumPy direction")
    fig.tight_layout()
    image_buffer = io.BytesIO()
    fig.savefig(image_buffer, format="png", dpi=170)
    plt.close(fig)
    # Numerical derivative check at a nonzero point.
    weights = np.array([0.2, -0.3])
    epsilon = 1e-6
    exact = logistic_gradient(weights, X, y)
    numeric = np.array([
        (
            logistic_loss(weights + epsilon * np.eye(2)[j], X, y)
            - logistic_loss(weights - epsilon * np.eye(2)[j], X, y)
        ) / (2 * epsilon)
        for j in range(2)
    ])
    difference = float(np.max(np.abs(exact - numeric)))
    if difference > 1e-7:
        raise AssertionError(f"finite difference gradient check failed: {difference}")
    report = {
        "provenance": "synthetic_two_feature_validation_not_mnist",
        "purpose": "Numerical and directional smoke validation, NOT classification accuracy evidence",
        "features": X.tolist(),
        "labels": y.tolist(),
        "initial_weights": [0.0, 0.0],
        "geometry_comparison": rows,
        "finite_difference_epsilon": epsilon,
        "max_absolute_gradient_difference": difference,
        "numpy_version": np.__version__,
    }
    # Validate BOTH destinations before publishing either artifact.
    _guard(figpath)
    _guard(resultpath)
    _publish(figpath, image_buffer.getvalue())
    _publish(resultpath, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"Wrote {figpath.relative_to(root)} and {resultpath.relative_to(root)}")
    print(f"Finite-difference maximum absolute gradient error: {difference:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
