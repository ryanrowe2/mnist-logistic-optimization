"""Full-batch descent and norm-steepest first-order alternatives from MATH 173A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .logistic import (
    logistic_gradient, logistic_loss, real_matrix, validate_data,
    validate_real_parameter, validate_weights,
)

Method = Literal["gd", "linf", "coordinate"]


@dataclass(frozen=True)
class OptimizationResult:
    weights: NDArray[np.float64]
    iterations: NDArray[np.int64]
    losses: NDArray[np.float64]
    update_counts: NDArray[np.int64]
    method: str
    selection_counts: NDArray[np.int64] | None = None


def descent_direction(gradient: ArrayLike, method: Method) -> tuple[NDArray[np.float64], int | None]:
    """The signed steepest-direction scaling stated in 173A HW4.

    gd         : gradient
    linf       : ||gradient||_1 * sign(gradient)
    coordinate : ||gradient||_inf * sign(gradient[j]) * e_j

    Both alternate directions are steepest descent for different norm geometries.
    This is *not* the erroneously reciprocal L-infinity step in the old coursework.
    """
    g = real_matrix(gradient)
    if g.ndim != 1 or not np.all(np.isfinite(g)) or len(g) == 0:
        raise ValueError("gradient must be a finite, nonempty vector")
    if method == "gd":
        return g.copy(), None
    if method == "linf":
        try:
            with np.errstate(over="raise", invalid="raise"):
                direction = np.sign(g) * np.sum(np.abs(g))
        except FloatingPointError as exc:
            raise FloatingPointError("L-infinity direction overflow: rescale extreme gradients") from exc
        if not np.all(np.isfinite(direction)):
            raise FloatingPointError("L-infinity direction became non-finite")
        return direction, None
    if method == "coordinate":
        if not np.any(g):
            return np.zeros_like(g), None
        j = int(np.argmax(np.abs(g)))  # deterministic tie rule: first maximum
        direction = np.zeros_like(g)
        direction[j] = g[j]  # same as ||g||_inf * sign(g[j])
        return direction, j
    raise ValueError(f"unsupported method: {method}")


def optimize(
    X: ArrayLike,
    y: ArrayLike,
    *,
    method: Method = "gd",
    step_size: float = 1e-4,
    iterations: int = 10000,
    record_every: int = 50,
    initial_weights: ArrayLike | None = None,
    l2: float = 0.0,
    bias_index: int | None = None,
) -> OptimizationResult:
    """Perform exactly `iterations` full-batch updates, recording loss after updates.

    The `iterations=0` initial loss is recorded for a diagnostic baseline.
    Does not promise descent for arbitrary step sizes; callers must inspect results.
    """
    features, labels = validate_data(X, y)
    if isinstance(iterations, (bool, np.bool_)) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if isinstance(record_every, (bool, np.bool_)) or not isinstance(record_every, int) or record_every < 1:
        raise ValueError("record_every must be a positive integer")
    step_size = validate_real_parameter(step_size, "step_size", positive=True)
    l2 = validate_real_parameter(l2, "l2")
    if method not in ("gd", "linf", "coordinate"):
        raise ValueError(f"unsupported method: {method}")
    weights = (
        np.zeros(features.shape[1], dtype=np.float64)
        if initial_weights is None
        else validate_weights(initial_weights, features.shape[1]).copy()
    )
    counts = np.zeros(features.shape[1], dtype=np.int64)
    selections = np.zeros(features.shape[1], dtype=np.int64)
    history_iterations = [0]
    history_losses = [logistic_loss(weights, features, labels, l2=l2, bias_index=bias_index)]
    for step in range(1, iterations + 1):
        gradient = logistic_gradient(weights, features, labels, l2=l2, bias_index=bias_index)
        direction, index = descent_direction(gradient, method)
        previous_coordinate = weights[index].copy() if index is not None else None
        try:
            with np.errstate(over="raise", invalid="raise"):
                weights -= step_size * direction
        except FloatingPointError as exc:
            raise FloatingPointError(f"weights overflowed on iteration {step}") from exc
        if not np.all(np.isfinite(weights)):
            raise FloatingPointError(f"weights became non-finite on iteration {step}")
        if index is not None:
            selections[index] += 1
            if weights[index] != previous_coordinate:
                counts[index] += 1
        if step % record_every == 0 or step == iterations:
            loss = logistic_loss(weights, features, labels, l2=l2, bias_index=bias_index)
            if not np.isfinite(loss):
                raise FloatingPointError(f"loss became non-finite on iteration {step}")
            history_iterations.append(step)
            history_losses.append(loss)
    return OptimizationResult(
        weights=weights,
        iterations=np.asarray(history_iterations, dtype=np.int64),
        losses=np.asarray(history_losses, dtype=np.float64),
        update_counts=counts,
        method=method,
        selection_counts=selections,
    )
