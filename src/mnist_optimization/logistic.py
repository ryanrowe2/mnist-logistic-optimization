"""Stable, NumPy-only binary logistic regression with labels in {-1, +1}."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_real_parameter(value: object, name: str, *, positive: bool = False) -> float:
    """Accept finite real scalars, excluding booleans and lossy numeric coercions."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        checked = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not np.isfinite(checked) or (checked <= 0 if positive else checked < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    if isinstance(value, (int, np.integer)) and int(checked) != int(value):
        raise ValueError(f"{name} integer cannot be represented exactly as float64")
    if isinstance(value, np.floating) and np.dtype(value.dtype).itemsize > np.dtype(np.float64).itemsize:
        # Preserve the original extended-precision value, including subnormal values.
        if value.dtype.type(checked) != value:
            raise ValueError(f"{name} cannot be represented exactly as float64")
    return checked


def real_matrix(X: ArrayLike) -> NDArray[np.float64]:
    """Reject complex, object, text and bool inputs before any lossy cast."""
    original = np.asarray(X)
    if original.dtype.kind not in "iuf":
        raise ValueError("X must contain real numeric values (no complex, bool or object arrays)")
    try:
        features = np.asarray(original, dtype=np.float64)
    except (ValueError, OverflowError) as exc:
        raise ValueError("X values cannot be represented as float64") from exc
    if not np.all(np.isfinite(features)):
        raise ValueError("X must contain finite float64-representable values")
    if original.dtype.kind in "iu":
        # Compare integer values, not their rounded float representations.
        beyond_exact_range = (original > 2**53) | (original < -(2**53))
        if np.any(beyond_exact_range):
            if not np.array_equal(original[beyond_exact_range].astype(object),
                                  features[beyond_exact_range].astype(object)):
                raise ValueError("X contains integers not exactly representable in float64")
    elif original.dtype.kind == "f" and original.dtype.itemsize > np.dtype("float64").itemsize:
        # Never silently round extended-precision features to the NumPy model dtype.
        if not np.array_equal(features.astype(original.dtype), original):
            raise ValueError("X contains extended-precision values not exactly representable in float64")
    return features


def validate_data(X: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Validate an n-by-d finite design matrix and finite binary labels."""
    features = real_matrix(X)
    labels = np.asarray(y)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("X must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(features)):
        raise ValueError("X must contain only finite values")
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError("y must be a vector with one label for each row of X")
    if labels.dtype.kind not in "iuf" or not np.all(np.isfinite(labels)) or not np.all(np.isin(labels, (-1, 1))):
        raise ValueError("y must contain only -1 and +1")
    return features, labels.astype(np.int64)


def validate_weights(w: ArrayLike, n_features: int) -> NDArray[np.float64]:
    weights = real_matrix(w)
    if weights.ndim != 1 or weights.shape != (n_features,):
        raise ValueError(f"weights must have shape ({n_features},)")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite")
    return weights


def _exact_dot(row: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    """Exactly accumulate binary64 inputs/products, then round the final result."""
    with localcontext() as context:
        # 2 * 308 decimal exponent range + binary mantissa and headroom for sums.
        context.prec = 1300
        total = sum((Decimal.from_float(float(a)) * Decimal.from_float(float(b))
                     for a, b in zip(row, weights, strict=True)), Decimal(0))
        try:
            return float(total)
        except OverflowError as exc:
            raise FloatingPointError("dot product is not representable in float64") from exc


def _safe_dot(X: NDArray[np.float64], w: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fast BLAS with accurate fallback for both overflow and finite cancellation.

    The conservative row-wise product bound detects loss of significance even
    when BLAS returns a finite value. Exact summation is only used for those
    rows, and for a nonfinite BLAS result; common MNIST inputs retain BLAS.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        result = X @ w
    result = np.asarray(result, dtype=np.float64)
    # Compute an upper bound in wide precision to avoid false overflow here.
    maxabs = np.max(np.abs(X.astype(np.longdouble)), axis=1)
    bound = maxabs * np.sum(np.abs(w.astype(np.longdouble)))
    tolerance = 16 * np.finfo(np.float64).eps * bound
    suspect = ~np.isfinite(result) | ((bound > 0) & (np.abs(result) <= tolerance))
    if np.any(suspect):
        result = result.copy()
        for i in np.flatnonzero(suspect):
            result[i] = _exact_dot(X[i], w)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("dot product is not representable in float64")
    return result


def _exact_weighted_coordinate(column: NDArray[np.float64],
                               factors: NDArray[np.float64],
                               regularizer: float, l2: float) -> float:
    with localcontext() as context:
        context.prec = 1300
        total = sum((Decimal.from_float(float(a)) * Decimal.from_float(float(f))
                     for a, f in zip(column, factors, strict=True)), Decimal(0))
        value = (-total / Decimal(len(factors))
                 + Decimal.from_float(l2) * Decimal.from_float(regularizer))
        try:
            return float(value)
        except OverflowError as exc:
            raise FloatingPointError("gradient is not representable in float64") from exc


def _safe_weighted_gradient(X: NDArray[np.float64], factors: NDArray[np.float64],
                            reg: NDArray[np.float64], l2: float) -> NDArray[np.float64]:
    """Accurate weighted reduction even if finite BLAS cancels small terms."""
    with np.errstate(over="ignore", invalid="ignore"):
        grad = -(X.T @ factors) / len(factors) + l2 * reg
    maxabs = np.max(np.abs(X.astype(np.longdouble)), axis=0)
    abs_sum = np.sum(np.abs(factors.astype(np.longdouble))) / len(factors)
    bound = maxabs * abs_sum + np.longdouble(l2) * np.abs(reg.astype(np.longdouble))
    tolerance = 16 * np.finfo(np.float64).eps * bound
    suspect = ~np.isfinite(grad) | ((bound > 0) & (np.abs(grad) <= tolerance))
    if np.any(suspect):
        grad = grad.copy()
        for j in np.flatnonzero(suspect):
            grad[j] = _exact_weighted_coordinate(X[:, j], factors, float(reg[j]), l2)
    if not np.all(np.isfinite(grad)):
        raise FloatingPointError("gradient is not representable in float64")
    return grad


def logistic_loss(
    w: ArrayLike, X: ArrayLike, y: ArrayLike, *, l2: float = 0.0, bias_index: int | None = None
) -> float:
    """Mean softplus(-y * (X @ w)) plus l2/2 * ||w_nonbias||^2."""
    features, labels = validate_data(X, y)
    weights = validate_weights(w, features.shape[1])
    l2 = validate_real_parameter(l2, "l2")
    reg = _regularizer(weights, l2, bias_index)
    with np.errstate(over="raise", invalid="raise"):
        try:
            margins = labels * _safe_dot(features, weights)
            # Normal float64 sum can overflow on four terms of 1e308 even
            # though the mean is representable. Normalize before reducing.
            terms = np.logaddexp(0.0, -margins)
            answer = math.fsum(float(term) / len(terms) for term in terms)
            # Do not evaluate ||w||² when l2 is zero: 0 * inf is NaN,
            # even when the unregularized objective is well-defined.
            if l2 != 0.0:
                # Scale before squaring; ||w||² may overflow even when
                # l2*||w||² is representable for small positive l2.
                weighted = np.sqrt(l2) * reg * np.sqrt(0.5)
                answer += np.dot(weighted, weighted)
        except FloatingPointError as exc:
            raise FloatingPointError("logistic objective overflow: rescale extreme inputs") from exc
    if not np.isfinite(answer):
        raise FloatingPointError("logistic objective became non-finite")
    return float(answer)


def logistic_gradient(
    w: ArrayLike, X: ArrayLike, y: ArrayLike, *, l2: float = 0.0, bias_index: int | None = None
) -> NDArray[np.float64]:
    """Gradient: -X.T @ [y * sigmoid(-y*(X@w))] / n + l2*w_nonbias."""
    features, labels = validate_data(X, y)
    weights = validate_weights(w, features.shape[1])
    l2 = validate_real_parameter(l2, "l2")
    reg = _regularizer(weights, l2, bias_index)
    with np.errstate(over="raise", invalid="raise"):
        try:
            margins = labels * _safe_dot(features, weights)
            # sigmoid(-margin) = exp(-logaddexp(0, margin)); no exp(+large).
            probabilities = np.exp(-np.logaddexp(0.0, margins))
            gradient = _safe_weighted_gradient(features, labels * probabilities, reg, l2)
        except FloatingPointError as exc:
            raise FloatingPointError("logistic gradient overflow: rescale extreme inputs") from exc
    if not np.all(np.isfinite(gradient)):
        raise FloatingPointError("logistic gradient became non-finite")
    return gradient


def _regularizer(weights: NDArray[np.float64], l2: float, bias_index: int | None) -> NDArray[np.float64]:
    validate_real_parameter(l2, "l2")
    result = weights.copy()
    if bias_index is not None:
        if type(bias_index) is not int or not 0 <= bias_index < len(result):
            raise ValueError("bias_index must be a valid feature index")
        result[bias_index] = 0.0
    return result


def scores(X: ArrayLike, w: ArrayLike) -> NDArray[np.float64]:
    features = real_matrix(X)
    if features.ndim != 2 or 0 in features.shape or not np.all(np.isfinite(features)):
        raise ValueError("X must be a nonempty finite two-dimensional array")
    weights = validate_weights(w, features.shape[1])
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = _safe_dot(features, weights)
        except FloatingPointError as exc:
            raise FloatingPointError("classification score overflow: rescale extreme inputs") from exc
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("classification scores became non-finite")
    return result


def predict(X: ArrayLike, w: ArrayLike, *, historical_sign: bool = False) -> NDArray[np.int64]:
    """Historical `np.sign` yields 0 on ties; modern prediction assigns +1."""
    value = scores(X, w)
    if historical_sign:
        return np.sign(value).astype(np.int64)
    return np.where(value >= 0.0, 1, -1).astype(np.int64)


def add_intercept(X: ArrayLike) -> NDArray[np.float64]:
    features = real_matrix(X)
    if features.ndim != 2 or 0 in features.shape or not np.all(np.isfinite(features)):
        raise ValueError("X must be a nonempty finite matrix")
    return np.column_stack((features, np.ones(len(features))))
