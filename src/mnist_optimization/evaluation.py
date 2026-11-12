"""Transparent binary classification metrics; no ML library estimator dependency."""

from __future__ import annotations

from math import sqrt
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike

from .logistic import validate_real_parameter


def wilson_interval(successes: int, trials: int, *, confidence: float = 0.95) -> dict:
    """Two-sided Wilson score CI for a binomial proportion (no SciPy dependency)."""
    if type(successes) is not int or type(trials) is not int or trials < 1 or successes < 0 or successes > trials:
        raise ValueError("require integer 0 <= successes <= trials, trials >= 1")
    try:
        confidence = validate_real_parameter(confidence, "confidence", positive=True)
    except ValueError as exc:
        raise ValueError("confidence must be a real probability strictly between zero and one") from exc
    if confidence >= 1:
        raise ValueError("confidence must lie strictly between zero and one")
    # Compute from the lower tail: 0.5 + confidence/2 rounds to 1.0
    # for representable values just below 1 and would raise StatisticsError.
    z = -NormalDist().inv_cdf((1.0 - confidence) / 2.0)
    p = successes / trials
    z2 = z * z
    den = 1 + z2 / trials
    centre = (p + z2 / (2 * trials)) / den
    half = (z / den) * sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials))
    return {
        "method": "Wilson",
        "confidence": float(confidence),
        "lower": max(0.0, centre - half),
        "upper": min(1.0, centre + half),
    }


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """Rows true (-1,+1); predictions (-1,+1) plus a zero-tie column if needed."""
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if truth.ndim != 1 or pred.ndim != 1 or truth.shape != pred.shape or not len(truth):
        raise ValueError("predicted and true labels must be equally sized nonempty vectors")
    if (
        truth.dtype.kind not in "iuf"
        or pred.dtype.kind not in "iuf"
        or not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(pred))
        or not np.all(np.isin(truth, (-1, 1)))
        or not np.all(np.isin(pred, (-1, 0, 1)))
    ):
        raise ValueError("truth labels must be ±1; predictions may additionally contain historical ties (0)")
    columns = (-1, 1, 0) if np.any(pred == 0) else (-1, 1)
    return np.array(
        [[np.sum((truth == a) & (pred == b)) for b in columns] for a in (-1, 1)],
        dtype=np.int64,
    )


def metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict:
    matrix = confusion_matrix(y_true, y_pred)
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    return {
        "count": total,
        "correct": correct,
        "errors": total - correct,
        "accuracy": correct / total,
        "accuracy_wilson_95_interval": wilson_interval(correct, total),
        "error_rate": (total - correct) / total,
        "confusion_rows_true_cols_pred_minus_plus_ties_if_present": matrix.tolist(),
        "confusion_rows_true_cols_pred_minus_plus": matrix[:, :2].tolist(),
        "unclassified_ties": int(matrix[:, 2].sum()) if matrix.shape[1] == 3 else 0,
        "errors_by_class": {
            "negative": int(matrix[0, 1:].sum()),
            "positive": int(matrix[1, [0, 2]].sum()) if matrix.shape[1] == 3 else int(matrix[1, 0]),
        },
        "recall_wilson_95_by_class": {
            "negative": wilson_interval(int(matrix[0, 0]), int(matrix[0].sum())) if matrix[0].sum() else None,
            "positive": wilson_interval(int(matrix[1, 1]), int(matrix[1].sum())) if matrix[1].sum() else None,
        },
        "recall_by_class": {
            "negative": float(matrix[0, 0] / matrix[0].sum()) if matrix[0].sum() else None,
            "positive": float(matrix[1, 1] / matrix[1].sum()) if matrix[1].sum() else None,
        },
        "error_rate_by_class": {
            "negative": float(matrix[0, 1:].sum() / matrix[0].sum()) if matrix[0].sum() else None,
            "positive": float((matrix[1].sum() - matrix[1, 1]) / matrix[1].sum()) if matrix[1].sum() else None,
        },
    }
