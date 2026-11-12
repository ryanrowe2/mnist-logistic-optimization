"""MNIST v1 source and two *explicitly different* evaluation protocols."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Protocol = Literal["historical", "official"]


@dataclass(frozen=True)
class BinarySplit:
    X_train: NDArray[np.float64]
    y_train: NDArray[np.int64]
    X_test: NDArray[np.float64]
    y_test: NDArray[np.int64]
    train_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    digit_pair: tuple[int, int]
    protocol: str
    scaling: str
    seed: int
    bias_index: int | None


def load_mnist(*, data_home: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load OpenML dataset 554 (mnist_784 version 1) using sklearn's cache.

    scikit-learn is used only for data transport; *not* for learning or metrics.
    Raise an actionable error if this network-reliant action is unavailable.
    """
    try:
        from sklearn.datasets import fetch_openml
    except ImportError as exc:
        raise RuntimeError("Install the MNIST extra: pip install -e '.[mnist]'") from exc
    try:
        X, y = fetch_openml(
            data_id=554,
            return_X_y=True,
            as_frame=False,
            parser="auto",
            data_home=str(data_home) if data_home is not None else None,
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to fetch OpenML MNIST dataset 554. Connect to the internet and retry; "
            "scikit-learn caches successful downloads. No other dataset will be substituted."
        ) from exc
    return validate_mnist(X, y)


def _validated_digit_labels(raw: ArrayLike) -> np.ndarray:
    """Accept decimal MNIST labels, rejecting booleans, complex and arbitrary objects."""
    values = np.asarray(raw)
    if values.dtype.kind in "iu":
        digits = values.astype(np.int64)
    elif values.dtype.kind == "f":
        if not np.all(np.isfinite(values)) or not np.all(np.equal(values, np.floor(values))):
            raise ValueError("digit labels must be decimal integers 0..9")
        digits = values.astype(np.int64)
    elif values.dtype.kind in "US" or (values.dtype.kind == "O" and all(isinstance(v, str) for v in values.flat)):
        strings = values.astype(str)
        if not np.all(np.isin(strings, tuple(str(d) for d in range(10)))):
            raise ValueError("digit labels must be decimal integers 0..9")
        digits = strings.astype(np.int64)
    else:
        raise ValueError("digit labels must be decimal integers 0..9 (not bool or complex)")
    if not np.all((digits >= 0) & (digits <= 9)):
        raise ValueError("digit labels must be decimal integers 0..9")
    return digits


def validate_mnist(X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(X)
    labels = np.asarray(y)
    if features.shape != (70000, 784) or labels.shape != (70000,):
        raise ValueError("MNIST v1 must be exactly 70000 by 784 with 70000 labels")
    if features.dtype.kind not in "iuf" or not np.all(np.isfinite(features)):
        raise ValueError("MNIST pixels must be finite numeric values")
    if np.min(features) < 0 or np.max(features) > 255:
        raise ValueError("MNIST pixel intensities must be in [0, 255]")
    compact_pixels = features.astype(np.uint8, copy=False)
    if not np.array_equal(features, compact_pixels):
        raise ValueError("MNIST pixels must have integer intensities")
    digits = _validated_digit_labels(labels)
    return compact_pixels, digits


def prepare_binary(
    X: ArrayLike,
    y: ArrayLike,
    *,
    pair: tuple[int, int] = (4, 9),
    protocol: Protocol = "historical",
    train_per_class: int = 500,
    test_per_class: int = 500,
    seed: int = 2024,
) -> BinarySplit:
    """Construct disjoint binary train/test sets from full canonical-order MNIST.

    `historical`: class-order slices 0:train, train:train+test of *all 70k*;
    raw 0..255, no intercept, independently seeded train/test shuffles.
    The original did NOT record seeds; here seeded shuffles are a controlled
    reconstruction, not a claim to reproduce the exact original permutation.

    `official`: seed-sample from MNIST rows 0:60000 (train) and 60000:70000
    (test), fixed 1/255 scaling, added intercept, no split leakage.
    """
    # The lower-level public API enforces the same canonical pixel domain
    # as the OpenML loader (including integer-valued pixels and valid digits).
    features, labels = validate_mnist(X, y)
    if (
        not isinstance(pair, (tuple, list))
        or len(pair) != 2
        or any(type(d) is not int or d < 0 or d > 9 for d in pair)
        or pair[0] == pair[1]
    ):
        raise ValueError("pair must contain two distinct digits in 0..9")
    for count in (train_per_class, test_per_class):
        if type(count) is not int or count <= 0:
            raise ValueError("sample counts must be positive integers")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if protocol not in ("historical", "official"):
        raise ValueError("protocol must be historical or official")
    pair = tuple(pair)  # Canonical, immutable representation for preparation and downstream validation.
    rng = np.random.default_rng(seed)  # Preserve the historical seeded shuffle sequence.
    # Official random choices depend on actual digit identity, not label sign
    # or the class's position in the caller-supplied pair. Train/evaluation
    # selections and shuffles have independent, explicitly keyed seed domains.
    if protocol == "official":
        low_digit, high_digit = sorted(pair)
        training_rngs = {
            digit: np.random.default_rng(np.random.SeedSequence([seed, 0, digit])) for digit in (low_digit, high_digit)
        }
        evaluation_rngs = {
            digit: np.random.default_rng(np.random.SeedSequence([seed, 1, digit])) for digit in (low_digit, high_digit)
        }
        training_shuffle = np.random.default_rng(np.random.SeedSequence([seed, 2, low_digit, high_digit]))
        evaluation_shuffle = np.random.default_rng(np.random.SeedSequence([seed, 3, low_digit, high_digit]))
        selection_order = (low_digit, high_digit)
    else:
        selection_order = pair
    train_pieces: list[np.ndarray] = []
    test_pieces: list[np.ndarray] = []
    for digit in selection_order:
        if protocol == "historical":
            indices = np.flatnonzero(labels == digit)
            if len(indices) < train_per_class + test_per_class:
                raise ValueError("not enough digit examples for the historical samples")
            train_pieces.append(indices[:train_per_class])
            test_pieces.append(indices[train_per_class : train_per_class + test_per_class])
        else:
            pool_train = np.flatnonzero(labels[:60000] == digit)
            pool_test = np.flatnonzero(labels[60000:] == digit) + 60000
            if len(pool_train) < train_per_class or len(pool_test) < test_per_class:
                raise ValueError("not enough samples in official train/test partition")
            train_pieces.append(training_rngs[digit].choice(pool_train, train_per_class, replace=False))
            test_pieces.append(evaluation_rngs[digit].choice(pool_test, test_per_class, replace=False))
    train_indices = np.concatenate(train_pieces)
    test_indices = np.concatenate(test_pieces)
    train_shuffle_rng = rng if protocol == "historical" else training_shuffle
    test_shuffle_rng = rng if protocol == "historical" else evaluation_shuffle
    train_indices = train_indices[train_shuffle_rng.permutation(len(train_indices))]
    test_indices = test_indices[test_shuffle_rng.permutation(len(test_indices))]
    if np.intersect1d(train_indices, test_indices).size:
        raise AssertionError("data leakage: train/test indices overlap")
    y_train = np.where(labels[train_indices] == pair[0], -1, 1).astype(np.int64)
    y_test = np.where(labels[test_indices] == pair[0], -1, 1).astype(np.int64)
    scaling = "raw_0_255" if protocol == "historical" else "fixed_divide_255"
    factor = 1.0 if protocol == "historical" else 1.0 / 255.0
    X_train = np.asarray(features[train_indices], dtype=np.float64) * factor
    X_test = np.asarray(features[test_indices], dtype=np.float64) * factor
    bias_index = None
    if protocol == "official":
        X_train = np.column_stack((X_train, np.ones(len(X_train))))
        X_test = np.column_stack((X_test, np.ones(len(X_test))))
        bias_index = 784
    return BinarySplit(
        X_train,
        y_train,
        X_test,
        y_test,
        train_indices,
        test_indices,
        pair,
        protocol,
        scaling,
        seed,
        bias_index,
    )
