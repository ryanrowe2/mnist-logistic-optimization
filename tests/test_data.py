import numpy as np
import pytest

from mnist_optimization.data import prepare_binary, validate_mnist


def test_historical_exact_consecutive_global_indices(fixture_mnist):
    X, y = fixture_mnist
    split = prepare_binary(X, y, pair=(4, 9), protocol="historical", train_per_class=3, test_per_class=2)
    for digit in (4, 9):
        first = np.flatnonzero(y == digit)
        assert set(first[:3]).issubset(set(split.train_indices))
        assert set(first[3:5]).issubset(set(split.test_indices))
    assert split.X_train.shape == (6, 784)
    assert split.bias_index is None
    assert np.intersect1d(split.train_indices, split.test_indices).size == 0
    assert np.sum(split.y_train == -1) == np.sum(split.y_train == 1) == 3
    assert set(np.unique(split.y_test)) == {-1, 1}
    assert float(split.X_train.max()) > 1.0  # raw pixels not normalized


def test_official_protocol_uses_official_partition_and_intercept(fixture_mnist):
    X, y = fixture_mnist
    a = prepare_binary(X, y, pair=(4, 9), protocol="official", train_per_class=15, test_per_class=8, seed=42)
    b = prepare_binary(X, y, pair=(4, 9), protocol="official", train_per_class=15, test_per_class=8, seed=42)
    np.testing.assert_array_equal(a.train_indices, b.train_indices)
    np.testing.assert_array_equal(a.test_indices, b.test_indices)
    assert (a.train_indices < 60000).all()
    assert (a.test_indices >= 60000).all()
    assert a.X_train.shape == (30, 785)
    assert a.X_test.shape == (16, 785)
    assert a.bias_index == 784
    assert np.all(a.X_train[:, -1] == 1)
    assert np.max(a.X_train[:, :-1]) <= 1.0
    assert len(set(a.train_indices).intersection(a.test_indices)) == 0
    assert not np.array_equal(
        a.train_indices,
        prepare_binary(
            X, y, pair=(4, 9), protocol="official", train_per_class=15, test_per_class=8, seed=43
        ).train_indices,
    )


def test_validate_canonical_shape_and_digits(fixture_mnist):
    X, y = fixture_mnist
    features, labels = validate_mnist(X, y.astype(str))
    assert features.dtype == np.uint8
    assert labels.dtype == np.int64
    with pytest.raises(ValueError):
        validate_mnist(X[:100], y[:100])
    with pytest.raises(ValueError):
        validate_mnist(X, np.full(70000, 10))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pair": (4, 4)},
        {"pair": (-1, 9)},
        {"pair": (4, 10)},
        {"protocol": "wrong"},
        {"train_per_class": 0},
        {"test_per_class": -1},
        {"seed": -1},
        {"train_per_class": 9000},
    ],
)
def test_invalid_sampling_requests(fixture_mnist, kwargs):
    with pytest.raises(ValueError):
        prepare_binary(*fixture_mnist, **kwargs)


def test_non_integral_labels_are_rejected(fixture_mnist):
    X, y = fixture_mnist
    labels = y.astype(float)
    labels[4] = 4.5
    with pytest.raises(ValueError):
        prepare_binary(X, labels, train_per_class=2, test_per_class=1)


def test_oversized_official_test_sample_rejected(fixture_mnist):
    with pytest.raises(ValueError):
        prepare_binary(*fixture_mnist, protocol="official", test_per_class=5000)
