import numpy as np
import pytest

from mnist_optimization.logistic import logistic_gradient, logistic_loss, validate_data


def test_zero_weights_loss_is_log_two():
    X = np.array([[1.0, 3.0], [2.0, 5.0]])
    y = np.array([-1, 1])
    assert logistic_loss(np.zeros(2), X, y) == pytest.approx(np.log(2))


def test_gradient_at_zero_matches_analytical_formula():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([-1, 1])
    expected = -X.T @ y / 4
    np.testing.assert_allclose(logistic_gradient(np.zeros(2), X, y), expected)


@pytest.mark.parametrize("regularization,bias_index", [(0.0, None), (0.4, None), (0.1, 2)])
def test_finite_differences(regularization, bias_index):
    rng = np.random.default_rng(99)
    X = rng.standard_normal((12, 3))
    y = rng.choice([-1, 1], 12)
    w = rng.standard_normal(3)
    analytic = logistic_gradient(w, X, y, l2=regularization, bias_index=bias_index)
    epsilon = 1e-6
    finite = np.empty_like(w)
    for j in range(len(w)):
        h = np.zeros_like(w)
        h[j] = epsilon
        finite[j] = (
            logistic_loss(w + h, X, y, l2=regularization, bias_index=bias_index)
            - logistic_loss(w - h, X, y, l2=regularization, bias_index=bias_index)
        ) / (2 * epsilon)
    np.testing.assert_allclose(analytic, finite, rtol=2e-6, atol=1e-8)


def test_extreme_scores_never_overflow_and_gradients_are_finite():
    X = np.array([[1000.0, -1000.0], [-1000.0, 1000.0]])
    y = np.array([-1, 1])
    for w in (np.array([1000.0, -1000.0]), np.array([-1000.0, 1000.0])):
        assert np.isfinite(logistic_loss(w, X, y))
        assert np.all(np.isfinite(logistic_gradient(w, X, y)))


def test_regularization_excludes_intercept():
    X = np.array([[2.0, 1.0], [-1.0, 1.0]])
    y = np.array([1, -1])
    w = np.array([1.0, 3.0])
    assert logistic_loss(w, X, y, l2=1.0, bias_index=1) == pytest.approx(logistic_loss(w, X, y) + 0.5)
    np.testing.assert_allclose(
        logistic_gradient(w, X, y, l2=1.0, bias_index=1) - logistic_gradient(w, X, y), [1.0, 0.0]
    )


@pytest.mark.parametrize("X,y", [([[1, 2]], [0]), ([[1, np.nan]], [1]), ([[1, 2]], [1, -1]), ([1, 2], [1, -1])])
def test_invalid_data_rejected(X, y):
    with pytest.raises(ValueError):
        validate_data(X, y)


@pytest.mark.parametrize("l2,bias_index", [(-1, None), (np.inf, None), (0.0, 9)])
def test_bad_regularization_rejected(l2, bias_index):
    with pytest.raises(ValueError):
        logistic_loss(np.zeros(2), [[1, 2]], [1], l2=l2, bias_index=bias_index)


def test_nan_weights_rejected():
    with pytest.raises(ValueError):
        logistic_loss([np.nan], [[1.0]], [1])


def test_shuffle_invariance_of_average_objective_and_gradient():
    rng = np.random.default_rng(71)
    X = rng.normal(size=(27, 4))
    y = rng.choice([-1, 1], size=27)
    w = rng.normal(size=4)
    perm = rng.permutation(len(y))
    assert logistic_loss(w, X, y) == pytest.approx(logistic_loss(w, X[perm], y[perm]), abs=1e-13)
    np.testing.assert_allclose(logistic_gradient(w, X, y), logistic_gradient(w, X[perm], y[perm]), atol=1e-13)


def test_extreme_wrong_margin_loss_is_finite_and_has_correct_value():
    assert logistic_loss([100.0], [[1000.0]], [-1]) == pytest.approx(100000.0)
    assert logistic_gradient([100.0], [[1000.0]], [-1])[0] == pytest.approx(1000.0)
