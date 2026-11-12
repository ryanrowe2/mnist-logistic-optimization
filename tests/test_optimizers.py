import numpy as np
import pytest

from mnist_optimization.logistic import logistic_gradient, logistic_loss
from mnist_optimization.optimizers import descent_direction, optimize


def test_directions_follow_stated_norm_geometry():
    gradient = np.array([-2.0, 3.0, 0.0])
    np.testing.assert_array_equal(descent_direction(gradient, "gd")[0], gradient)
    np.testing.assert_array_equal(descent_direction(gradient, "linf")[0], [-5, 5, 0])
    direction, chosen = descent_direction(gradient, "coordinate")
    np.testing.assert_array_equal(direction, [0, 3, 0])
    assert chosen == 1


def test_zero_gradient_all_direction_types():
    for method in ("gd", "linf", "coordinate"):
        direction, _ = descent_direction(np.zeros(3), method)
        np.testing.assert_array_equal(direction, np.zeros(3))


def test_one_step_matches_gradient_update():
    X = np.array([[1.0, 2.0], [-1.0, 3.0]])
    y = np.array([1, -1])
    expected = -0.01 * logistic_gradient(np.zeros(2), X, y)
    actual = optimize(X, y, iterations=1, step_size=0.01)
    np.testing.assert_allclose(actual.weights, expected)
    assert actual.iterations.tolist() == [0, 1]


@pytest.mark.parametrize("method,step_size", [("gd", 0.1), ("linf", 0.02), ("coordinate", 0.1)])
def test_descent_reduces_controlled_objective(method, step_size):
    X = np.array([[2, 1], [2, -1], [-2, -1], [-2, 1]], dtype=float)
    y = np.array([1, 1, -1, -1])
    result = optimize(X, y, iterations=100, step_size=step_size, record_every=10, method=method)
    assert result.losses[-1] < result.losses[0]
    assert logistic_loss(result.weights, X, y) == pytest.approx(result.losses[-1])
    if method == "coordinate":
        assert result.update_counts.sum() == 100
    else:
        assert result.update_counts.sum() == 0


def test_repeated_runs_bitwise_reproducible():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(35, 7))
    y = rng.choice([-1, 1], 35)
    a = optimize(X, y, iterations=50, record_every=7)
    b = optimize(X, y, iterations=50, record_every=7)
    np.testing.assert_array_equal(a.weights, b.weights)
    np.testing.assert_array_equal(a.losses, b.losses)
    assert a.iterations[-1] == 50


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": 0},
        {"iterations": 1.5},
        {"step_size": 0},
        {"step_size": np.inf},
        {"record_every": 0},
        {"method": "not_real"},
        {"initial_weights": [1, 2, 3]},
    ],
)
def test_bad_optimizer_parameters(kwargs):
    with pytest.raises(ValueError):
        optimize([[1.0, 2.0]], [1], **kwargs)


def test_initial_weights_not_mutated():
    original = np.array([2.0, 3.0])
    optimize([[1, 2]], [1], initial_weights=original, iterations=1)
    np.testing.assert_array_equal(original, [2, 3])


def test_invalid_direction_and_overflow_guard():
    with pytest.raises(ValueError):
        descent_direction([np.inf, 1.0], "gd")
    with pytest.raises(ValueError):
        descent_direction([1.0, 2.0], "unknown")
    with pytest.raises(FloatingPointError):
        optimize([[1e200]], [1], iterations=2, step_size=1e200)
