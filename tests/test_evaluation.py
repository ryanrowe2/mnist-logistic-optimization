import numpy as np
import pytest

from mnist_optimization.evaluation import confusion_matrix, metrics
from mnist_optimization.logistic import add_intercept, predict, scores


def test_confusion_and_rates():
    y = [-1, -1, 1, 1, 1]
    p = [-1, 1, 1, -1, 1]
    np.testing.assert_array_equal(confusion_matrix(y, p), [[1, 1], [1, 2]])
    m = metrics(y, p)
    assert m["count"] == 5
    assert m["errors"] == 2
    assert m["accuracy"] == 0.6
    assert m["recall_by_class"]["positive"] == pytest.approx(2 / 3)


def test_prediction_tie_is_positive():
    X = np.array([[0, 1], [2, 1], [-2, 1]])
    w = np.array([1, 0])
    assert predict(X, w).tolist() == [1, 1, -1]
    assert scores(X, w).tolist() == [0, 2, -2]


def test_add_intercept():
    X = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(add_intercept(X), [[1, 2, 1], [3, 4, 1]])


@pytest.mark.parametrize("pred", [[-1], [-1, 2], [1, 1, 1]])
def test_invalid_confusion(pred):
    with pytest.raises(ValueError):
        confusion_matrix([-1, 1], pred)


def test_prediction_rejects_bad_weights():
    with pytest.raises(ValueError):
        predict([[1.0, 2.0]], [1.0])
