"""Edge case tests for BasicKNNExplainer.

These tests check rare or boundary conditions for the basic KNN-Shapley explainer,
e.g. K > n, missing target class, and parameterized configurations.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student.basic_knn_explainer import KNNExplainer


@pytest.mark.parametrize(
    ("K", "x_test", "y_test_class"),
    [
        (1, np.array([[0.0]]), 0),
        (2, np.array([[1.1]]), 0),
        (3, np.array([[2.0]]), 1),
    ],
)
def test_knn_shapley_multiple_configs(K, x_test, y_test_class):
    """Check that KNNExplainer works across multiple K and input settings."""
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0, 0, 1])

    model = KNeighborsClassifier(n_neighbors=K)
    model.fit(X_train, y_train)

    explainer = KNNExplainer(model, X_train, y_train, class_index=y_test_class, K=K)
    values = explainer.explain(x_test).values

    assert len(values) == len(X_train)


def test_knn_explainer_k_larger_than_n():
    """Ensure explainer works when K > number of training samples."""
    X_train = np.array([[0.0], [1.0]])
    y_train = np.array([0, 1])
    x_test = np.array([[0.5]])
    K = 5  # K larger than number of players

    model = KNeighborsClassifier(n_neighbors=K)
    model.fit(X_train, y_train)

    explainer = KNNExplainer(model, X_train, y_train, class_index=0, K=K)
    values = explainer.explain(x_test).values

    assert len(values) == len(X_train)


def test_knn_explainer_target_class_not_in_majority():
    """Test that all Shapley values are zero if target class does not occur."""
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([1, 1, 1])  # Only class 1 present
    x_test = np.array([[1.1]])
    target_class = 0  # class 0 is missing

    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X_train, y_train)

    explainer = KNNExplainer(model, X_train, y_train, class_index=target_class, K=2)
    values = explainer.explain(x_test).values

    assert np.allclose(values, 0), f"Expected all zeros, got: {values}"


def test_knn_explainer_unequal_data_label_lengths():
    """Check that ValueError is raised when data and labels length mismatch."""
    X = np.array([[0], [1]])
    y = np.array([0])  # Fewer labels than samples
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit([[0], [1]], [0, 1])

    with pytest.raises(ValueError, match="same length"):
        KNNExplainer(model, X, y)


def test_knn_explainer_predicts_when_class_index_none():
    """Ensure model.predict is used if class_index is None."""
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    x_test = np.array([[0.5]])
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X, y)

    explainer = KNNExplainer(model, X, y)  # class_index=None
    values = explainer.explain(x_test).values

    assert len(values) == len(X)
