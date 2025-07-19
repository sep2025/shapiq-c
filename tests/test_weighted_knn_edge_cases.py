"""Unit tests for the WeightedKNNExplainer implementation.

These tests verify correctness, stability, and edge case handling of the
WKNN-Shapley explainer (Wang et al. 2024) using scikit-learn KNN models.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student.weighted_knn_explainer import KNNExplainer

# Constants to avoid magic numbers in tests
THRESHOLD_POSITIVE_INFLUENCE = 0.5
THRESHOLD_NO_INFLUENCE = 1e-6
EXPECTED_LENGTH = 3
EXPECTED_VALUES_2 = 2
EXPECTED_VALUES_3 = 3
MIN_NORM = 0.1


def get_basic_model_and_data():
    """Returns a simple fitted KNN model and example data."""
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 0, 1])
    x_test = np.array([[1.1]])
    model = KNeighborsClassifier(n_neighbors=2, weights="distance")
    model.fit(X, y)
    return model, X, y, x_test


def test_invalid_weights():
    """Ensure unsupported weight types raise a ValueError."""
    model = KNeighborsClassifier(n_neighbors=2, weights="custom")
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    with pytest.raises(ValueError, match=".*weights are supported.*"):
        KNNExplainer(model=model, data=X, labels=y)


def test_mismatched_data_labels():
    """Ensure mismatch in training data and label length raises ValueError."""
    model = KNeighborsClassifier(n_neighbors=2, weights="distance")
    model.fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError, match=".*data and labels.*"):
        KNNExplainer(model=model, data=[[0], [1]], labels=[0])  # Too few labels


def test_mode_assignment():
    """Verify explainer correctly sets mode based on KNN weight type."""
    model_w = KNeighborsClassifier(n_neighbors=2, weights="distance")
    model_u = KNeighborsClassifier(n_neighbors=2, weights="uniform")
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    model_w.fit(X, y)
    model_u.fit(X, y)

    e1 = KNNExplainer(model_w, X, y)
    e2 = KNNExplainer(model_u, X, y)
    assert e1.mode == "weighted"
    assert e2.mode == "normal"


def test_explain_instances_with_tie():
    """Check explain_instances handles equal distances properly."""
    X = np.array([[0.0], [2.0], [4.0]])
    y = np.array([1, 0, 1])
    x_test = np.array([[2.0]])  # Equidistant to points at indices 1 and 2
    model = KNeighborsClassifier(n_neighbors=2, weights="distance")
    model.fit(X, y)
    explainer = KNNExplainer(model, X, y, class_index=1)
    values = explainer.explain_instances(x_test).values

    assert values.shape[0] == EXPECTED_LENGTH
    # The target class point should have approx 1/3 influence due to tie
    assert pytest.approx(values[0], abs=1e-3) == 1 / 3
    # Opposing class point influence should be near zero
    assert values[1] < THRESHOLD_NO_INFLUENCE


def test_explain_returns_real_values():
    """Ensure explain returns meaningful, nonzero values."""
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1])
    x = np.array([[0.5, 0.5]])
    model = KNeighborsClassifier(n_neighbors=1, weights="distance")
    model.fit(X, y)
    explainer = KNNExplainer(model, X, y, class_index=1)
    result = explainer.explain(x)

    assert np.linalg.norm(result.values) > MIN_NORM
    assert result.values.shape[0] == X.shape[1]


def test_k_equals_n():
    """Ensure explainer supports K equal to number of training samples."""
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1, 0, 1])
    x_test = np.array([[1.0]])
    model = KNeighborsClassifier(n_neighbors=3, weights="distance")
    model.fit(X, y)
    explainer = KNNExplainer(model, X, y, class_index=1, K=3)
    values = explainer.explain_instances(x_test).values

    assert len(values) == EXPECTED_VALUES_3


def test_k_equals_1():
    """Ensure explainer supports K=1."""
    X = np.array([[0.0], [2.0]])
    y = np.array([1, 0])
    x_test = np.array([[1.0]])
    model = KNeighborsClassifier(n_neighbors=1, weights="distance")
    model.fit(X, y)
    explainer = KNNExplainer(model, X, y, class_index=1, K=1)
    values = explainer.explain_instances(x_test).values

    assert len(values) == EXPECTED_VALUES_2


def test_explain_multiple_instances():
    """Check that explain handles multiple test inputs (batch mode)."""
    model, X, y, _ = get_basic_model_and_data()
    explainer = KNNExplainer(model, X, y, class_index=0)
    x_batch = np.array([[1.1], [1.2], [1.3]])
    result = explainer.explain(x_batch)

    assert result.values.shape[0] == X.shape[1]


def test_explain_instances_multiple_inputs():
    """Check that explain_instances supports multiple test points."""
    model, X, y, _ = get_basic_model_and_data()
    explainer = KNNExplainer(model, X, y, class_index=0)
    x_batch = np.array([[1.1], [1.2]])
    result = explainer.explain_instances(x_batch)

    assert result.values.shape[0] == X.shape[0]
