"""Unit tests for the WeightedKNNExplainer (Wang et al. 2024).

These tests verify that the weighted variant of KNN-Shapley works as expected:
- Runs without error and outputs valid results.
- Respects symmetry, distance influence, and label effects.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student.weighted_knn_explainer import WeightedKNNExplainer

TOLERANCE = 1e-6
POSITIVE_INFLUENCE_THRESHOLD = 0.5
NO_INFLUENCE_THRESHOLD = 1e-6


def test_weighted_knn_explainer_runs():
    """Basic sanity check: explainer runs and outputs valid results."""
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0, 0, 1])
    x_test = np.array([[1.1]])

    model = KNeighborsClassifier(n_neighbors=2, weights="distance")
    model.fit(X_train, y_train)

    explainer = WeightedKNNExplainer(model=model, data=X_train, labels=y_train, class_index=0)
    explanation = explainer.explain_instances(x_test)

    assert explanation is not None
    assert explanation.index == "SV"
    assert explanation.max_order == 1
    assert explanation.n_players == X_train.shape[0]
    assert len(explanation.values) == len(X_train)


def test_symmetry():
    """Symmetric samples with same label should have identical influence."""
    X = np.array([[0.0], [2.0]])
    y = np.array([1, 1])
    x_test = np.array([[1.0]])

    model = KNeighborsClassifier(n_neighbors=2, weights="distance")
    model.fit(X, y)
    explainer = WeightedKNNExplainer(model, X, y)
    values = explainer.explain_instances(x_test).values

    assert np.allclose(values[0], values[1], atol=TOLERANCE), f"Symmetry violated: {values}"


def test_distance_influence():
    """Closer correct point should have stronger influence than distant incorrect one."""
    X = np.array([[0.0], [10.0]])
    y = np.array([0, 1])
    x_test = np.array([[1.0]])

    model = KNeighborsClassifier(n_neighbors=2, weights="distance")
    model.fit(X, y)
    explainer = WeightedKNNExplainer(model, X, y, class_index=0)
    values = explainer.explain_instances(x_test).values

    assert values[0] > values[1], f"Distance influence violated: {values}"


def test_balance_positive_vs_negative():
    """Only the target-class sample should have positive influence."""
    X = np.array([[0.0], [2.0]])
    y = np.array([1, 0])
    x_test = np.array([[1.0]])

    model = KNeighborsClassifier(n_neighbors=2, weights="distance")
    model.fit(X, y)
    explainer = WeightedKNNExplainer(model, X, y, class_index=1)
    values = explainer.explain_instances(x_test).values

    assert values[0] > POSITIVE_INFLUENCE_THRESHOLD, (
        f"Target class point has weak influence: {values}"
    )
    assert values[1] < NO_INFLUENCE_THRESHOLD, (
        f"Opposing class point should have no influence: {values}"
    )
