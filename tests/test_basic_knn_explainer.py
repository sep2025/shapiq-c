"""Unit tests for the basic KNNExplainer implementation."""

from __future__ import annotations

import numpy as np
from shapiq import ExactComputer
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student.basic_knn_explainer import KNNExplainer

TOLERANCE = 1e-6


def test_knn_shapley_vs_exact():
    """Tests that the basic KNNExplainer returns exact Shapley values.

    This test compares the output of the custom KNNExplainer against the result
    of shapiq's ExactComputer for a simple 1D classification task with K=2 neighbors.

    The test setup includes:
        - A training set with 3 labeled 1D points.
        - A KNeighborsClassifier with n_neighbors=2.
        - A single test instance at 1.1.
        - Comparison of the KNNExplainer output to the exact Shapley values computed via ExactComputer.

    The test passes if the L2 norm between the two explanations is below the defined tolerance.
    """
    # Training data
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0, 0, 1])
    x_test = np.array([1.1])
    y_test_class = 0
    K = 2
    n_players = len(X_train)

    # Fit a KNN model
    model = KNeighborsClassifier(n_neighbors=K)
    model.fit(X_train, y_train)

    # Use custom explainer to compute Shapley values
    explainer = KNNExplainer(model, X_train, y_train, class_index=y_test_class, K=K)
    approx_shapley = explainer.explain(x_test)

    # Define coalition game: returns 1.0 if prediction equals class, else 0.0
    def game(coalitions: np.ndarray) -> np.ndarray:
        values = []
        for coalition in coalitions:
            idx = np.where(coalition)[0]
            if len(idx) == 0:
                values.append(0.0)
                continue
            X_sub = X_train[idx]
            y_sub = y_train[idx]
            k = min(K, len(idx))
            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(X_sub, y_sub)
            pred = clf.predict([x_test])[0]
            values.append(float(pred == y_test_class))
        return np.array(values)

    # Compute exact Shapley values
    exact_result = ExactComputer(n_players=n_players, game=game)(index="SV")

    # Compare approximate to exact result using L2 norm
    error = np.linalg.norm(approx_shapley.values - exact_result.get_n_order_values(1))
    assert error < TOLERANCE
