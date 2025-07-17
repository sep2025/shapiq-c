"""Unit tests for the weighted KNNExplainer implementation."""

from __future__ import annotations

import numpy as np
from shapiq import ExactComputer
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student.weighted_knn_explainer import KNNExplainer

TOLERANCE = 1e-6


def test_weighted_knn_instance_shapley():
    """Tests the weighted KNNExplainer against exact Shapley values.

    This test evaluates the distance-weighted variant of KNNExplainer using a
    simple 3-point 1D classification problem with K=2 neighbors.

    It compares the approximate Shapley values returned by the explainer
    to the exact Shapley values computed using shapiq's ExactComputer.

    The test passes if the L2 distance between the two is below the defined tolerance.
    """
    # Training data: 3 points in 1D
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0, 0, 1])
    x_test = np.array([1.1])  # test point close to class 0
    y_class = 0  # class we explain
    K = 2  # number of neighbors

    # Train distance-weighted KNN model
    model = KNeighborsClassifier(n_neighbors=K, weights="distance")
    model.fit(X_train, y_train)

    # Instantiate the weighted KNN explainer
    explainer = KNNExplainer(model, X_train, y_train, class_index=y_class, K=K)
    approx = explainer.explain(x_test)

    # Define Shapley game: coalition -> model prediction accuracy
    def game(coalitions: np.ndarray) -> np.ndarray:
        results = []
        for coalition in coalitions:
            idx = np.where(coalition)[0]
            if len(idx) == 0:
                results.append(0.0)
                continue

            X_sub = X_train[idx]
            y_sub = y_train[idx]
            k_eff = min(K, len(idx))

            model_sub = KNeighborsClassifier(n_neighbors=k_eff, weights="distance")
            model_sub.fit(X_sub, y_sub)
            pred = model_sub.predict([x_test])[0]
            results.append(float(pred == y_class))
        return np.array(results)

    # Compute exact Shapley values
    exact = ExactComputer(n_players=3, game=game)(index="SV")

    # Compare L2 error between approximate and exact values
    error = np.linalg.norm(approx.values - exact.get_n_order_values(1))

    # Assert values are effectively equal
    assert error < TOLERANCE
