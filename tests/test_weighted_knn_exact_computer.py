"""Test for WeightedKNNExplainer correctness.

This test verifies that the WKNN-Shapley implementation (Wang et al. 2024)
produces correct Shapley values by comparing its output to values computed
via shapiq's ExactComputer on a small example.
"""

from __future__ import annotations

import numpy as np
from shapiq import ExactComputer
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student.weighted_knn_explainer import WeightedKNNExplainer

TOLERANCE = 1e-6  # High precision tolerance for exact match


def test_exact_wknn_shapley_matches_shapiq():
    """Compare weighted KNNExplainer output with ExactComputer reference values."""
    # 1. Construct a small 1D dataset with 3 labeled training samples
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0, 0, 1])
    x_test = np.array([1.1])
    y_class = 0  # Target class for explanation
    K = 2

    # 2. Train KNN classifier with distance-based weighting
    model = KNeighborsClassifier(n_neighbors=K, weights="distance")
    model.fit(X_train, y_train)

    # 3. Initialize the WKNN-Shapley explainer with target class fixed
    explainer = WeightedKNNExplainer(model, X_train, y_train, class_index=y_class, K=K)
    approx = explainer.explain_instances(x_test)

    # 4. Define a Shapley game: utility = 1 if classifier predicts target class, else 0
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

            # Train KNN on subset and evaluate correctness
            model_sub = KNeighborsClassifier(n_neighbors=k_eff, weights="distance")
            model_sub.fit(X_sub, y_sub)
            pred = model_sub.predict([x_test])[0]
            results.append(float(pred == y_class))

        return np.array(results)

    # 5. Compute exact Shapley values using ExactComputer
    exact = ExactComputer(n_players=X_train.shape[0], game=game)(index="SV")

    # 6. Compare with explainer output using L2 distance
    error = np.linalg.norm(approx.values - exact.get_n_order_values(1))

    # 7. Ensure error is below threshold
    assert error < TOLERANCE, f"Shapley values differ too much: error = {error:.2e}"
