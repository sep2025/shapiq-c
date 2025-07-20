"""Test for BasicKNNExplainer correctness.

This test validates that the KNNExplainer (based on Jia et al. 2019)
produces exact Shapley values by comparing its output to the ground truth
computed using shapiq's ExactComputer on a simple synthetic dataset.
"""

from __future__ import annotations

import numpy as np
from shapiq import ExactComputer
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student.basic_knn_explainer import BasicKNNExplainer

TOLERANCE = 0.25  # Acceptable L2 deviation for numerical comparison


def test_knn_shapley_vs_exact():
    """Compare KNNExplainer Shapley values against ExactComputer on a toy example."""
    # 1. Define a simple 1D dataset with 3 training points
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0, 0, 1])
    x_test = np.array([[1.1]])  # single test instance (2D shape required)
    y_test_class = 0
    K = 2
    n_players = len(X_train)

    # 2. Train a standard KNN classifier
    model = KNeighborsClassifier(n_neighbors=K)
    model.fit(X_train, y_train)

    # 3. Run the custom KNNExplainer on the test instance
    explainer = BasicKNNExplainer(model, X_train, y_train, class_index=y_test_class, K=K)
    approx_shapley = explainer.explain(x_test).values

    # 4. Define the Shapley game: average label match among top-K neighbors
    def value_function(coalitions: np.ndarray) -> np.ndarray:
        """Utility = average number of correct labels among top-K neighbors."""
        results = []
        for coalition in coalitions:
            idx = np.where(coalition)[0]
            if len(idx) == 0:
                results.append(0.0)
                continue
            X_subset = X_train[idx]
            y_subset = y_train[idx]
            k_eff = min(K, len(idx))

            # Top-K neighbor selection based on Euclidean distance
            distances = np.linalg.norm(X_subset - x_test, axis=1)
            sorted_idx = np.argsort(distances)
            top_k_labels = y_subset[sorted_idx[:k_eff]]
            utility = np.mean(top_k_labels == y_test_class)
            results.append(utility)
        return np.array(results)

    # 5. Compute the exact Shapley values using shapiq's ExactComputer
    exact_computer = ExactComputer(n_players=n_players, game=value_function)
    exact_shapley = exact_computer(index="SV").get_n_order_values(1)

    # 6. Compute and compare the L2 error between implementations
    error = np.linalg.norm(approx_shapley - exact_shapley)

    # 7. Assert correctness within the allowed tolerance
    assert error < TOLERANCE, f"Shapley values differ too much: L2 error = {error:.2e}"
