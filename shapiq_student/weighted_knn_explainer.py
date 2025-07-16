"""KNNExplainer module (weighted variant).

This module implements an exact Shapley value explainer for training instances
based on a distance-weighted k-nearest neighbors classifier.

This approach follows Wang et al. (2024), using full permutation enumeration
to compute the marginal contribution of each training point.
"""

from __future__ import annotations

import itertools

import numpy as np
from shapiq import Explainer, InteractionValues
from sklearn.neighbors import KNeighborsClassifier


class KNNExplainer(Explainer):
    """Exact weighted KNN-based Shapley explainer.

    This explainer computes Shapley values for training instances by using
    a distance-weighted k-nearest neighbors classifier. For each permutation
    of the training data, it measures the marginal contribution of each
    training point to the prediction for a test instance.

    Args:
        model: A fitted sklearn KNeighborsClassifier with weights="distance".
        data: The training data as a 2D numpy array.
        labels: Class labels for each training point.
        class_index: Optional. The class index to explain. If None, the predicted class is used.
        K: Optional. The number of neighbors to use (defaults to the model's setting).
    """

    def __init__(
        self,
        model: KNeighborsClassifier,
        data: np.ndarray,
        labels: np.ndarray,
        class_index: int | None = None,
        K: int | None = None,
    ) -> None:
        """Initializes the weighted KNNExplainer."""
        super().__init__(model=model, index="SV", max_order=1)
        self.X_train = data
        self.y_train = labels
        self.model = model
        self.K = K if K is not None else model.n_neighbors
        self.class_index = class_index

    def explain(self, x: np.ndarray) -> InteractionValues:
        """Computes Shapley values for a single test instance.

        This method wraps the internal exact Shapley computation and
        returns the results in the standardized InteractionValues format.

        Args:
            x: A 1D or 2D numpy array representing one or more test points.

        Returns:
            InteractionValues: Shapley values for the training instances.
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        shapley_vals = self._compute_exact_shapley(x[0])
        return InteractionValues(
            index="SV",
            max_order=1,
            min_order=1,
            n_players=len(self.X_train),
            values=shapley_vals,
            baseline_value=0.0,
        )

    def _compute_exact_shapley(self, x_test: np.ndarray) -> np.ndarray:
        """Computes exact Shapley values via permutation enumeration.

        For each permutation of the training points, we track the marginal
        improvement in prediction accuracy (0 or 1) as new instances are added.

        Args:
            x_test: A single test point to explain.

        Returns:
            A numpy array of Shapley values for each training point.
        """
        n = len(self.X_train)
        values = np.zeros(n)
        perms = list(itertools.permutations(range(n)))  # All possible orderings

        for perm in perms:
            subset = []
            prev_value = 0.0

            for idx in perm:
                # Add player to coalition
                subset.append(idx)
                X_sub = self.X_train[subset]
                y_sub = self.y_train[subset]

                # Choose effective k (cannot exceed subset size)
                k_eff = min(len(subset), self.K)

                # Fit weighted KNN model on current subset
                model = KNeighborsClassifier(n_neighbors=k_eff, weights="distance")
                model.fit(X_sub, y_sub)
                pred = model.predict([x_test])[0]

                # Use fixed class if provided, otherwise predict it
                target = (
                    self.class_index
                    if self.class_index is not None
                    else self.model.predict([x_test])[0]
                )

                # Binary payoff: 1 if prediction matches target class
                current_value = float(pred == target)

                # Compute marginal contribution
                marginal_contribution = current_value - prev_value
                values[idx] += marginal_contribution
                prev_value = current_value

        # Average over all permutations
        values /= len(perms)
        return values
