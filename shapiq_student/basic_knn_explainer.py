"""KNNExplainer module.

This module provides an exact KNN-based Shapley explainer that attributes predictions
to individual training instances using exhaustive Shapley value computations.

This approach follows Jia et al. (2019) and calculates marginal contributions of each
training point by evaluating all possible permutations of the training set.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
from shapiq.explainer import Explainer
from shapiq.interaction_values import InteractionValues
from sklearn.neighbors import KNeighborsClassifier

# Define named constant to avoid magic number usage
NDIM_FEATURE_AXIS = 2


class KNNExplainer(Explainer):
    """Exact KNN-based Shapley explainer using exhaustive permutations.

    This class explains the prediction for a given test instance by computing
    exact Shapley values for each training point using a k-nearest neighbors classifier.

    For each permutation of the training set, the marginal contribution of each
    instance is computed by observing its effect on the prediction when added
    to a subset of preceding instances.

    Args:
        model: A fitted sklearn KNeighborsClassifier instance.
        data: Training data as a 2D numpy array of shape (n_samples, n_features).
        labels: Labels corresponding to the training data.
        class_index: (Optional) Class index to explain. If None, the predicted class is used.
        K: (Optional) Number of neighbors to consider. Defaults to the model's setting.

    Raises:
        TypeError: If model is not an instance of KNeighborsClassifier.
        ValueError: If data and labels have inconsistent lengths.
    """

    def __init__(
        self,
        model: KNeighborsClassifier,
        data: np.ndarray,
        labels: np.ndarray,
        class_index: int | None = None,
        K: int | None = None,
    ) -> None:
        """Initializes the KNNExplainer with training data and model."""
        if not isinstance(model, KNeighborsClassifier):
            msg = "Only sklearn's KNeighborsClassifier is supported."
            raise TypeError(msg)

        self.model = model
        self.X_train = np.asarray(data)
        self.y_train = np.asarray(labels)
        self.class_index = class_index
        self.K = K if K is not None else model.n_neighbors

        self.mode = "normal"
        self.max_order = 1
        self.index = "SV"

        if self.X_train.shape[0] != len(self.y_train):
            msg = "Mismatch between data and labels."
            raise ValueError(msg)

    def _compute_single(self, x_test: np.ndarray) -> np.ndarray:
        """Computes exact Shapley values for a single test point.

        For each permutation of the training data, the model is retrained on growing
        subsets of the permutation, and the prediction's agreement with the target
        class is used to measure marginal contributions.

        Args:
            x_test: A single test instance as a 1D numpy array.

        Returns:
            A numpy array of shape (n_train,) containing the Shapley values.
        """
        n = len(self.X_train)
        shapley = np.zeros(n)

        # Determine which class to explain
        y_target = (
            self.class_index if self.class_index is not None else self.model.predict([x_test])[0]
        )

        # Generate all permutations of the training indices (brute-force Shapley)
        all_perms = list(permutations(range(n)))

        for pi in all_perms:
            S = []
            pred_prev = 0.0

            for player in pi:
                S.append(player)

                # Fit a KNN model on the current subset S
                X_S = self.X_train[S]
                y_S = self.y_train[S]
                k = min(self.K, len(S))

                clf = KNeighborsClassifier(n_neighbors=k)
                clf.fit(X_S, y_S)
                pred = clf.predict([x_test])[0]

                value = float(pred == y_target)
                marginal_contribution = value - pred_prev
                shapley[player] += marginal_contribution
                pred_prev = value

        shapley /= len(all_perms)
        return shapley

    def explain(self, x: np.ndarray) -> InteractionValues:
        """Returns averaged Shapley values for one or more test points.

        If multiple test points are provided, their individual Shapley values
        are averaged to provide a stable overall attribution.

        Args:
            x: A 1D or 2D array of test points to explain.

        Returns:
            InteractionValues: An object containing the Shapley values per training instance.
        """
        x = np.atleast_2d(x)
        shapley_matrix = [self._compute_single(xi) for xi in x]
        averaged_shapley = np.mean(shapley_matrix, axis=0)

        return InteractionValues(
            values=averaged_shapley,
            index="SV",
            max_order=1,
            min_order=1,
            baseline_value=0.0,
            n_players=x.shape[1] if x.ndim == NDIM_FEATURE_AXIS else 1,
        )
