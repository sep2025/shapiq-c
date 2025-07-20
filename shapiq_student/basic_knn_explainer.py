"""BasicKNNExplainer module.

This module implements the original KNN-Shapley algorithm proposed by Jia et al. (2019).
It assigns Shapley values to each training instance in a KNN classifier by computing
its contribution to classification correctness over all coalitions.
"""

from __future__ import annotations

import numpy as np
from shapiq.explainer import Explainer
from shapiq.interaction_values import InteractionValues
from sklearn.neighbors import KNeighborsClassifier


class BasicKNNExplainer(Explainer):
    """Exact KNN-Shapley explainer for KNeighborsClassifier (Jia et al. 2019).

    Computes the influence of each training point on the classification of a
    test instance by calculating exact Shapley values based on label agreement
    within the top-K nearest neighbors.

    Args:
        model: Fitted sklearn KNeighborsClassifier instance.
        data: Training data used to fit the model.
        labels: Corresponding labels for the training data.
        class_index: Target class to explain. If None, model prediction is used.
        K: Number of neighbors to use. Defaults to model.n_neighbors.
    """

    def __init__(
        self,
        model: KNeighborsClassifier,
        data: np.ndarray,
        labels: np.ndarray,
        class_index: int | None = None,
        K: int | None = None,
    ) -> None:
        """Initialize the KNNExplainer with model, data, and parameters."""
        msg_type = "Only sklearn's KNeighborsClassifier is supported."
        if not isinstance(model, KNeighborsClassifier):
            raise TypeError(msg_type)

        super().__init__(model=model, index="SV", max_order=1)

        self.X_train = np.asarray(data)
        self.y_train = np.asarray(labels)
        self.model = model
        self.class_index = class_index
        self.K = K if K is not None else model.n_neighbors

        msg_len = "Training data and labels must have same length."
        if len(self.X_train) != len(self.y_train):
            raise ValueError(msg_len)

    def _compute_single(self, x_test: np.ndarray) -> np.ndarray:
        """Compute exact Shapley values for a single test instance.

        Args:
            x_test: Test instance, shape (n_features,).

        Returns:
            Array of Shapley values, one per training example.
        """
        n = len(self.X_train)
        shapley = np.zeros(n)

        msg_attr = "Provided model does not have a `predict` method."
        if not hasattr(self.model, "predict"):
            raise AttributeError(msg_attr)

        # Determine which class to explain
        y_target = (
            self.class_index if self.class_index is not None else self.model.predict([x_test])[0]
        )

        # Compute distances to training points and sort by proximity
        distances = np.linalg.norm(self.X_train - x_test, axis=1)
        sorted_indices = np.argsort(distances)
        sorted_labels = self.y_train[sorted_indices]

        # Mark whether each sorted point agrees with the target class
        is_correct = np.array([int(y == y_target) for y in sorted_labels])

        # Compute Shapley value contributions using recurrence
        s_sorted = np.zeros(n)
        s_sorted[-1] = is_correct[-1] / n

        for i in reversed(range(n - 1)):
            k = min(self.K, i + 1)
            delta = (is_correct[i] - is_correct[i + 1]) * k / (self.K * (i + 1))
            s_sorted[i] = s_sorted[i + 1] + delta

        # Map back to original training point indices
        for i, idx in enumerate(sorted_indices):
            shapley[idx] = s_sorted[i]

        return shapley

    def explain(self, x: np.ndarray) -> InteractionValues:
        """Compute Shapley values for one or more test instances.

        Args:
            x: Test instance(s), shape (n_samples, n_features) or (n_features,).

        Returns:
            InteractionValues object containing average Shapley values.
        """
        msg_input = "Missing input for 'x' in explain()."
        if x is None:
            raise ValueError(msg_input)

        x = np.atleast_2d(x)
        shapley_matrix = [self._compute_single(xi) for xi in x]
        averaged = np.mean(shapley_matrix, axis=0)

        return InteractionValues(
            values=averaged,
            index="SV",
            max_order=1,
            min_order=1,
            baseline_value=0.0,
            n_players=self.X_train.shape[0],
        )
