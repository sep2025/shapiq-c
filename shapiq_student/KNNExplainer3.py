from __future__ import annotations

import numpy as np
from shapiq import Explainer, InteractionValues
from sklearn.neighbors import KNeighborsClassifier


class KNNExplainer(Explainer):

    def __init__(
        self,
        model: KNeighborsClassifier,
        data: np.ndarray,
        labels: np.ndarray,
        class_index: int = None
    ):

        super().__init__(model=model, index="SV", max_order=1)

        self.model = model
        self.data = data
        self.labels = labels
        self.k = model.n_neighbors
        self.class_index = class_index


        self.max_order = 1
        self.index = "SV"


        if hasattr(model, "weights") and model.weights is not None:
            if (isinstance(model.weights, str) and model.weights == "distance") or callable(model.weights):
                self.mode = "weighted"
            else:
                self.mode = "normal"
        else:
            self.mode = "normal"

    def explain_function(self, x: np.ndarray) -> InteractionValues:
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        n_train, n_features = self.data.shape
        shapley_values = np.zeros((x.shape[0], n_features))

        for idx, x_val in enumerate(x):
            weights = self._compute_weights(x_val)
            scores = self._compute_weighted_knn_shapley(x_val, weights)
            shapley_values[idx] = scores

        return InteractionValues(
            index="SV",
            max_order=1,
            min_order=1,
            n_players=n_features,
            values=shapley_values[0],
            baseline_value=0.0
        )

    def _compute_weights(self, x_val: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(self.data - x_val, axis=1)
        weights = np.exp(-distances)
        return self._discretize_weights(weights)

    def _discretize_weights(self, weights: np.ndarray, bits: int = 3) -> np.ndarray:
        W = 2 ** bits
        discrete_levels = np.linspace(0, 1, W)
        return discrete_levels[np.abs(discrete_levels[:, None] - weights).argmin(axis=0)]

    def _compute_weighted_knn_shapley(self, x_val: np.ndarray, weights: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(self.data - x_val, axis=1)
        nearest_indices = np.argsort(distances)[:self.k]
        feature_contributions = np.zeros(self.data.shape[1])

        label = self.model.predict(x_val.reshape(1, -1))[0]
        y_val = label if self.class_index is None else self.class_index

        weighted_votes = weights[nearest_indices] * (2 * (self.labels[nearest_indices] == y_val) - 1)
        influence = np.sum(weighted_votes)

        for f in range(self.data.shape[1]):
            x_masked = np.copy(x_val)
            x_masked[f] = 0
            dists_masked = np.linalg.norm(self.data - x_masked, axis=1)
            nearest_masked = np.argsort(dists_masked)[:self.k]
            weights_masked = self._discretize_weights(np.exp(-dists_masked))
            contrib = np.sum(weights_masked[nearest_masked] * (2 * (self.labels[nearest_masked] == y_val) - 1))
            feature_contributions[f] = influence - contrib

        return feature_contributions

