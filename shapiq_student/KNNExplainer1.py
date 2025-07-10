from __future__ import annotations

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
from scipy.spatial.distance import cdist
from shapiq import InteractionValues
from shapiq.explainer import Explainer
from sklearn.neighbors import KNeighborsClassifier


class KNNExplainer(Explainer):
    def __init__(self, model, data, labels, class_index=None, K=None):
        if not isinstance(model, KNeighborsClassifier):
            raise ValueError("Only sklearn's KNeighborsClassifier is supported.")

        self.model = model
        self.X_train = np.asarray(data)
        self.y_train = np.asarray(labels)
        self.class_index = class_index
        self.K = K if K is not None else model.n_neighbors

        self.mode = "normal"
        self.max_order = 1
        self.index = "SV"

        if self.X_train.shape[0] != len(self.y_train):
            raise ValueError("Mismatch between data and labels.")

    def _compute_single(self, x_test):
        if self.class_index is not None:
            y_test = self.class_index
        else:
            y_test = self.model.predict([x_test])[0]

        distances = cdist(self.X_train, [x_test], metric="euclidean").flatten()
        sorted_idx = np.argsort(distances)
        N = len(sorted_idx)
        s = np.zeros(N)
        last = sorted_idx[-1]
        s[-1] = int(self.y_train[last] == y_test) / N

        for i in reversed(range(N - 1)):
            idx_i = sorted_idx[i]
            idx_next = sorted_idx[i + 1]
            delta = (int(self.y_train[idx_i] == y_test) - int(self.y_train[idx_next] == y_test)) / self.K
            s[i] = s[i + 1] + delta * (min(self.K, i + 1) / (i + 1))

        shapley = np.zeros(N)
        for i, idx in enumerate(sorted_idx):
            shapley[idx] = s[i]
        return shapley

    def explain(self, x):
        x = np.atleast_2d(x)
        shapley_matrix = [self._compute_single(xi) for xi in x]
        averaged_shapley = np.mean(shapley_matrix, axis=0)

        return InteractionValues(
            values=averaged_shapley,
            index="SV",
            max_order=1,
            min_order=1,
            baseline_value=0.0,
            n_players=x.shape[1],
        )


