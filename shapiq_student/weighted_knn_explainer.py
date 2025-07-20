"""WeightedKNNExplainer module.

This module implements the WKNN-Shapley algorithm (Wang et al. 2024),
supporting both exact and approximated variants for instance-wise influence
estimation in distance-weighted KNN classifiers.
"""

from __future__ import annotations

from itertools import combinations
import math
from typing import TYPE_CHECKING

import numpy as np
from shapiq import Explainer, InteractionValues

if TYPE_CHECKING:
    from sklearn.neighbors import KNeighborsClassifier


class WeightedKNNExplainer(Explainer):
    """KNN-Shapley Explainer for distance-based KNN classifiers.

    Args:
        model: A KNeighborsClassifier with weights='distance' or 'uniform'.
        data: Training input data.
        labels: Training labels.
        class_index: Optional fixed target class.
        K: Number of neighbors (default: model.n_neighbors).
        approx_M: Optional number of subset samples for approximation.
    """

    def __init__(
        self,
        model: KNeighborsClassifier,
        data: np.ndarray,
        labels: np.ndarray,
        class_index: int | None = None,
        K: int | None = None,
        approx_M: int | None = None,
    ) -> None:
        """Initialize the WKNN explainer."""
        super().__init__(model=model, index="SV", max_order=1)

        if model.weights not in ("distance", "uniform"):
            msg = "Only 'distance' or 'uniform' weights are supported."
            raise ValueError(msg)

        self.model = model
        self.X_train = np.asarray(data)
        self.y_train = np.asarray(labels)
        self.class_index = class_index
        self.K = K if K is not None else model.n_neighbors
        self.N = len(self.X_train)
        self.mode = "weighted" if model.weights == "distance" else "normal"
        self.approx_M = approx_M
        self.top_S: list[int] = []

        if self.X_train.shape[0] != self.y_train.shape[0]:
            msg = "Mismatch between data and labels."
            raise ValueError(msg)

    def explain(self, x: np.ndarray) -> InteractionValues:
        """Compute feature-wise attributions for a batch of test inputs."""
        x = np.atleast_2d(x)
        inst_values = np.mean(
            [
                self._compute_wknn_shapley_dp(
                    xi,
                    y_val=self.class_index
                    if self.class_index is not None
                    else self.model.predict([xi])[0],
                )
                for xi in x
            ],
            axis=0,
        )
        self.instance_values = inst_values

        n_features = x.shape[1]

        return InteractionValues(
            index="SV",
            max_order=1,
            min_order=1,
            n_players=n_features,
            values=inst_values[:n_features],
            baseline_value=0.0,
        )

    def explain_instances(self, x: np.ndarray) -> InteractionValues:
        """Compute training point attributions for a batch of test instances."""
        x = np.atleast_2d(x)
        values = np.mean([self._compute_exact_by_enumeration(xi) for xi in x], axis=0)

        return InteractionValues(
            index="SV",
            max_order=1,
            min_order=1,
            n_players=self.N,
            values=values,
            baseline_value=0.0,
        )

    def _build_signed_weights(self, dists: np.ndarray, y_val: int) -> np.ndarray:
        """Create signed weights: positive if label == y_val, else negative."""
        weights = 1.0 / (dists + 1e-8) if self.mode == "weighted" else np.ones_like(dists)
        weights /= weights.max()
        return np.where(self.y_train == y_val, weights, -weights)

    def _prepare_distance_rankings(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return distances and index rankings to training points."""
        dists = np.linalg.norm(self.X_train - x_test, axis=1)
        ranks = np.argsort(dists)
        return dists, ranks

    def _compute_wknn_shapley_dp(self, x_test: np.ndarray, y_val: int | None = None) -> np.ndarray:
        """Approximate WKNN Shapley values by subset enumeration with depth pruning."""
        N, K = self.N, self.K
        if y_val is None:
            y_val = (
                self.class_index
                if self.class_index is not None
                else self.model.predict([x_test])[0]
            )
        dists, _ = self._prepare_distance_rankings(x_test)
        signed_weights = self._build_signed_weights(dists, y_val)

        shapley = np.zeros(N)
        M = self.approx_M or N

        for i in range(N):
            contrib = 0.0
            for k_size in range(K):
                num = 0.0
                denom = math.comb(N - 1, k_size)
                if denom == 0:
                    continue
                subset_candidates = [j for j in range(N) if j != i][:M]
                for S in combinations(subset_candidates, k_size):
                    S_list = list(S)
                    S_plus = [*S_list, i]

                    top_S: list[int] = np.argsort(dists[S_list])[: min(K, len(S_list))].tolist() if S_list else []
                    top_S_plus = np.argsort(dists[S_plus])[: min(K, len(S_plus))]

                    vote_S = np.sum(signed_weights[S_list][top_S]) if S_list else 0.0
                    vote_S_plus = np.sum(signed_weights[S_plus][top_S_plus])

                    delta = int(vote_S_plus >= 0) - int(vote_S >= 0)
                    num += delta
                contrib += num / denom
            shapley[i] = contrib / N

        return shapley

    def _compute_exact_by_enumeration(self, x_test: np.ndarray) -> np.ndarray:
        """Compute exact WKNN-Shapley values via full enumeration (slow)."""
        N, K = self.N, self.K
        y_val = (
            self.class_index if self.class_index is not None else self.model.predict([x_test])[0]
        )

        dists = np.linalg.norm(self.X_train - x_test, axis=1)
        weights = 1 / (dists + 1e-8)
        weights /= weights.max()

        shapley = np.zeros(N)

        for i in range(N):
            count = 0.0
            for k_size in range(N):
                subsets = list(combinations([j for j in range(N) if j != i], k_size))
                for S in subsets:
                    S_list = list(S)
                    S_plus_i = [*S_list, i]

                    d_S = dists[S_list] if S_list else np.array([])
                    y_S = self.y_train[S_list] if S_list else np.array([])
                    w_S = weights[S_list] if S_list else np.array([])

                    d_Si = dists[S_plus_i]
                    y_Si = self.y_train[S_plus_i]
                    w_Si = weights[S_plus_i]

                    if len(S_list) == 0:
                        v_S = 0.0
                    else:
                        top_k = np.argsort(d_S)[: min(K, len(S_list))]
                        vote_S = np.sum(np.where(y_S[top_k] == y_val, w_S[top_k], -w_S[top_k]))
                        v_S = float(vote_S >= 0)

                    top_k_i = np.argsort(d_Si)[: min(K, len(S_plus_i))]
                    vote_Si = np.sum(
                        np.where(y_Si[top_k_i] == y_val, w_Si[top_k_i], -w_Si[top_k_i])
                    )
                    v_Si = float(vote_Si >= 0)

                    marginal = v_Si - v_S
                    count += marginal / (N * math.comb(N - 1, k_size))

            shapley[i] = count

        return shapley
