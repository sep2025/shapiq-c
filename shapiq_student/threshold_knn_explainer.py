"""Threshold KNN-Shapley Explainer.

This file implements the threshold variant of the KNN-Shapley Explainer (Wang et al., 2024).
Only training points within a certain "distance threshold" around the tested point are considered.

For every permutation the marginal contribution of each point to the payoff (0 or 1) is measured.
The payoff is 1 when the coalition of training points predicts the target class, otherwise 0.

The Shapley value of each training point is then defined as the average of its marginal contributions
over all permutations.
"""

from __future__ import annotations

from itertools import permutations
import math
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np
from shapiq import Explainer, InteractionValues
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

if TYPE_CHECKING:
    from collections.abc import Iterable


def _mode(labels: Iterable[int | float | str]) -> int | float | str:
    """Return the most frequent label in a list.

    Putting the mode-finding logic in a single, easily visible function makes unit testing
    easier, as well as improving readability.

    Parameters
    ----------
    labels
        Iterable of labels. Can be either ints, floats, or strings.

    Returns:
    -------
    int | float | str
        Label that appears most often, ties are broken by the first occurrence in *labels*.

    Raises:
    ------
    ValueError
        If *labels* is empty.
    """
    labels_list: list[int | float | str] = list(labels)
    if not labels_list:
        msg = "Cannot compute mode of an empty iterable."
        raise ValueError(msg)

    values, counts = np.unique(labels_list, return_counts=True)
    return values[counts.argmax()]


class ThresholdKNNExplainer(Explainer):
    """Exact Shapley explainer using a "threshold" K-nearest neighbours model.

    For a given distance threshold *tau*, only the training points within that
    radius around the test point are considered when forming a prediction. The
    payoff function *v(S)* for a coalition *S* of training points is defined as

    v(S) = 1,  if  majority_class(S ∩ B_tau(x_test)) == target_class
    v(S) = 0,  otherwise

    where *B_tau(x_test)* is the closed ball of radius tau around the test
    point. Shapley values are computed exactly by enumerating all permutations
    of the training data. Practical only for small training sets (Around 10
    points or less).

    Parameters
    ----------
    model
        KNeighborsClassifier or *RadiusNeighborsClassifier* from sklearn, used to
        determind target class (model.predict).
    data
        Training data *X* with shape (n_samples, n_features).
    labels
        Class labels *y* with shape (n_samples,).
    tau
        Distance threshold that defines the neighbourhood, must be positive.
    class_index
        Fixes explained class.  If *None* the predicted class of *model* for test
        point is used. Optional.
    """

    # Only first-order interaction indices (Shapley values per player) are needed.
    index: Literal["SV"] = "SV"
    max_order: int = 1

    def __init__(
        self,
        *,
        model: KNeighborsClassifier | RadiusNeighborsClassifier,
        data: np.ndarray,
        labels: np.ndarray,
        tau: float,
        class_index: int | None = None,
    ) -> None:
        """Validate inputs and cache training data."""
        # Accept both KNeighborsClassifier and RadiusNeighborsClassifier
        if not isinstance(model, (KNeighborsClassifier, RadiusNeighborsClassifier)):
            msg = "*model* must be a KNeighborsClassifier or RadiusNeighborsClassifier."
            raise TypeError(msg)
        if tau <= 0:
            msg = "tau (distance threshold) has to be > 0."
            raise ValueError(msg)
        if data.shape[0] != labels.shape[0]:
            msg = "Mismatch between number of data points and labels."
            raise ValueError(msg)

        super().__init__(model=model, index=self.index, max_order=self.max_order)

        self.X_train: np.ndarray = np.asarray(data, dtype=float)
        self.y_train: np.ndarray = np.asarray(labels)
        self.tau: float = float(tau)
        self.class_index: int | None = class_index
        self.mode = "threshold"

    def explain(  # type: ignore[override]
        self,
        x: np.ndarray | None = None,
    ) -> InteractionValues:
        """Computes Shapley values for one or more test points.

        If more than one test sample is given, resulting Shapley values are
        averaged over all provided samples.

        Parameters
        ----------
        x
            Test sample or samples. Accepts the shape *(n_features,)*, or
            *(n_points, n_features)*. Multiple points are averaged to
            produce a single vector.

        Returns:
        -------
        InteractionValues
            Vector of Shapley values, with one entry per training point. Also marks
            these as first-order contributions (*index='SV'*, *max_order=1*).
        """
        if x is None:
            msg = "x may not be None"  # Ruff TRY003 / EM101: store message firs
            raise TypeError(msg)
        x = np.atleast_2d(np.asarray(x, dtype=float))
        shapley_matrix = np.asarray([self._compute_single(xi) for xi in x])
        shapley_vals = shapley_matrix.mean(axis=0)

        return InteractionValues(
            index=self.index,
            max_order=self.max_order,
            min_order=1,
            n_players=len(self.X_train),
            values=shapley_vals,
            baseline_value=0.0,
        )

    def _compute_single(self, x_test: np.ndarray) -> np.ndarray:
        """Exact Shapley computation for a single test point.

        Parameters
        ----------
        x_test
            Test point with the shape *(n_features,)*.

        Returns:
        -------
        np.ndarray
            A vector of length *n_train*. Element *i* is equal to the Shapley
            value of the training instance *i* in regards to *x_test*.
        """
        n_train = len(self.X_train)
        shapley = np.zeros(n_train, dtype=float)

        @runtime_checkable
        class _Predictor(Protocol):
            def predict(self, X: np.ndarray) -> np.ndarray: ...

        if self.class_index is not None:
            target_class = self.class_index
        elif isinstance(self.model, _Predictor):
            x_test_2d = x_test[np.newaxis, :]          # ndarray, not list
            target_class = self.model.predict(x_test_2d)[0]
        else:  # fall back to callable game interface
            x_test_2d = x_test[np.newaxis, :]
            target_class = (self.model)(x_test_2d)[0]  # type: ignore[arg-type]

        # Pre-compute Euclidean distances once. This preserves the original
        # training-index order and is faster than calling *kneighbors*,
        # which sorts the neighbours.
        dists = np.linalg.norm(self.X_train - x_test, axis=1)

        # *All* permutations of player indices (brute-force Shapley).
        all_perms = permutations(range(n_train))

        for perm in all_perms:
            coalition: list[int] = []
            value_prev = 0.0

            for player in perm:
                coalition.append(player)
                # Identify members of *coalition* within radius tau.
                in_radius = [idx for idx in coalition if dists[idx] <= self.tau]

                if in_radius:
                    y_neigh = self.y_train[in_radius]
                    pred = _mode(y_neigh)
                    value_curr = float(pred == target_class)
                else:
                    value_curr = 0.0  # No information -> payoff 0

                # Marginal contribution of *player*.
                shapley[player] += value_curr - value_prev
                value_prev = value_curr

        shapley /= math.factorial(n_train)
        return shapley
