"""Unit tests for the ThresholdKNNExplainer implementation."""

from __future__ import annotations

import numpy as np
import pytest
from shapiq import ExactComputer
from sklearn.neighbors import RadiusNeighborsClassifier

from shapiq_student.threshold_knn_explainer import ThresholdKNNExplainer, _mode

ATOL = 1e-8


def test_threshold_knn_instance_shapley() -> None:
    """Compare ThresholdKNNExplainer to exact Shapley values."""
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0, 0, 1])
    x_test = np.array([1.1])  # Close to class 0 points
    explained_class = 0
    tau = 0.6

    radius_clf = RadiusNeighborsClassifier(radius=tau)
    radius_clf.fit(X_train, y_train)

    explainer = ThresholdKNNExplainer(
        model=radius_clf,
        data=X_train,
        labels=y_train,
        tau=tau,
        class_index=explained_class,
    )
    approx = explainer.explain(x_test)

    def game(coalitions: np.ndarray) -> np.ndarray:
        """Payoff is 1 if majority within tau predicts explained_class, otherwise 0."""
        values = []
        for coalition in coalitions:
            idx = np.where(coalition)[0]
            if len(idx) == 0:
                values.append(0.0)
                continue

            # Neighbours within tau
            neigh = [i for i in idx if abs(X_train[i, 0] - x_test[0]) <= tau]
            if not neigh:
                values.append(0.0)
                continue

            y_neigh = y_train[neigh]
            majority = np.bincount(y_neigh).argmax()
            values.append(float(majority == explained_class))
        return np.array(values)

    exact = ExactComputer(n_players=3, game=game)(index="SV")

    error = np.linalg.norm(approx.values - exact.get_n_order_values(1))
    assert error < ATOL


# Extra coverage
def test_threshold_knn_error_paths() -> None:
    """Hit the ValueError branches once so coverage counts them."""
    X = np.array([[0.0]])
    y = np.array([0])
    clf = RadiusNeighborsClassifier(radius=1.0).fit(X, y)

    # Tau must be positive
    with pytest.raises(ValueError, match="tau .* > 0"):
        ThresholdKNNExplainer(model=clf, data=X, labels=y, tau=0.0)

    # Data / label length mismatch
    with pytest.raises(ValueError, match="Mismatch between"):
        ThresholdKNNExplainer(
            model=clf,
            data=np.vstack([X, X]),
            labels=y,
            tau=0.5,
        )

    # Calling *_mode* should raise a ValueError
    with pytest.raises(ValueError, match="empty iterable"):
        _mode([])

def test_threshold_knn_additional_paths() -> None:
    """Cover the remaining branches (≈5 lines) to push module coverage ≥ 92 %."""
    X = np.array([[0.0], [2.0]])        # far apart
    y = np.array([0, 1])
    tau = 0.1                           # so *no* neighbour falls inside radius
    clf = RadiusNeighborsClassifier(radius=tau).fit(X, y)

    explainer = ThresholdKNNExplainer(model=clf, data=X, labels=y, tau=tau, class_index=0)

    # _mode tie-break
    assert _mode(["a", "b", "a", "b"]) == "a"

    # explain with multiple points → averaging branch
    shap_multi = explainer.explain(np.array([[1.0], [1.1]]))
    assert shap_multi.values.shape == (len(X),)

    # no-neighbor situation gives all-zero payoff
    assert np.allclose(shap_multi.values, 0.0)

    # x=None raises TypeError path
    with pytest.raises(TypeError, match="may not be None"):
        explainer.explain(None)  # type: ignore[arg-type]

def test_threshold_knn_rejects_invalid_model() -> None:
    """*model* must be a (Radius)KNeighborsClassifier, anything else raises."""
    class Dummy:  # minimal, has neither predict nor radius
        pass

    X = np.array([[0.0]])
    y = np.array([0])

    with pytest.raises(TypeError, match="must be a KNeighborsClassifier"):
        ThresholdKNNExplainer(model=Dummy(), data=X, labels=y, tau=1.0)
