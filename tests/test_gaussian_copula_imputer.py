"""Tests for the GaussianCopulaImputer class."""

from __future__ import annotations

import numpy as np

from shapiq_student.gaussian_copula_imputer import GaussianCopulaImputer


class DummyModel:
    """A dummy model for testing purposes."""

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Dummy model that sums the features."""
        return np.sum(X, axis=1)  # callable

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dummy predict method that sums the features."""
        return np.sum(X, axis=1)


def test_init_gaussian_copula_imputer():
    """Test Initialization of the GaussianCopulaImputer."""
    rng = np.random.default_rng(42)
    data = rng.random((100, 5))
    model = DummyModel()

    imputer = GaussianCopulaImputer(model=model, data=data)

    assert isinstance(imputer, GaussianCopulaImputer), (
        "Should be an instance of GaussianCopulaImputer"
    )
    assert imputer.model is model, "Model should be set correctly"
    assert imputer.data.shape == (100, 5), "Data shape should be (100, 5)"
    assert imputer.transformed_data is None, "transformed_data should be None initially"
    assert imputer.correlation is None, "correlation should be None initially"


def test_fit_gaussian_copula_imputer():
    """Test the fit method of the GaussianCopulaImputer."""
    rng = np.random.default_rng(0)
    data = rng.random((100, 4))
    model = DummyModel()
    imputer = GaussianCopulaImputer(model=model, data=data)
    imputer.fit(data)  # fit should complete without error


def test_impute_output_shape_single_missing():
    """Test the impute method with a single missing value."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((100, 3))
    model = DummyModel()
    imputer = GaussianCopulaImputer(model=model, data=data)
    imputer.fit(data)

    known_idx = [0, 1]
    missing_idx = [2]
    x_known = data[0, known_idx]  # Known features
    x_missing = imputer.impute(x_known, known_idx, missing_idx)

    assert isinstance(x_missing, np.ndarray), "Imputed value should be a numpy array"
    assert x_missing.shape == (1,)


def test_impute_multiple_missing_values():
    """Test the impute method with multiple missing values."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((100, 4))
    model = DummyModel()
    imputer = GaussianCopulaImputer(model=model, data=data)
    imputer.fit(data)

    known_idx = [0]
    missing_idx = [1, 2, 3]
    x_known = data[0, known_idx]
    x_missing = imputer.impute(x_known, known_idx, missing_idx)

    assert isinstance(x_missing, np.ndarray), "Imputed values should be a numpy array"
    assert x_missing.shape == (3,), "Should return three imputed values"


def test_call_gaussian_copula_imputer():
    """Test the call method of the GaussianCopulaImputer."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((200, 5))
    x_explain = rng.standard_normal((1, 5))  # Known features
    coalitions = rng.integers(0, 2, size=(10, 5))  # Random coalitions
    model = DummyModel()
    imputer = GaussianCopulaImputer(model=model, data=data)
    imputer.fit(x_explain)  # Fit with known features
    imputer._x_internal = x_explain  # Set the internal x for testing  # noqa: SLF001

    predictions = imputer(coalitions)

    assert isinstance(predictions, np.ndarray), "Result should be a numpy array"
    assert predictions.shape == (10,), "Should return predictions for each coalition"
    assert np.all(np.isfinite(predictions)), "All predictions should be finite numbers"


def test_get_x_property():
    """Test the x property of the GaussianCopulaImputer."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((10, 3))
    model = DummyModel()
    imputer = GaussianCopulaImputer(model=model, data=data)
    imputer.fit(data)
    result = imputer.x
    assert result.shape == (10, 3)


def test_call_with_empty_coalition():
    """Test the call method with an empty coalition."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((10, 3))
    model = DummyModel()
    imputer = GaussianCopulaImputer(model=model, data=data)

    x_explain = data[0:1, :]  # Known features
    imputer.fit(x_explain)

    coalitions = np.array([[0, 0, 0]])  # Empty coalition
    result = imputer(coalitions)

    assert result.shape == (1,), "Should return a single prediction for the coalition"
    assert result[0] == 0.0, "Prediction for empty coalition should be 0.0"


def test_dummy_model_predict_directly():
    """Test that DummyModel.predict works."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((10, 3))
    model = DummyModel()

    # Directly call the model
    predictions = model.predict(data)

    assert predictions.shape == (10,), "Predictions shape should match number of samples"
    assert np.allclose(predictions, np.sum(data, axis=1)), (
        "Predictions should match the sum of features"
    )
