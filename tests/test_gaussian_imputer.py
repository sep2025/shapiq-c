"""Tests for the GaussianImputer class."""

from __future__ import annotations

import numpy as np

from shapiq_student.gaussian_imputer import GaussianImputer


# copied from tests_grading/test_imputer.py
def dummy_model(x: np.ndarray) -> np.ndarray:
    """A dummy model that returns the sum of the features.

    Note:
        This callable is just here that we satisfy the Imputer's model parameter and tha we can
        check if the Imputer can be called with coalitions and returns a vector of "predictions".

    Args:
        x: Input data as a 2D numpy array with shape (n_samples, n_features).

    Returns:
        A 1D numpy array with the sum of the features for each sample.
    """
    return np.sum(x, axis=1)


def test_init_gaussian_imputer():
    """Test init method of the GaussianImputer class."""
    # Testdata
    rng = np.random.default_rng(42)
    data = rng.random((100, 5))
    sample_size = 100
    random_state = 123
    verbose = True

    # Imputer-Instance
    imputer = GaussianImputer(
        model=dummy_model,
        data=data,
        sample_size=sample_size,
        random_state=random_state,
        verbose=verbose,
    )

    # Checking the instance and attributes
    assert isinstance(imputer, GaussianImputer)
    assert imputer.model is dummy_model
    assert imputer.data.shape == (100, 5), "Data shape should be (100, 5)."
    assert imputer.sample_size == sample_size, "Sample size should be 100."
    assert imputer.random_state == random_state, "Random state should be 123."
    assert imputer.verbose == verbose, "Verbose should be True."
    assert imputer.cond_idx is None, "cond_idx should be None initially."
    assert imputer.cond_values is None, "cond_values should be None initially."


def test_fit_gaussian_imputer():
    """Test fit method of the GaussianImputer class."""
    # Testdata and explanation point
    rng = np.random.default_rng(42)
    data = rng.random((100, 5))
    x_explain = np.array([[0.1, 0.2, -0.3, 0.4, -0.5]])

    # Imputer-Instance
    imputer = GaussianImputer(
        model=dummy_model, data=data, sample_size=100, random_state=123, verbose=True
    )

    # Checking initial state
    assert imputer.cond_idx is None
    assert imputer.cond_values is None
    assert imputer.x is None

    # Fit the imputer with the explanation point
    imputer.fit(x_explain)

    # Checking the state after fit
    assert imputer.x is not None, "x should be set after fit."
    assert imputer.x.shape == (1, 5)
    assert np.array_equal(imputer.x, x_explain), "x should match the explanation point."
    assert np.allclose(imputer.x, x_explain), "x should match the explanation point."

    # Checking after fit
    assert imputer.cond_idx is None, "cond_idx should be set after fit."
    assert imputer.cond_values is None, "cond_values should be set after fit."


def test_call_gaussian_imputer():
    """Test call method of the GaussianImputer class."""
    # Testdata
    rng = np.random.default_rng(42)
    data = rng.random((100, 5))
    sample_size = 100
    random_state = 123
    verbose = True
    n_features = 5
    n_coalitions = 10

    # Imputer-Instance
    imputer = GaussianImputer(
        model=dummy_model,
        data=data,
        sample_size=sample_size,
        random_state=random_state,
        verbose=verbose,
    )

    # Explanation point
    rng = np.random.default_rng(42)
    data = rng.random((1000, n_features))
    x_explain = rng.standard_normal((1, n_features))
    imputer.fit(x_explain)

    # Test-Coalitions
    coalitions = rng.integers(0, 2, size=(n_coalitions, n_features))

    # Check if the GaussianImputer can be called with coalitions
    output = imputer(coalitions=coalitions)

    assert isinstance(output, np.ndarray), "Output should be a numpy array."
    assert output.shape == (n_coalitions,), f"Output shape should be ({n_coalitions},)."
    assert np.all(np.isfinite(output)), "Output should be finite."
    assert np.issubdtype(output.dtype, np.number), "Output should be numeric."


def test_sample_conditional_gaussian():
    """Test sample_conditional_gaussian method of the GaussianImputer class."""
    # Testdata
    rng = np.random.default_rng(0)
    mean = [0, 0, 0]
    cov = [[1, 0.8, 0.5], [0.8, 1, 0.3], [0.5, 0.3, 1]]
    sample_data = rng.multivariate_normal(mean, cov, size=10000)

    # Conditional Index and Values
    cond_idx = [0]
    cond_values = [0.5]

    n_samples = 10000

    imputer = GaussianImputer(
        model=dummy_model, data=sample_data, sample_size=n_samples
    )
    samples, mu_cond, sigma_cond = imputer.sample_conditional_gaussian(
        sample_data, cond_idx, cond_values, n_samples=n_samples, random_state=123
    )

    # Calculate empirical mean and covariance
    empirical_mean = samples.mean(axis=0)
    empirical_cov = np.cov(samples, rowvar=False)

    # Validate empirical mean against theoretical conditional mean
    assert np.allclose(empirical_mean, mu_cond, atol=0.02)

    # Validate empirical covariance against theoretical conditional covariance
    assert np.allclose(empirical_cov, sigma_cond, atol=0.02)
