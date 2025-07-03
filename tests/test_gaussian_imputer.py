import numpy as np
from shapiq_student.GaussianImputer import GaussianImputer

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


def test_sample_conditional_gaussian():

    """ Prüft die Methode sample_conditional_gaussian der Klasse GaussianImputer. """

    np.random.seed(0)
    mean = [0, 0, 0]
    cov = [[1, 0.8, 0.5],
           [0.8, 1, 0.3],
           [0.5, 0.3, 1]]
    sample_data = np.random.multivariate_normal(mean, cov, size=10000)

    # Bedingungsindizes und -werte
    cond_idx = [0]
    cond_values = [0.5]

    n_samples = 10000

    imputer = GaussianImputer(model=dummy_model, data=sample_data, sample_size=n_samples)
    samples, mu_cond, sigma_cond = imputer.sample_conditional_gaussian(
        sample_data, cond_idx, cond_values, n_samples=n_samples, random_state=123
    )

    # Überprüfung der bedingten Verteilung
    empirical_mean = samples.mean(axis=0)
    empirical_cov = np.cov(samples, rowvar=False)
    print(f"Empirical mean: {empirical_mean}")
    print(f"Theoretical mean (mu_cond): {mu_cond}")
    print(f"Empirical covariance: {empirical_cov}")
    print(f"Theoretical covariance (sigma_cond): {sigma_cond}")

    # Vergleich der empirischen Mittelwerte mit dem theoretischen Mittelwert
    assert np.allclose(empirical_mean, mu_cond, atol=0.05), "Empirical mean does not match theoretical mean."

    # Vergleich der Kovarianzmatrix
    assert np.allclose(empirical_cov, sigma_cond, atol=0.05), "Empirical covariance does not match theoretical covariance."
