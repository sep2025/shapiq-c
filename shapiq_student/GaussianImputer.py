"""GaussianImputer module.

This module provides an imputer that fills in missing features by sampling from their conditional distribution, assuming the data is multivariate normal.

This approach follows Aas et al. (2021), which showed that conditional Gaussian distributions can be used to impute missing values in a principled way.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal
from shapiq.games.imputer.base import Imputer


class GaussianImputer(Imputer):
    """Imputes missing features assuming a conditional multivariate normal distribution.

    This class extends the base Imputer and uses background data to fill in missing values for features in a data point.
    It assumes that the data follows a multivariate normal distribution.
    It works by conditioning on the known features and sampling the unknown ones accordingly.

    Args:
        model: A predictive model used to generate output values.
        data: Background/reference data as a 2D array (n_samples, n_features).
        x: The specific data point to explain, potentially with missing values.
        sample_size: Number of samples to draw for estimation (default: 100).
        random_state: Random seed for reproducibility (default: None).
        verbose: If True, prints additional information during processing (default: False).

    Notes:
        - Handles "empty" coalitions (all features unknown) using a pre-defined empty prediction.
        - Supports "full" coalitions where no target variables remain.
    """

    def __init__(
        self,
        model: object,
        data: np.ndarray,
        x: np.ndarray | None = None,
        sample_size: int = 100,
        random_state: int | None = None,
        *,
        verbose: bool = False,
    ) -> None:
        """Initializes the GaussianImputer."""
        super().__init__(
            model,
            data,
            x=x,
            sample_size=sample_size,
            random_state=random_state,
            verbose=verbose,
        )
        self.cond_idx = None
        self.cond_values = None

    def fit(self, x: np.ndarray) -> GaussianImputer:
        """Fits the imputer to a specific data point `x`.

        Stores the data point to be explained and rests any previous conditioning information.

        Args:
            x: The data point to explain.

        Returns:
            self: The fitted imputer instance.
        """
        super().fit(x)
        self.cond_idx = None
        self.cond_values = None
        return self

    def __call__(self, coalitions: np.ndarray) -> np.ndarray:
        """Computes model predictions for given coalitions.

        For each coalition:
            - Samples missing features from the conditional Gaussian distribution.
            - Uses the model to predict based on these samples.
            - Returns the mean prediction for each coalition.

        Args:
            coalitions: List or array of binary vectors indication known features for each coalition.

        Returns:
            numpy.ndarray: Array of mean predictions for each coalition.
        """
        results = []

        for coalition in coalitions:
            # 1: Extract conditional and target indices
            cond_idx = np.where(coalition == 1)[0]
            target_idx = np.setdiff1d(np.arange(self.n_players), cond_idx)

            if len(cond_idx) == 0 or len(target_idx) == 0:
                # Coalition is empty or has no target variables
                results.append(self.empty_prediction)
                continue

            # 2: Extract conditional values
            cond_values = self.x[0, cond_idx]

            # 3: Sampling from conditional Gaussian distribution
            samples, _, _ = self.sample_conditional_gaussian(
                self.data,
                cond_idx,
                cond_values,
                n_samples=self.sample_size,
                random_state=self.random_state,
            )

            # 4: Prediction using the model
            prediction = self.predict(samples)
            mean_pred = np.mean(prediction, axis=0)

            # 5: Store the mean prediction
            results.append(mean_pred)

        # 6: Return results as a numpy array
        return np.array(results)

    def sample_conditional_gaussian(
        self,
        data: np.ndarray,
        cond_idx: np.ndarray,
        cond_values: np.ndarray,
        n_samples: int = 1,
        random_state: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Samples from the conditional multivariate normal distribution.

        Computes the conditional mean and covariance given known features, then draws samples for the unknown features accordingly.

        Args:
            data: Background data as a 2D array.
            cond_idx: Indices of the known (conditional) features.
            cond_values: The known feature values for conditional features.
            n_samples: Number of samples to draw (default: 1).
            random_state: Seed for reproducibility (default: None).

        Returns:
            tuple:
            - conditional_samples (np.ndarray): Samples from the conditional distribution.
            - mu_cond (np.ndarray): Conditional mean vector.
            - sigma_cond (np.ndarray): Conditional covariance matrix.

        Notes:
            - Uses the formulas from Aas et al.(2021), Equation (10) and (11) for conditional Gaussian distributions.
        """
        data = np.asanyarray(data)
        cond_values = np.asanyarray(cond_values)

        # Compute the overall mean vector μ and covariance matrix Σ
        mu = np.mean(data, axis=0)
        sigma_full = np.cov(data, rowvar=False)

        # Partition feature indices into S (conditional) and S̄ (target)
        feature_indices = np.arange(data.shape[1])
        target_idx = np.setdiff1d(feature_indices, cond_idx, assume_unique=True)

        # Partition mean vector
        mu_s = np.take(mu, cond_idx)  # μ_S
        mu_sbar = np.take(mu, target_idx)  # μ_{S̄}

        # Partition covariance matrix Σ into blocks
        sigma_ss = sigma_full[np.ix_(cond_idx, cond_idx)]  # Σ_{SS}
        sigma_sbar_sbar = sigma_full[np.ix_(target_idx, target_idx)]  # Σ_{S̄S̄}
        sigma_sbar_s = sigma_full[np.ix_(target_idx, cond_idx)]  # Σ_{S̄S}
        sigma_s_sbar = sigma_sbar_s.T  # Σ_{SS̄}

        # Compute x*_S - μ_S
        delta = cond_values - mu_s

        # Compute pseudoinverse of Σ_{SS}
        inv_sigma_ss = np.linalg.pinv(sigma_ss)

        # Compute conditional mean μ_{S̄|S} (Equation (10) in Aas et al.)
        mu_cond = mu_sbar + sigma_sbar_s @ inv_sigma_ss @ delta

        # Compute conditional covariance Σ_{S̄|S} (Equation (11) in Aas et al.)
        sigma_cond = sigma_sbar_sbar - sigma_sbar_s @ inv_sigma_ss @ sigma_s_sbar

        # Draw samples from N(μ_{S̄|S}, Σ_{S̄|S}) using the computed conditional parameters
        conditional_samples = multivariate_normal(
            mean=mu_cond, cov=sigma_cond, allow_singular=True
        ).rvs(size=n_samples, random_state=random_state)

        # Ensure 2D shape
        if conditional_samples.ndim == 1:
            conditional_samples = conditional_samples.reshape(1, -1)

        return conditional_samples, mu_cond, sigma_cond
