"""GaussianCopulaImputer module.

Provides an imputer for missing data estimation using Gaussian copulas.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.linalg import pinv
from scipy.stats import multivariate_normal, norm
from shapiq.games.imputer.base import Imputer

# ECDF: empirical cumulative distribution function


class GaussianCopulaImputer(Imputer):
    """An imputer that uses Gaussian copula to impute missing values.

    It transforms the data into Gaussian space using empirical cumulative distribution functions (ECDFs).
    And uses conditional Gaussian distributions to impute missing values.

    Note:
        - This method is non-parametric and handles non-gaussian marginals by transforming thee marginals.
    """
    def __init__(
        self, model: Any = None, data: np.ndarray | None = None, x: np.ndarray | None = None
    ) -> None:
        """Initializes the GaussianCopulaImputer.

        Args:
            model: A prediction model that takes a 2D array as input and returns a 1D array of outputs.
            data: The data used to fit the imputer.
            x: A reference instance to use for imputation.
        """
        super().__init__(model=model, data=data)
        self._x_internal = x
        self.ecdfs: list[Any] = []
        self.inverse_ecdfs: list[Any] = []  # Inverse ECDFs to og val
        self.transformed_data: np.ndarray | None = None
        self.correlation: np.ndarray | None = None  # Correlation matrix

    @property
    def x(self) -> np.ndarray | None:
        """Get the reference instance x."""
        return self._x_internal

    def fit(self, x: np.ndarray) -> None:
        """Preps Imputer by transforming the data into Gaussian space.

        Using ECDFs and storing the correlation.

        Args:
            x(np.ndarray): The reference instance to be imputed.
        """
        self._x_internal = x
        x = np.asarray(x)
        n, d = x.shape
        self.ecdfs = []
        self.inverse_ecdfs = []
        z_data = np.zeros_like(x)

        for j in range(d):
            col = x[:, j]
            sorted_col = np.sort(col)

            # ECDF
            def ecdf_func(x: float, sorted_col: np.ndarray = sorted_col) -> float:
                return float(np.searchsorted(sorted_col, x, side="right") / len(sorted_col))

            # Inverse ECDF
            def inverse_ecdf_func(p: float, sorted_col=sorted_col) -> float:
                p = np.clip(p, 1e-6, 1 - 1e-6)
                idx = np.round(p * (len(sorted_col) - 1)).astype(int)
                return sorted_col[idx]

            self.ecdfs.append(ecdf_func)
            self.inverse_ecdfs.append(inverse_ecdf_func)

            # Convert to z-score

            u = np.array([ecdf_func(xi) for xi in col])
            u = np.clip(u, 1e-6, 1 - 1e-6)  # this avoids infs
            z = norm.ppf(u)
            z_data[:, j] = z

        # Store transformed data and correlation matrix
        self.transformed_data = z_data
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(z_data, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

        self.correlation = corr

    def impute(
        self, x_known: np.ndarray, known_idx: list[int], missing_idx: list[int]
    ) -> np.ndarray:
        """Impute the missing values given known features using the Gaussian copula.

        Transforms the known values into z-space and samples the missing ones conditionally and transforms them back to the original scale using inverse ECDFs.

        Args:
            x_known (np.ndarray): The known values of the features.
            known_idx (list[int]): Indices of the known features.
            missing_idx (list[int]): Indices of the missing features.

        Returns:
            np.ndarray: The imputed values for the missing features.
        """
        # Convert values to z-space
        z_known = np.array([
            norm.ppf(np.clip(self.ecdfs[idx](x_known[i]), 1e-6, 1 - 1e-6))
            for i, idx in enumerate(known_idx)
        ])

        assert self.correlation is not None, "Correlation matrix must be computed before imputation."

        # Pull out submatrixes from correlation matrix
        Sigma = self.correlation
        Sigma_SS = Sigma[np.ix_(known_idx, known_idx)]
        Sigma_barS_S = Sigma[np.ix_(missing_idx, known_idx)]
        Sigma_barS_barS = Sigma[np.ix_(missing_idx, missing_idx)]
        Sigma_S_barS = Sigma_barS_S.T

        # Application of Gaussian copula imputation
        mu_S = np.zeros(len(known_idx))
        mu_barS = np.zeros(len(missing_idx))

        delta = z_known - mu_S
        inv_Sigma_SS = pinv(Sigma_SS)

        z_barS_cond_mean = mu_barS + Sigma_barS_S @ inv_Sigma_SS @ delta
        z_barS_cond_cov = Sigma_barS_barS - Sigma_barS_S @ inv_Sigma_SS @ Sigma_S_barS

        # Sample from conditional distribution
        z_missing = multivariate_normal(
            mean=z_barS_cond_mean, cov=z_barS_cond_cov, allow_singular=True
        ).rvs()

        # If one value is missing then put it in a list
        if len(missing_idx) == 1:
            z_missing = np.array([z_missing])

        # Convert to original val
        x_missing = []
        for i, idx in enumerate(missing_idx):
            inv_ecdf = self.inverse_ecdfs[idx]
            percentile = norm.cdf(z_missing[i])
            x_val = inv_ecdf(percentile)
            x_missing.append(x_val)

        return np.array(x_missing)

    def __call__(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate the model output for the given coalitions. Using the Gaussian copula imputation.

        Each coalition is a binary vector indicating which features are known (1) and which are missing (0).
        Missing values are imputed based on the conditional distribution derived from the Gaussian copula.

        Args:
            coalitions (np.ndarray): A binary matrix of shape (n_coalitions, n_features) where each row indicates which features are known (1) for each coalition.

        Returns:
            np.ndarray: Model predictions for each coalition.
        """
        values = []
        for coalition in coalitions:
            known_idx = list(np.where(coalition)[0])
            missing_idx = list(np.where(~coalition)[0])

            if len(known_idx) == 0:
                values.append(0.0)
                continue

            assert self.x is not None, "Reference instance x must be set before calling imputer."

            x_known = self.x[0, known_idx]
            x_imputed = self.impute(x_known, known_idx, missing_idx)

            full_x = np.zeros(self.x.shape[1])
            full_x[known_idx] = x_known
            full_x[missing_idx] = x_imputed

            values.append(float(self.predict(full_x.reshape(1, -1))[0]))
        return np.array(values)
