import numpy as np
from scipy.stats import multivariate_normal
from shapiq.games.imputer.base import Imputer


class GaussianImputer(Imputer):
    """
    ToDo: Translate DocStrings in English and format them better!
    GaussianImputer:
    Schätzt fehlende Werte mit einer bedingten multivariaten Normalverteilung.
    Erweiterung der Imputer-Klasse. Nutzt Hintergrunddaten als Referenz und kann fehlende Werte auf Basis von bekannten Features imputieren.

    Args:
        model: Modell, das Vorhersagen liefert.
        data: Hintergrunddaten als 2D-Array (n_samples, n_features).
        x: Expliziter Punk, der erklärt werden soll.
        sample_size: Anzahl der zu ziehenden Proben. (Standardmäßig 100)
        random_state: Zufallszustand für Reproduzierbarkeit.
        verbose: Flag für ausführliche Ausgabe (Standardmäßig False).

    Special Notes:
        - Behandelt leere Coalitions (alle Feature unbekannt) mit empty_prediction.
        - Behandelt Coalitions mit allen Features (keine Zielvariablen)
    """

    def __init__(self, model, data, x=None, sample_size=100, random_state=None, verbose=False):
        """
        __init__:
           Initialisiert den Imputer mit den gegebenen Parametern.
        """
        print("Initializing GaussianImputer")
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

    def fit(self, x):
        """
        fit:
            Speichert den Erklärungs-Punkt und bereitet den Imputer vor.
        """
        print("Fit GaussianImputer")
        super().fit(x)
        self.cond_idx = None
        self.cond_values = None
        return self

    def __call__(self, coalitions):
        """
        __call__:
            Erwartet Coalition-Vektoren (binär).
            Für jede Coalition:
            - berechnet bedingte Normalverteilung.
            - zieht Stichproben.
            - wendet Modell an.
            - gibt Mittelwert der Vorhersagen zurück.
        """
        print("Calling GaussianImputer with coalitions:")
        results = []

        for coalition in coalitions:
            # 1: Extract conditional and target indices
            cond_idx = np.where(coalition == 1)[0]
            target_idx = np.setdiff1d(np.arange(self.n_players), cond_idx)

            if len(cond_idx) == 0 or len(target_idx) == 0:
                # Coalition is empty or has no target variables
                print(f"Skipping coalition: cond_idx={cond_idx}, target_idx={target_idx}")
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
        self, data, cond_idx, cond_values, n_samples=1, random_state=None
    ):
        """
        sample_conditional_gaussian:
            Computes the conditional mean (μ_{S̄|S}) and conditional covariance (Σ_{S̄|S})
            of a multivariate normal distribution given conditioning variables.
            Draws samples from the resulting conditional distribution.
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
