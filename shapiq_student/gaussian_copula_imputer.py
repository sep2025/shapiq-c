import numpy as np
from scipy.stats import norm

#ECDF: empirical cumulative distribution function




class GaussianCopulaImputer:
    def __init__(self):
        self.ecdfs = []  
        self.inverse_ecdfs = []      #Inverse ECDFs to og val
        self.transformed_data = None
        self.correlation = None      #Correlation matrix

    
    
    
    
    def fit(self, X: np.ndarray):
        """
        Preps Imputer by transforming the data into Gaussian space
        using ECDFs and storing the correlation
        """
        X = np.asarray(X)
        n, d = X.shape
        self.ecdfs = []
        self.inverse_ecdfs = []
        z_data = np.zeros_like(X)

        for j in range(d):
            col = X[:, j]
            sorted_col = np.sort(col)

            #ECDF
            def ecdf_func(x, sorted_col=sorted_col):
                return np.searchsorted(sorted_col, x, side='right') / len(sorted_col)

            #Inverse ECDF
            def inverse_ecdf_func(p, sorted_col=sorted_col):
                p = np.clip(p, 1e-6, 1 - 1e-6)
                idx = np.round(p * (len(sorted_col) - 1)).astype(int)
                return sorted_col[idx]

            self.ecdfs.append(ecdf_func)
            self.inverse_ecdfs.append(inverse_ecdf_func)

            #Convert to z-score
            u = np.array([ecdf_func(xi) for xi in col])
            z = norm.ppf(u)
            z_data[:, j] = z

        #Store transformed data and correlation matrix
        self.transformed_data = z_data
        self.correlation = np.corrcoef(z_data, rowvar=False)

        
        
        
        
        
        
        
        
        
    def impute(self, x_known: np.ndarray, known_idx: list[int], missing_idx: list[int]) -> np.ndarray:

        """
        Impute the missing values given known features using the Gaussian copula.
        """
        from numpy.linalg import pinv

        #Convert values to z-space
        z_known = []
        for i, idx in enumerate(known_idx):
            ecdf = self.ecdfs[idx]
            percentile = ecdf(x_known[i])
            z_known.append(norm.ppf(np.clip(percentile, 1e-6, 1 - 1e-6)))
        z_known = np.array(z_known)

        #Pull out submatrixes from correlation matrix
        Sigma = self.correlation
        Sigma_SS = Sigma[np.ix_(known_idx, known_idx)]
        Sigma_barS_S = Sigma[np.ix_(missing_idx, known_idx)]
        Sigma_barS_barS = Sigma[np.ix_(missing_idx, missing_idx)]
        Sigma_S_barS = Sigma_barS_S.T



        #Application of Gaussian copula imputation
        mu_S = np.zeros(len(known_idx))       
        mu_barS = np.zeros(len(missing_idx))

 
        delta = z_known - mu_S
        inv_Sigma_SS = pinv(Sigma_SS)

        z_barS_cond_mean = mu_barS + Sigma_barS_S @ inv_Sigma_SS @ delta
        z_barS_cond_cov = Sigma_barS_barS - Sigma_barS_S @ inv_Sigma_SS @ Sigma_S_barS

        #Sample from conditional distribution
        z_missing = multivariate_normal(mean=z_barS_cond_mean, cov=z_barS_cond_cov).rvs()

        #If one value is missing then put it in a list
        if len(missing_idx) == 1:
            z_missing = [z_missing] 


        #Convert to original val
        x_missing = []
        for i, idx in enumerate(missing_idx):
            inv_ecdf = self.inverse_ecdfs[idx]
            percentile = norm.cdf(z_missing[i])
            x_val = inv_ecdf(percentile)
            x_missing.append(x_val)

        return np.array(x_missing) 