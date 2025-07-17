import numpy as np
from shapiq_student.gaussian_copula_imputer import GaussianCopulaImputer

class DummyModel:
    def __call__(self, X):
        return np.sum(X, axis=1)  #callable
    
    def predict(self, X):
        return np.sum(X, axis=1)

def test_fit_runs_without_error():
    X = np.random.randn(10, 3)
    model = DummyModel()
    imputer = GaussianCopulaImputer(model=model, data=X)
    imputer.fit(X)  # Just check it runs without crashing

def test_impute_output_shape():
    X = np.random.randn(100, 3)
    model = DummyModel()
    imputer = GaussianCopulaImputer(model=model, data=X)
    imputer.fit(X)
    
    x_known = np.array([X[0, 0], X[0, 1]])
    known_idx = [0, 1]
    missing_idx = [2]
    x_missing = imputer.impute(x_known, known_idx, missing_idx)
    #x_known = [0.5, -0.3]
    
    #result = imputer.impute(x_known, known_idx, missing_idx)
    assert x_missing.shape == (1, )
    #assert result.shape == (1,), "Imputed value should be a 1D array with 1 element"

def test_impute_multiple_missing_values():
    X = np.random.randn(100, 4)
    model = DummyModel()
    imputer = GaussianCopulaImputer(model=model, data=X)
    imputer.fit(X)
    
    x_known = np.array([X[0, 0]])
    known_idx = [0]
    missing_idx = [1, 2, 3]
    #x_known = [1.2]
    x_missing = imputer.impute(x_known, known_idx, missing_idx)

    #result = imputer.impute(x_known, known_idx, missing_idx)
    #assert result.shape == (2,), "Should return two imputed values"
    assert x_missing.shape == (3,), "Should return three imputed values"