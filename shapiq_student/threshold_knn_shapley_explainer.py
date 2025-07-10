import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.base import is_classifier, is_regressor


class ThresholdKNNShapleyExplainer:
    def __init__(
        self, model=None, X_train=None, y_train=None, k=None, threshold=None, model_type=None
    ):
        if model is not None:
            self.model = model
            self.X_train = model._fit_X
            self.y_train = model._y
            self.is_classifier = is_classifier(model)
            self.is_regressor = is_regressor(model)
        else:
            if X_train is None or y_train is None:
                raise ValueError("Must provide X_train and y_train if model is not given.")
            if model_type is None:
                raise ValueError(
                    "Must specify model_type ('classifier' or 'regressor') if model is not provided."
                )
            if model_type == "classifier":
                self.model = KNeighborsClassifier(n_neighbors=k or 5)
                self.is_classifier = True
                self.is_regressor = False
            elif model_type == "regressor":
                self.model = KNeighborsRegressor(n_neighbors=k or 5)
                self.is_classifier = False
                self.is_regressor = True
            else:
                raise ValueError("model_type must be 'classifier' or 'regressor'.")
            self.model.fit(X_train, y_train)
            self.X_train = X_train
            self.y_train = y_train
        self.k = k
        self.threshold = threshold
        if self.threshold is None and self.k is not None:
            nbrs = self.model.kneighbors(self.X_train, n_neighbors=self.k)
            self.threshold = np.max(nbrs[0][:, -1])  #set threshold as the max distance to the Kth neighbor
        elif self.threshold is None:
            raise ValueError("Must specify either k or threshold.")

    def _get_within_threshold(self, x):
        dists = np.linalg.norm(self.X_train - x, axis=1)
        return np.where(dists <= self.threshold)[0]  #indices of training points within threshold

    def shapley_values(self, x, n_permutations=100, random_state=None):
        rng = np.random.default_rng(random_state)
        idx_within = self._get_within_threshold(x)
        n = len(idx_within)
        shapley = np.zeros(len(self.X_train))
        if n == 0:
            return shapley  #no points within threshold, contributions will be zero
        for _ in range(n_permutations):
            perm = rng.permutation(idx_within)
            subset = []
            prev_pred = self._predict_with_subset(subset, x)
            for i, idx in enumerate(perm):
                subset.append(idx)
                curr_pred = self._predict_with_subset(subset, x)
                contrib = curr_pred - prev_pred
                shapley[idx] += contrib
                prev_pred = curr_pred
        shapley /= n_permutations
        return shapley

    def _predict_with_subset(self, subset, x):
        if len(subset) == 0:
            if self.is_classifier:
                return 1.0 / len(self.model.classes_)  #same probability, when no data
            else:
                return 0.0  #default regression prediction, when no data
        X_sub = self.X_train[subset]
        y_sub = self.y_train[subset]
        if self.is_classifier:
            temp_knn = KNeighborsClassifier(n_neighbors=min(self.k or 1, len(subset)))
            temp_knn.fit(X_sub, y_sub)
            proba = temp_knn.predict_proba([x])[0]
            orig_pred = self.model.predict([x])[0]
            if orig_pred in temp_knn.classes_:
                class_idx = np.where(temp_knn.classes_ == orig_pred)[0][0]
                return proba[class_idx]  #probability for original predicted class
            else:
                return 0.0
        else:
            temp_knn = KNeighborsRegressor(n_neighbors=min(self.k or 1, len(subset)))
            temp_knn.fit(X_sub, y_sub)
            return temp_knn.predict([x])[0]
