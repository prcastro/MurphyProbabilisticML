import numpy as np


class LinearRegression():
    def __init__(self):
        pass

    def fit(self, X, y):
        self.W = np.linalg.pinv(X.T @ X) @ (X.T @ y)

    def _predict_point(self, x):
        return self.W.T @ x

    def predict(self, X):
        if len(X.shape) == 1:
            return self._predict_point(X)
        elif len(X.shape) == 2:
            return np.array([self._predict_point(xi) for xi in X])
        else:
            raise ValueError
