import numpy as np

class LinearRegression():
    def __init__(self):
        pass

    def fit(self, X, y):
        self.W = np.linalg.pinv(X.T @ X) @ (X.T @ y) # the @ operator calls __matmul__

    def _predict_point(self, x):
        return self.W.T @ x 

    def predict(self, x):
        if len(x.shape) == 1:
            return self._predict_point(x)
        elif len(x.shape) == 2:
            return np.array([self._predict_point(x_i) for x_i in x])
        else:
            raise ValueError
