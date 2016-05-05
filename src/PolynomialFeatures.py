import numpy as np

class PolynomialFeatures():
    def __init__(self, n):
        self.n = n

    def fit_transform(self, X):
        return np.hstack([X**i for i in range(1, self.n +1)])
