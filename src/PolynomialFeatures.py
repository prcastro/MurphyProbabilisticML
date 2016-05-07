import numpy as np
import itertools


class PolynomialFeatures():
    def __init__(self, n):
        self.n = n

    def fit_transform(self, X):
        if len(X.shape) > 2:
            raise ValueError
        result = X
        last_dim = len(X.shape) - 1
        number_features = X.shape[last_dim]
        combinations = itertools.combinations_with_replacement(
                                 range(number_features), self.n)
        for combination in combinations:
            if len(X.shape) == 1:
                feature = X[combination, ]
            else:
                feature = X[:, combination]
            feature = feature.prod(axis=last_dim, keepdims=True)
            result = np.hstack((result, feature))
        return result
