import numpy as np
import itertools


class PolynomialFeatures():
    def __init__(self, n):
        self.n = n

    def fit_transform(self, X):
        if len(X.shape) > 2:
            raise ValueError
        result = X
        last_dim_index = len(X.shape) - 1
        number_features = X.shape[last_dim_index]
        for coef in range(2, self.n + 1):
            combinations = itertools.combinations_with_replacement(
                           range(number_features), coef)
            for combination in combinations:
                if len(X.shape) == 1:
                    feature = X[combination, ]
                else:
                    feature = X[:, combination]
                feature = feature.prod(axis=last_dim_index, keepdims=True)
                result = np.hstack((result, feature))
        return result
