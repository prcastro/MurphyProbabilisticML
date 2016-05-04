import numpy as np
from scipy import stats
from statistics import mode

class KNearestNeighbors():
    def __init__(self, K):
        self.K = K

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, x):
        k_indexes = self._k_nearest_neighbors(x)
        return mode(list(self.y[k_indexes]))

    def _distance(self, x, y):
        return np.linalg.norm(x-y)

    def _k_nearest_neighbors(self, x):
        distances = np.array([self._distance(x, xi) for xi in self.X])
        ordered_indexes = np.array([x[0] for x in sorted(enumerate(distances),
                                                         key = lambda x: x[1])])
        k_nearest_neighbors_indexes = ordered_indexes[0: self.K]
        return k_nearest_neighbors_indexes
