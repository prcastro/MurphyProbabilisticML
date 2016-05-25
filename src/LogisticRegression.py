import numpy as np


class LogisticRegression():
    def __init__(self, learnrate=0.01, eps=0.1):
        self.learnrate = learnrate
        self.eps = eps

    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def _predict_point(self, x):
        theta = self._sigmoid(self.w * x)
        if theta > 0.5:
            return 1
        return 0

    def _predict_proba(self, X):
        prob_class_1 = self._sigmoid(X @ self.w)
        return np.hstack((prob_class_1, 1 - prob_class_1))

    def predict_proba(self, X):
        bias = np.ones((X.shape[0], 1))
        X = np.hstack((bias, X))
        return self._predict_proba(X)

    def predict(self, X):
        bias = np.ones((X.shape[0], 1))
        X = np.hstack((bias, X))
        prob_class_1 = self._predict_proba(X)[:,0:1]
        return (prob_class_1 > 0.5).astype(int)

    def _gradientloss(self, w, X, y):
        m = X.shape[0]
        return ((self._sigmoid(X @ w) - y ).T @ X)/m

    def _gradientdescent(self, w0, learnrate, grad, eps):
        w = w0
        while True:
            w_new = w - learnrate * grad(w)
            if np.linalg.norm(w_new - w) < eps:
                return w
            w = w_new

    def fit(self, X, y):
        bias = np.ones((X.shape[0], 1))
        X = np.hstack((bias, X))
        self.w = np.random.rand(X.shape[1],1)
        self.w = self._gradientdescent(self.w, self.learnrate, lambda w : self._gradientloss(w, X, y), self.eps)
