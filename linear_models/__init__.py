"""
base description
"""

import numpy as np
from tqdm import tqdm

class LinearRegression():
    def __init__(
            self,
            L1: float = 0.01,
            L2: float = 0.001,
            epochs: int = 100,
            lr: float = 1e-4,
        ):
        self.lr = lr
        self.L1 = L1
        self.L2 = L2
        self.epochs = epochs
        self.weights = None

    def fit(self, x: np.array, y: np.array):
        print(x.shape, y.shape)
        self.weights = np.random.normal(loc=0, scale=1.0, size=(x.shape[1] + 1, 1))
        self.weight_history = []
        x_effective = np.hstack([x, np.ones((x.shape[0], 1))])

        for _ in tqdm(range(self.epochs)):
            pred = np.dot(x_effective, self.weights)
            error = pred - y
            grad = np.dot(x_effective.T, error) / x.shape[0]
            l1_grad = self.L1 * np.sign(self.weights)
            l2_grad = self.L2 * self.weights
            self.weights = self.weights - self.lr * grad - l1_grad - l2_grad
            if self.weight_history is not None:
                self.weight_history.append(self.weights.copy())

    def predict(self, x: np.array):
        x_effective = np.hstack([x, np.ones((x.shape[0], 1))])
        return np.dot(x_effective, self.weights)