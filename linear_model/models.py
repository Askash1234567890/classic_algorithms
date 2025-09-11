"""
scrip.py

Модуль содержит реализацию простой линейной регрессии.
Включает функции генерации данных, обучения модели и визуализации результата.

Автор: Санёк
Дата: Май 2025
"""

import numpy as np

class LinearRegression():
    def __init__(self, lr=3e-4, lam=1e-5, epochs=1000, loss="MSE"):
        self.lr = lr
        self.lam = lam
        self.epochs = epochs
        self.loss_hist = []
        self.loss = loss
        self.w = None
    
    def loss_func(self, x, w, y):
        data = (y - np.dot(x, w))
        return (np.dot(data.T, data) / x.shape[0]).flatten()[0]
    
    def fit(self, x, y):
        self.w = w = np.ones((x.shape[1] + 1, 1))
        x_stacked = np.hstack([np.ones((x.shape[0], 1)), x])
        self.loss_hist.append(self.loss_func(x_stacked, w, y))

        for epoch in range(self.epochs):
            w -= -self.lr * 2 * np.cos(np.pi * epoch / self.epochs / 2) * np.dot(x_stacked.T, (y - np.dot(x_stacked, w))) / x.shape[0] + self.lam * w
            self.loss_hist.append(self.loss_func(x_stacked, w, y))
        
    def predict(self, x):
        x_stacked = np.hstack([np.ones((x.shape[0], 1)), x])
        return np.dot(x_stacked, self.w)