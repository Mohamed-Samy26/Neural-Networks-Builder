import random
import numpy as np
import activation_functions as af
import metrics as mt
import matplotlib.pyplot as plt


class adaline :
    
    def __init__(self, epochs: int = 10, learning_rate: float = 0.01, bias: float = 0.0):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight = np.random.rand(2)
        self.bias = bias
    
    def forward(self, X: np.ndarray):
        # y = w1*x1 + w2*x2 + b
        return af.linear(np.dot(X, self.weight) + self.bias)
        
    def backward(self, X: np.ndarray, Y:np.ndarray):
        y_pred = self.forward(X)
        error = y_pred - Y
        self.weight -= self.learning_rate * error * X
        
    def train(self, X: np.ndarray, Y:np.ndarray):
        for _ in range(self.epochs):
            for x_i, y_i in zip(X, Y):
                self.backward(x_i, y_i, self.learning_rate)
                
    def predict(self, X: np.ndarray):
        return af.signum(self.forward(X))
    
    def accuracy(self, X: np.ndarray, Y: np.ndarray):
        return mt.classification_accuracy(self.predict(X), Y)
  
    def plot(self, X: np.ndarray, Y: np.ndarray):
        plt.scatter(X[:, 0], X[:, 1], c=Y)
        plt.plot(X[:, 0], -(self.weight[0] * X[:, 0] + self.bias) / self.weight[1])
        plt.show()
        
    def __str__(self):
        return f"Adaline(epochs={self.epochs}, learning_rate={self.learning_rate}, bias={self.bias}), weights={self.weight}"
        