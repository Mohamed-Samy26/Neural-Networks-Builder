import numpy as np
import pandas as pd
from helpers import activation_functions as af, metrics as mt
import matplotlib.pyplot as plt


class Adaline:
    def __init__(
        self, epochs: int = 10, learning_rate: float = 0.01, bias: float = 0.0, use_bias=True,
        mse_threshold=None
    ):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight = np.random.rand(2)
        self.use_bias = use_bias
        self.mse_threshold = mse_threshold        
        if use_bias:
            self.bias = bias
        else:
            self.bias = 0
            

    def forward(self, X: np.ndarray):
        # y = w1*x1 + w2*x2 + b
        return af.linear(np.dot(X, self.weight) + self.bias)

    def backward(self, X: np.ndarray, Y: np.ndarray):
        y_pred = self.forward(X)
        error = Y - y_pred  # Calculate the error
        self.weight += self.learning_rate * np.dot(X.T, error.astype(float))  # Update weights
        
        if self.use_bias:
            self.bias += self.learning_rate * np.sum(error.astype(float))  # Update bias
        return np.mean(error**2)

    def train(self, X: np.ndarray, Y: np.ndarray):
        for _ in range(self.epochs):
            mse = self.backward(X, Y)
            if self.mse_threshold is not None and mse < self.mse_threshold:
                break

    def predict(self, X: np.ndarray):
        return af.vectorized_activation(self.forward(X), af.signum)

    def accuracy(self, X: np.ndarray, Y: np.ndarray):
        return mt.classification_accuracy(self.predict(X), Y)

    def plot_decision_boundary(
        self,
        X,
        Y,
        labels=["Class 1", "Class -1"],
        feature_names=["Feature 1", "Feature 2"],
    ):
        X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        Y_np = Y.to_numpy() if isinstance(Y, pd.DataFrame) else Y

        # Plot the decision boundary
        plt.scatter(
            X_np[Y_np == 1][:, 0], X_np[Y_np == 1][:, 1], color="blue", label=labels[0]
        )
        plt.scatter(
            X_np[Y_np == -1][:, 0], X_np[Y_np == -1][:, 1], color="red", label=labels[1]
        )

        x_min, x_max = X_np[:, 0].min() - 1, X_np[:, 0].max() + 1
        y_min, y_max = X_np[:, 1].min() - 1, X_np[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02)
        )
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title("Adaline Decision Boundary")
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self, X, Y, labels=["Class 1", "Class -1"]):
        X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        Y_np = Y.to_numpy() if isinstance(Y, pd.DataFrame) else Y
        y_pred = self.predict(X_np)
        cm = mt.confusion_matrix(y_pred, Y_np)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.get_cmap("Blues"))
        plt.title("Confusion Matrix")
        plt.colorbar()

        # Display the matrix elements as annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )

        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.xticks(
            ticks=[0, 1], labels=["Predicted " + labels[0], "Predicted " + labels[1]]
        )
        plt.yticks(ticks=[0, 1], labels=["Actual " + labels[0], "Actual " + labels[1]])
        plt.show()

    def __str__(self):
        return f"Adaline(epochs={self.epochs}, learning_rate={self.learning_rate}, bias={self.bias}), weights={self.weight}"
