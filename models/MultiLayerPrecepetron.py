from typing import List

import numpy as np
from models.LayerInfo import LayerInfo
from models.NetLayer import NetLayer
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import tqdm


class MultiLayerPrecepetron:
    def __init__(
        self,
        num_input: int,
        num_output: int,
        layersInfo: List[LayerInfo],
        activation: ["sigmoid", "tanh"] = "sigmoid",
        classes: List[str] = None,
    ):
        self.num_input = num_input
        self.classes = classes  # for classification
        self.layers = []
        self.layers.append(
            NetLayer(
                LayerInfo(False, num_input, "input", "linear"),
                layersInfo[0].neuron_count,
            )
        )
        for i in range(1, len(layersInfo)):
            self.layers.append(NetLayer(layersInfo[i - 1], layersInfo[i].neuron_count))
        self.layers.append(
            NetLayer(
                LayerInfo(False, layersInfo[-1].neuron_count, "output", activation),
                num_output,
            )
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
            # print("Forwarding layer:", x.shape)
        return x

    def get_layer(self, index):
        return self.layers[index]

    def predict(self, x):
        pred = self.forward(x)
        if self.classes is None:
            return pred, None
        class_names = np.array(self.classes)
        return pred, class_names[np.argmax(pred, axis=1)]

    def backward(self, x, y, y_pred, lr):
        dA = y_pred - y # dA is the error of the last layer
        dW = [] # dW is the gradients of the last layer weights
        db = [] # db is the gradients of the last layer bias
        
        for i in range(len(self.layers) - 1, 0, -1):
            dZ = dA * self.layers[i].d_activation(self.layers[i].output) # dZ is the error of the current layer
            dW.insert(0, np.dot(self.layers[i - 1].output.T, dZ)) # dW is the gradients of the current layer weights
            db.insert(0, np.sum(dZ, axis=0, keepdims=True)) # db is the gradients of the current layer bias
            dA = np.dot(dZ, self.layers[i].W.T) # update dA for the next layer
        # for the first layer
        dZ = dA * self.layers[0].d_activation(self.layers[0].output) 
        dW.insert(0, np.dot(x.T, dZ))
        db.insert(0, np.sum(dZ, axis=0, keepdims=True))
        # update weights and bias for all layers
        for i in range(len(self.layers)):
            self.layers[i].W -= lr * dW[i]
            self.layers[i].b -= lr * db[i]

        return dA, dW, db

    def train(self, x, y, epochs=1000, lr=0.01):
        for epoch in tqdm.tqdm(range(epochs)):
            mse = 0
            for i in range(len(x)):
                y_pred = self.forward(x[i])
                dA, dW, db = self.backward(x[i], y[i], y_pred, lr) # y[i] is one hot encoded
                # dA is the error of the last layer, dW and db are the gradients of the last layer
                mse += np.sum((y[i] - y_pred) ** 2)
            print("Epoch: ", epoch+1, ", MSE Loss: ", mse/len(x))

    def accuracy(self, x, y):
        x = np.array(x)
        y = np.array(y)
        correct = 0
        for i in range(len(x)):
            y_pred = self.forward(x[i])
            if np.argmax(y_pred) == np.argmax(y[i]):
                correct += 1
        return (correct / len(x)) * 100.0

    def confusion_matrix(self, x, y):
        conf_matrix = np.zeros((len(self.classes), len(self.classes)))
        for i in range(len(x)):
            y_pred = self.forward(x[i])
            conf_matrix[np.argmax(y_pred)][np.argmax(y[i])] += 1
        return conf_matrix

    def plot_confusion_matrix(self, x, y, labels_map=None, title="Confusion Matrix"):
        conf_matrix = self.confusion_matrix(x, y)
        df_cm = pd.DataFrame(conf_matrix, index=self.classes, columns=self.classes)
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(title)
        plt.show()

    def print_layers(self):
        for layer in self.layers:
            print(layer)

    def __str__(self):
        return f"MultiLayerPrecepetron(num_input={self.num_input}, hidden_layers={self.layers})"
