from typing import List

import numpy as np
from models.LayerInfo import LayerInfo
from models.NetLayer import NetLayer


class MultiLayerPrecepetron:
    def __init__(self, num_input:int, num_output:int, layersInfo: List[LayerInfo], classes: List[str] = None):
        self.num_input = num_input
        self.hidden_layers = []
        self.hidden_layers.append(NetLayer(LayerInfo(False, num_input, 'linear', 'input'), layersInfo[0].neuron_count))
        for i in range(1, len(layersInfo)):
            self.hidden_layers.append(NetLayer(layersInfo[i-1], layersInfo[i].neuron_count))
        self.hidden_layers.append(NetLayer(LayerInfo(False, layersInfo[-1].neuron_count, 'softmax', 'output'), num_output))
        
    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer.forward(x)
        return x
    
    def backward(self, x, grad):
        for layer in reversed(self.hidden_layers):
            grad = layer.backward(x, grad)
        return grad
    
    def update(self, x, grad, lr):
        for layer in self.hidden_layers:
            layer.update(x, grad, lr)
            
    def get_layer(self, index):
        return self.hidden_layers[index]
    
    def predict(self, x):
        pred = self.forward(x)
        if self.classes is None:
            return pred, None
        class_names = np.array(self.classes)
        return pred, class_names[np.argmax(pred, axis=1)]     
    
    def train(self, x, y, epochs, lr):
        for _ in range(epochs):
            y_pred = self.forward(x)
            grad = y_pred - y
            self.backward(x, grad)
            self.update(x, grad, lr)
    
    def accuracy(self, x, y):
        y_pred = self.predict(x)
        return np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)) / len(y)
    
    def __str__(self):
        return f'MultiLayerPrecepetron(num_input={self.num_input}, hidden_layers={self.hidden_layers})'