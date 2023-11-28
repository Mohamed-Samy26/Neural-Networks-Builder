from typing import List
import numpy as np
from helpers.activation_data import *
from helpers.activation_functions import *
from models.LayerInfo import LayerInfo


class NetLayer:        
    
    def __init__(self, info:LayerInfo, num_output:int):
        self.number_nodes = info.neuron_count
        self.has_bias = info.has_bias
        self.activation = info.activation
        self.W = np.random.randn(info.neuron_count, num_output)
        self.type = info.layer_type
        
        if info.has_bias:
            self.b = np.random.randn(self.number_nodes) 
        else:
            self.b = None
        self.activation, self.d_activation = get_vectorized_activation(info.activation)
    
    def forward(self, x):
        if self.has_bias:
            x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        return self.activation(np.dot(x, self.W) + self.b)
        
    def backward(self, x, grad):
        grad = grad * self.d_activation(np.dot(x, self.W) + self.b)
        if self.type == 'output':
            return grad
        return np.dot(grad, self.W.T)
        
    def update(self, x, grad, lr):
        grad = np.dot(x.T, grad)
        self.W -= lr * grad
        if self.has_bias:
            self.b -= lr * np.sum(grad, axis=0)
    
    
    def __str__(self):
        return f'NetLayer(number_nodes={self.number_nodes}, has_bias={self.has_bias}, activation={self.activation}, W={self.W}, b={self.b})'