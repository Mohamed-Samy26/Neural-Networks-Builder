from typing import List
import numpy as np
from helpers.activation_data import *
from helpers.activation_functions import *
from models.LayerInfo import LayerInfo


class NetLayer:        
    
    def __init__(self, info:LayerInfo, num_output:int):
        self.number_nodes = info.neuron_count
        self.has_bias = info.has_bias
        self.W = np.random.randn(info.neuron_count, num_output)
        self.type = info.layer_type
        self.b = 0
        self.activation, self.d_activation = get_vectorized_activation(info.activation)
        self.output = None
    
    def forward(self, x):
        # print("in: ", x.shape)
        net =  self.activation(np.dot(x, self.W) + self.b)
        self.output = net
        # print("out: ",net.shape, "Forwarding layer", self.type, "Dim", self.W.shape )
        return net
              
    def __str__(self):
        return f'NetLayer(number_nodes={self.number_nodes}, has_bias={self.has_bias}, activation={self.activation}, W={self.W.shape}, b={self.b.shape if self.b is not None else None})'