from helpers.activation_derivatives import *
from helpers.activation_functions import *


activation_data = {
    'sigmoid': {
        'function': sigmoid,
        'derivative': d_sigmoid
    },
    'tanh': {
        'function': tanh,
        'derivative': d_tanh
    },
    'relu': {
        'function': relu,
        'derivative': d_relu
    },
    'leaky_relu': {
        'function': leaky_relu,
        'derivative': d_leaky_relu
    },
    'linear': {
        'function': linear,
        'derivative': d_linear
    },
    'softmax': {
        'function': softmax,
        'derivative': d_softmax
    },
    'softmax_stable': {
        'function': softmax_stable,
        'derivative': d_softmax_stable
    }
}

def get_activation(activation_name):
    if activation_name not in activation_data:
        raise Exception('Invalid activation function')
    return activation_data[activation_name]['function'], activation_data[activation_name]['derivative']


def get_vectorized_activation(activation_name):
    if activation_name not in activation_data:
        raise Exception('Invalid activation function')
    return np.vectorize(activation_data[activation_name]['function']), np.vectorize(activation_data[activation_name]['derivative'])