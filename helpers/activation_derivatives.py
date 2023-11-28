from helpers.activation_functions import *


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def d_tanh(x):
    return 1 - tanh(x)**2

def d_relu(x):
    return np.where(x >= 0, 1, 0)

def d_leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, 1, alpha)

def d_linear(x= None):
    return 1

def d_softmax(x):
    return softmax(x) * (1 - softmax(x))

def d_softmax_stable(x):
    return softmax_stable(x) * (1 - softmax_stable(x))

def vectorized_activation_derivative(x: np.ndarray, func: callable):
    return np.vectorize(func)(x)
