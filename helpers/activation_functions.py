import numpy as np


def signum(x):
    return np.where(x >= 0, 1, -1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    # return np.tanh(x)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)


def linear(x):
    return x


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def softmax_stable(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)


def vectorized_activation(x: np.ndarray, func: callable):
    return np.vectorize(func)(x)
