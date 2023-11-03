import numpy as np

def classification_accuracy(y_pred, Y):
    return np.sum(Y == y_pred) / len(Y)