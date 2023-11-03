import numpy as np

def classification_accuracy(y_pred, Y):
    return np.sum(Y == y_pred) / len(Y)

def confusion_matrix(y_pred, Y):
    true_positive = np.sum((Y == 1) & (y_pred == 1))
    true_negative = np.sum((Y == 0) & (y_pred == 0))
    false_positive = np.sum((Y == 0) & (y_pred == 1))
    false_negative = np.sum((Y == 1) & (y_pred == 0))
    return np.array([[true_positive, false_positive],
                     [false_negative, true_negative]])