import numpy as np

def classification_accuracy(y_pred, Y):
    return np.sum(Y == y_pred) / len(Y)

def f1_score(y_pred, Y):
    true_positive = np.sum((Y == 1) & (y_pred == 1))
    false_positive = np.sum((Y == 0) & (y_pred == 1))
    false_negative = np.sum((Y == 1) & (y_pred == 0))
    return 2 * true_positive / (2 * true_positive + false_positive + false_negative)

def precision(y_pred, Y):
    true_positive = np.sum((Y == 1) & (y_pred == 1))
    false_positive = np.sum((Y == 0) & (y_pred == 1))
    return true_positive / (true_positive + false_positive)

def recall(y_pred, Y):
    true_positive = np.sum((Y == 1) & (y_pred == 1))
    false_negative = np.sum((Y == 1) & (y_pred == 0))
    return true_positive / (true_positive + false_negative)

def confusion_matrix(y_pred, Y, negative_class=-1, positive_class=1):
    true_positive = np.sum((Y == positive_class) & (y_pred == positive_class))
    true_negative = np.sum((Y == negative_class) & (y_pred == negative_class))
    false_positive = np.sum((Y == negative_class) & (y_pred == positive_class))
    false_negative = np.sum((Y == positive_class) & (y_pred == negative_class))
    return np.array([[true_positive, false_positive],
                     [false_negative, true_negative]])