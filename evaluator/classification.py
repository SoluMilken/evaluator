import functools

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    average_precision_score,
    fbeta_score,
)


beta2_fbeta_score = functools.partial(fbeta_score, beta=2)
beta5_fbeta_score = functools.partial(fbeta_score, beta=5)
beta10_fbeta_score = functools.partial(fbeta_score, beta=10)



def balanced_accuracy(y_true, y_pred):
    true_positive = true_positive(y_true, y_pred)
    true_negative = true_negative(y_true, y_pred)
    positive = len(y_true[y_ture > 0])
    negative = len(y_true[y_ture < 0])
    return ((true_positive/ positive) + (true_negative / negative)) / 2



def num_of_false_positive(y_true, y_pred, average='binary', sample_weight=None):
    """count the number of false positive
    TODO : for multiclass version
    """
    return np.sum((y_true != y_pred) & (y_pred == -1))


def num_of_false_negative(y_true, y_pred, average='binary', sample_weight=None):
    """count the number of false negative
    TODO : for multiclass version
    """
    return np.sum((y_true != y_pred) & (y_pred == 1))
