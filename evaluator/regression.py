import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def root_mean_squared_logarithmic_error(y_true, y_pred):
    return np.sqrt(np.square(np.log1p(y_pred) - np.log1p(y_true)).mean())


def mean_absolute_difference_ratio(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) * (1./ y_true))


def standard_deviation_difference_ratio(y_true, y_pred):
    return np.std((y_true - y_pred) * (1./ y_true))
