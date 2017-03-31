import numpy as np
import pandas as pd

from .classification import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score,
                             log_loss,
                             balanced_accuracy,
                             average_precision_score,
                             beta2_fbeta_score,
                             beta5_fbeta_score,
                             beta10_fbeta_score,
                             num_of_false_positive,
                             num_of_false_negative,)
from .regression import (root_mean_squared_error,
                         root_mean_squared_logarithmic_error,
                         mean_absolute_error,
                         mean_absolute_difference_ratio,
                         standard_deviation_difference_ratio)


def summary(metrics, y_true, y_pred_dict,
            sample_weight=None, label_name=None,
            result_prefix="", threshold=0.5):
    """y_pred_dict, y_pred_dict_proba, y_true should be numpy array"""
    if len(y_true) != len(y_pred_dict['pred']):
        raise ValueError("length is not correct!!!!")

    isbiclassifier = False
    ismulticlassifier = False
    isregressor = False

    metric_need_proba_list = [
        'log_loss',
        'roc_auc_score',
        'average_precision_score',
    ]
    if y_pred_dict['ordered_classes'] is not None:
        if y_pred_dict['pred_proba'].shape[1] == 2:
            # case: binary classification (1: positive, -1: negative)
            if 1 and -1 not in y_pred_dict['ordered_classes']:
                raise ValueError("binary classification label should be 1 and -1")
            isbiclassifier = True
            ind_positive_label = np.where(y_pred_dict['ordered_classes'] == 1)[0][0]
            true_prob = y_pred_dict['pred_proba'][:, ind_positive_label]
            y_pred_need_threshold = np.zeros(len(y_pred_dict['pred']))
            y_pred_need_threshold[true_prob >= threshold] = 1
            y_pred_need_threshold[true_prob < threshold] = -1
        elif y_pred_dict['pred_proba'].shape[1] > 2:
            # case: multiclass classification
            ismulticlassifier = True
            y_true_one_hot = pd.get_dummies(
                np.append(y_pred_dict['ordered_classes'], y_true)).astype(np.int32)
            y_true_one_hot = y_true_one_hot[len(y_pred_dict['ordered_classes']): ]
            y_true_one_hot = y_true_one_hot[y_pred_dict['ordered_classes']].values
    else:
        isregressor = True

    if label_name == "log1p_label":
        y_true = np.expm1(y_true)

    result_dict = {}
    for metric, alias in metrics.items():
        # print(metric)
        metric_func = globals()[metric]
        if isbiclassifier:
            average_choice = "binary"
            # print("isbiclassifier")
            if metric in metric_need_proba_list:
                y_pred_output = true_prob
            else:
                y_pred_output = y_pred_need_threshold
            y_true_output = y_true

        elif ismulticlassifier:
            average_choice = "micro"
            # print("ismulticlassifier")
            if metric in metric_need_proba_list:
                # print("need proba")
                y_true_output = y_true_one_hot
                y_pred_output = y_pred_dict['pred_proba']
            else:
                y_true_output = y_true
                y_pred_output = y_pred_dict['pred']
        elif isregressor:
            y_true_output = y_true
            y_pred_output = y_pred_dict['pred']

        try:
            result_dict[result_prefix + alias] = metric_func(
                y_true_output, y_pred_output, sample_weight=sample_weight,
                average=average_choice)
        except:
            result_dict[result_prefix + alias] = metric_func(
                y_true_output, y_pred_output, sample_weight=sample_weight)

    return result_dict

# def main():
#     import pprint
#     metrics = {"root_mean_squared_error": "RMSE",
#                "root_mean_squared_logarithmic_error": "RMSLE"}
#     y_true = np.array([1, 4, 6])
#     y_pred_dict = np.array([1, 2, 3])
#     result = summary(metrics, y_true, y_pred_dict)
#     print(result)
#     pprint.pprint(result, indent=1, width=80, depth=None)

# if __name__ == '__main__':
#     main()
