"""Library for Evaluation of Input Data"""
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.metrics import confusion_matrix
from Libraries import data_preprocessing as pp
def count_predictions(y_true, y_pred):
    if len(y_pred.shape) == 3:
        FP = 0
        FN = 0
        TP = 0
        TN = 0
        for i in range(y_pred.shape[0]):
            y_true_i = y_true[i].reshape(y_true[i].shape[0],)
            y_pred_i = y_pred[i].reshape(y_pred[i].shape[0],)
            TN_loop, FP_loop, FN_loop, TP_loop = confusion_matrix(y_true_i, y_pred_i, labels=[0, 1]).ravel()
            FP += FP_loop
            FN += FN_loop
            TP += TP_loop
            TN += TN_loop
    else:
            TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return FP, FN, TP, TN

def get_overall_results(test_data, model, data_pd=False, dropChannels=None):
    FP = 1
    FN = 1
    TP = 1
    TN = 1
    for currData in test_data:
        if data_pd:
            X_test = pp.shape_Data_to_LSTM_format(currData[0], dropChannels)
            y = pp.shape_Labels_to_LSTM_format(currData[1])
        else:
            X_test = currData[0]
            y = currData[1]
        y_pred = model.predict_classes(X_test)
        FP_loop, FN_loop, TP_loop, TN_loop = count_predictions(y, y_pred)
        FP += FP_loop
        FN += FN_loop
        TP += TP_loop
        TN += TN_loop
    print('\nMCC: ' + str(get_MCC(FP, FN, TP, TN)))
    print('\n      1_pr 0_pr')
    print('\n 1_tr | ' + str(TP) + '  ' + str(FN))
    print('\n 0_tr | ' + str(FP) + '  ' + str(TN))
    return FP, FN, TP, TN

def get_MCC(FP, FN, TP, TN):
    try:
        return (TP*TN-FP*FN)/(math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    except ZeroDivisionError:
        return 0
    except ValueError:
        TP = TP/100
        FN = FN/100
        FP = FP/100
        TN = TN/100
        return (TP * TN - FP * FN) / (math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))

def countTD_MaxLabel(test_data, model, silent=False):
    for currData in test_data:
        X_test = currData[0]
        y_test = currData[1]
        y_pred = model.predict_classes(X_test)
        y_pred_single = np.zeros((y_pred.shape[0],))
        y_true_single = np.zeros((y_pred.shape[0],))
        for i in range(y_pred.shape[0]):
            y_pred_single[i] = max(y_pred[i])
            y_true_single[i] = max(y_test[i])
        FP, FN, TP, TN = count_predictions(y_true_single, y_pred_single)
        if not silent:
            print('\nMCC: ' + str(get_MCC(FP, FN, TP, TN)))
            print('\n      1_pr 0_pr')
            print('\n 1_tr | ' + str(TP) + '  ' + str(FN))
            print('\n 0_tr | ' + str(FP) + '  ' + str(TN))
    return FP, FN, TP, TN


def get_classified_dataSets(X_ts, labels_td, model, dropChannels, COLUMN_ID = 'stopId'):
    FP_X = X_ts
    FN_X = X_ts
    TP_X = X_ts
    TN_X = X_ts
    FP_y = labels_td
    FN_y = labels_td
    TP_y = labels_td
    TN_y = labels_td

    for stopId in X_ts[COLUMN_ID].unique():
        currX = X_ts[X_ts[COLUMN_ID] == stopId]
        currLabel = labels_td[labels_td[COLUMN_ID] == stopId]
        X = pp.shape_Data_to_LSTM_format(currX, dropChannels)
        y = pp.shape_Labels_to_LSTM_format(currLabel)

        FP, FN, TP, TN = countTD_MaxLabel([(X, y)], model, silent=True)
        if bool(FP):
            FN_X = FN_X[FN_X[COLUMN_ID] != stopId]
            TN_X = TN_X[TN_X[COLUMN_ID] != stopId]
            TP_X = TP_X[TP_X[COLUMN_ID] != stopId]
            FN_y = FN_y[FN_y[COLUMN_ID] != stopId]
            TN_y = TN_y[TN_y[COLUMN_ID] != stopId]
            TP_y = TP_y[TP_y[COLUMN_ID] != stopId]
        elif bool(FN):
            FP_X = FP_X[FP_X[COLUMN_ID] != stopId]
            TN_X = TN_X[TN_X[COLUMN_ID] != stopId]
            TP_X = TP_X[TP_X[COLUMN_ID] != stopId]
            FP_y = FP_y[FP_y[COLUMN_ID] != stopId]
            TN_y = TN_y[TN_y[COLUMN_ID] != stopId]
            TP_y = TP_y[TP_y[COLUMN_ID] != stopId]
        elif bool(TN):
            FP_X = FP_X[FP_X[COLUMN_ID] != stopId]
            FN_X = FN_X[FN_X[COLUMN_ID] != stopId]
            TP_X = TP_X[TP_X[COLUMN_ID] != stopId]
            FP_y = FP_y[FP_y[COLUMN_ID] != stopId]
            FN_y = FN_y[FN_y[COLUMN_ID] != stopId]
            TP_y = TP_y[TP_y[COLUMN_ID] != stopId]
        elif bool(TP):
            FP_X = FP_X[FP_X[COLUMN_ID] != stopId]
            FN_X = FN_X[FN_X[COLUMN_ID] != stopId]
            TN_X = TN_X[TN_X[COLUMN_ID] != stopId]
            FP_y = FP_y[FP_y[COLUMN_ID] != stopId]
            FN_y = FN_y[FN_y[COLUMN_ID] != stopId]
            TN_y = TN_y[TN_y[COLUMN_ID] != stopId]
    return (FP_X, FP_y), (FN_X, FN_y), (TP_X, TP_y), (TN_X, TN_y)

