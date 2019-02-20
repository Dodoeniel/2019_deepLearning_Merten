import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential
from Libraries import data_preprocessing as pp
from Libraries import data_evaluation as d_Eval
from Libraries import model_evaluation as m_Eval
from sklearn.model_selection import train_test_split
from Libraries import model_setup
from keras import callbacks
from keras.regularizers import L1L2

# list of names for Runs
RunNames = ['dropStopId']
# list of data set names
TrainfileNames = ['center8s_pad_TrainDataPandas.p']
TestfileNames = ['center8s_pad_TestDataPandas.p']
# list of models to run
models = ['Model5L_64']
path = '/media/computations/DATA/ExperimentalData/DataFiles/center8s_pad/'
Savepath = ''


dropStopId = ['212', '660', '666', '667', '670', '671', '674', '675', '676', '677', '681', '688', '689', '690', '695',
              '699', '709', '713', '717', '718', '723', '727', '731', '732', '733', '746', '1122', '1124', '1126',
              '1128', '1129', '1131', '1135', '1140', '1142', '1143', '1144', '1153', '1157', '1158', '1159', '1163',
              '1167', '1168', '1170', '1171', '1172', '1173', '1177', '1181', '1182', '1185', '1186', '1187', '1191',
              '1608', '1804', '1806', '1810', '1818', '1820']

#dropStopId = []

DataScaling = True
StopPatience = 15


def dropStopId_fromSet(X_ts, labels_td, dropStopId, COLUMN_ID='stopId', dataSetNr='1051'):
    for stopId in dropStopId:
        stopId = stopId + '.' + dataSetNr
        X_ts = X_ts[X_ts[COLUMN_ID] != stopId]
        labels_td = labels_td[labels_td[COLUMN_ID] != stopId]
    return X_ts, labels_td


for i in range(len(RunNames)):
    RunName = RunNames[i]
    Trainfilename = TrainfileNames[0]
    Testfilename = TestfileNames[0]
    model = models[i]

    file = open(path+Trainfilename, 'rb')
    TrainData = pickle.load(file)
    file = open(path+Testfilename, 'rb')
    TestData = pickle.load(file)

    TrainData = dropStopId_fromSet(TrainData[0], TrainData[1], dropStopId=dropStopId)
    TestData = dropStopId_fromSet(TestData[0], TestData[1], dropStopId=dropStopId)

    X_train = pp.shape_Data_to_LSTM_format(TrainData[0])
    y_train = pp.reduceNumpyTD(pp.shape_Labels_to_LSTM_format(TrainData[1]))

    X_test = pp.shape_Data_to_LSTM_format(TestData[0])
    y_test = pp.reduceNumpyTD(pp.shape_Labels_to_LSTM_format(TestData[1]))

    epochs = 300
    batch_size = 10

    m = Sequential()
    input_shape = (X_train.shape[1], X_train.shape[2])
    m = model_setup.modelDict[model](input_shape)

    class_weight = {0: 2.,
                    1: 1.
                    }

    callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=StopPatience, verbose=1, mode='auto')
    history = m.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[callback], class_weight=class_weight)

    print('\n results of ' + RunName + '  on Model ' + model + '  with data set ' + Testfilename)
    print('\n epochs: ' + str(epochs) + '\n batch size: ' + str(batch_size) + '\n stop patience:' + str(StopPatience) + ' \n scaling: ' + str(DataScaling))

    FP, FN, TP, TN = d_Eval.get_overall_results([(X_test, y_test)], m)
    m_Eval.eval_all([history], epochs, RunName, m, Savepath, TestData)
    MCC = d_Eval.get_MCC(FP, FN, TP, TN )
    print('&n&'+str(MCC)[0:4]+'&'+str(TP)+'&'+str(TN)+'&'+str(FP)+'&'+str(FN)+'\\'+'\\')



