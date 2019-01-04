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
import random

# list of names for Runs
RunNames = ['Diff10Sec_trained2']
# list of data set names
fileNames = ['smallDiff.p']
# list of models to run
models = ['m2l_allRegu']
path = ''
Savepath = '/home/computations/ExperimentalData/ModelHistoryFiles'


# regularizers to use
regularizers = [L1L2(l1=0.0, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]

DataScaling = True
StopPatience = 30

for i in range(len(regularizers)):
    RunName = RunNames[0]
    filename = fileNames[0]
    model = models[0]
    rreg = regularizers[i]
    breg = random.choice(regularizers)
    kreg = random.choice(regularizers)
    file = open(path + filename, 'rb')
    Data = pickle.load(file)

    dropChannels = ['time', 'stopId', 'trg1', 'n1', 'trot1', 'tlin1', 'tlin2', 'tamb1']
    InputDataSet = pp.shape_Data_to_LSTM_format(Data[0][0], dropChannels)
    input_shape = (None, InputDataSet.shape[2])
    m = model_setup.modelDict[model](input_shape, rreg, breg, kreg)
    histories = list()
    testData = list()

    batch_size = 10
    epochs = 10
    for currData in Data:
        X_ts, labels = pp.balanceSlicedData(currData[0], currData[1], target=50, distributed_Output=True, COLUMN_ID='stopId')
        TrainData, TestData = pp.splitDataPandasFormat(X_ts, labels, split=0.3)
        X = pp.shape_Data_to_LSTM_format(TrainData[0], dropChannels, scale=DataScaling)
        y = pp.shape_Labels_to_LSTM_format(TrainData[1])
        callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=StopPatience, verbose=1,
                                           mode='auto')
        if X.shape[0] >= batch_size:
            testData.append(TestData)
            histories.append(m.fit(X, y, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[callback]))

    print('\n results of ' + RunName + '  on Model ' + model + '  with data set ' + filename)
    print('\n epochs: ' + str(epochs) + '\n batch size: ' + str(batch_size) + '\n stop patience:' + str(
        StopPatience) + ' \n scaling: ' + str(DataScaling))
    print('\n bias-regu: ' + 'l1 %.2f,l2 %.2f' % (breg.l1, breg.l2))
    print('\n kernel-regu: ' + 'l1 %.2f,l2 %.2f' % (kreg.l1, kreg.l2))
    print('\n recu-regu: ' + 'l1 %.2f,l2 %.2f' % (rreg.l1, rreg.l2))

    d_Eval.get_overall_results(testData, m, data_pd=True, dropChannels=dropChannels)

    m_Eval.eval_all(histories, epochs, RunName, m, Savepath, testData)



