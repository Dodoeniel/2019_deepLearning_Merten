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
RunNames = ['A', 'B', 'C', 'D', 'E']
# list of data set names
TrainNames = ['center8s_pad_TrainDataPandas.p']
TestNames = ['center8s_pad_TestDataPandas.p']
# list of models to run
models = ['Model5L_64']
path = '/media/computations/DATA/ExperimentalData/DataFiles/center8s_pad/'
Savepath = '/dropChannels/'

additionalDropChannel = ['torq1', 'p1']
#additionalDropChannel = []
#dropChannels = ['time', 'stopId', 'trg1', 'n1', 'trot1', 'tlin1', 'tlin2', 'tamb1']

DataScaling = True
StopPatience = 30

for i in range(len(RunNames)):
    RunName = RunNames[i]
    model = models[0]

    TrainFile = open(path+TrainNames[0], 'rb')
    TestFile = open(path+TestNames[0], 'rb')

    TrainData = pickle.load(TrainFile)
    TestData = pickle.load(TestFile)

    dropChannels = ['time', 'stopId', 'rh1', 'tempg', 'n1', 'tfld1', 'frc1', 'v1', 'trg1', 'trot1', 'dec1', 'tlin1', 'tlin2', 'tamb1']
    dropChannels.append(additionalDropChannel[i])
    X_train = pp.shape_Data_to_LSTM_format(TrainData[0], dropChannels, scale=DataScaling)
    y_train = pp.reduceNumpyTD(pp.shape_Labels_to_LSTM_format(TrainData[1]))
    X_test = pp.shape_Data_to_LSTM_format(TestData[0], dropChannels, scale=DataScaling)
    y_test = pp.reduceNumpyTD(pp.shape_Labels_to_LSTM_format(TestData[1]))

    epochs = 300
    batch_size = 10

    class_weight = {0: 1.,
                    1: 1.
                    }

    m = Sequential()
    input_shape = (X_train.shape[1], X_train.shape[2])
    m = model_setup.modelDict[model](input_shape)

    callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=StopPatience, verbose=1, mode='auto')
    history = m.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[callback], class_weight=class_weight)

    print('\n results of ' + RunName + '  on Model ' + model + '  with data set ' + TrainNames[0])
    print('\n epochs: ' + str(epochs) + '\n batch size: ' + str(batch_size) + '\n stop patience:' + str(StopPatience) + ' \n scaling: ' + str(DataScaling))
    print('\n drop:' + additionalDropChannel[i])
    d_Eval.get_overall_results([(X_test, y_test)], m)
    m_Eval.eval_all([history], epochs, RunName, m, Savepath, TestData)



