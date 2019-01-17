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
RunNames = ['tests']
# list of data set names
fileNames = ['center6s_pad.p']
# list of models to run
models = ['Model2L_16td']
path = ''
Savepath = ''


# regularizers to use
regularizers = [L1L2(l1=0.01, l2=0.01)]

DataScaling = True
StopPatience = 30

for i in range(len(RunNames)):
    RunName = RunNames[i]
    filename = fileNames[i]
    model = models[0]

    file = open(path+filename, 'rb')
    Data = pickle.load(file)
    dropChannels = ['time', 'stopId']
    X, y = pp.balanceSlicedData(Data[0], Data[1], target=50, distributed_Output=True, COLUMN_ID='stopId')
    TrainData, TestData = pp.splitDataPandasFormat(X, y, split=0.3)
    X_train = pp.shape_Data_to_LSTM_format(TrainData[0], dropChannels, scale=DataScaling)
    y_train = pp.shape_Labels_to_LSTM_format(TrainData[1])
    X_test = pp.shape_Data_to_LSTM_format(TestData[0], dropChannels, scale=DataScaling)
    y_test = pp.shape_Labels_to_LSTM_format(TestData[1])

    epochs = 300
    batch_size = 10

    m = Sequential()
    input_shape = (X_train.shape[1], X_train.shape[2])
    m = model_setup.modelDict[model](input_shape)

    callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=StopPatience, verbose=1, mode='auto')
    history = m.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[callback])

    print('\n results of ' + RunName + '  on Model ' + model + '  with data set ' + filename)
    print('\n epochs: ' + str(epochs) + '\n batch size: ' + str(batch_size) + '\n stop patience:' + str(StopPatience) + ' \n scaling: ' + str(DataScaling))

    d_Eval.get_overall_results([(X_test, y_test)], m)
    m_Eval.eval_all([history], epochs, RunName, m, Savepath, TestData)



