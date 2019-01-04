import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential
from Libraries import data_preprocessing as pp
from Libraries import data_evaluation as eval
from Libraries import model_evaluation as m_Eval
from sklearn.model_selection import train_test_split
from Libraries import model_setup


RunName = 'bla'
file = open('Data.p', 'rb')
Data = pickle.load(file)

x = Data[0]
m = Sequential()
dropChannels = ['time', 'stopId', 'trg1', 'n1', 'trot1', 'tlin1', 'tlin2', 'tamb1']

InputDataSet = pp.shape_Data_to_LSTM_format(Data[0][0], dropChannels)
input_shape = (None, InputDataSet.shape[2])
m = model_setup.distributed_into_one(input_shape)
test_data = list()
histories = list()
epochs = 1


for currData in Data:
    seed = 0
    X = pp.shape_Data_to_LSTM_format(currData[0], dropChannels)
    #y = pp.shape_Labels_to_LSTM_format(currData[1])
    y = np.reshape(pp.reduceLabel(currData[1]).values, (X.shape[0], 1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed)
    batch_size = 10
    if X_train.shape[0] >= 2:
        histories.append(m.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=2))
        test_data.append((X_test, y_test))

m_Eval.eval_all(histories, epochs, RunName, m)

FP, FN, TP, TN = eval.get_overall_results(test_data, m)

