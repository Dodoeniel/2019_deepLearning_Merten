import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Masking
from keras.models import load_model
from Libraries import data_preprocessing as pp
from Libraries import data_evaluation as eval
from sklearn.model_selection import train_test_split
from Libraries import model_setup
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from keras.utils import plot_model


file = open('FindArchitecture/Dataset_1051/picklefiles/Data', 'rb')
Data = pickle.load(file)

x = Data[0]
m = Sequential()
dropChannels = ['time', 'stopId', 'trg1', 'n1', 'trot1', 'tlin1', 'tlin2', 'tamb1']

InputDataSet = pp.shape_Data_to_LSTM_format(Data[0][0], dropChannels)
input_shape = (None, InputDataSet.shape[2])
m = model_setup.distributed_label(input_shape)
test_data = list()
epochs = 100
for currData in Data:
    seed = 0
    X = pp.shape_Data_to_LSTM_format(currData[0], dropChannels)
    y = pp.shape_Labels_to_LSTM_format(currData[1])
        #y = np.reshape(pp.reduceLabel(y).values, (X.shape[0], 1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    batch_size = 5
    if X_train.shape[0] >= 2:
        hist = m.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=2)
        test_data.append((X_test, y_test))


m.save('my_model.h5')
json_string = m.to_json()
FP, FN, TP, TN = eval.get_overall_results(test_data, m)
print('\nMCC: ' + str(eval.get_MCC(FP, FN, TP, TN)))
print('\n' + str(TP) + '  ' + str(FN))
print('\n' + str(FP) + '  ' + str(TN))

#print('\n%s: %.2f%%' % matthews_corrcoef(y_true, y_pred))
#print('\n%s: %.2f%%' % confusion_matrix(y_true, y_pred))
