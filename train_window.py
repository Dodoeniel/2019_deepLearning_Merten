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

# list of names for Runs
RunNames = ['m1l8' , 'm1l16', 'm1l32', 'm1l64', 'm1l128', 'm1l256']
# list of data set names
fileNames = ['SmallWindowData.p']
# list of models to run
models = ['Model1L_8', 'Model1L_16', 'Model1L_32', 'Model1L_64', 'Model1L_128', 'Model1L_256']
path = ''
Savepath = '/home/computations/ExperimentalData/ModelHistoryFiles'

DataScaling = True
StopPatience = 30

for i in range(len(RunNames)):
    RunName = RunNames[i]
    filename = fileNames[0]
    model = models[i]
    file = open(path + filename, 'rb')
    Data = pickle.load(file)
    dropChannels = ['time', 'stopId', 'trg1', 'n1', 'trot1', 'tlin1', 'tlin2', 'tamb1']
    X, y = pp.balanceSlicedData(Data[0], Data[1], target=50, distributed_Output=False)

    X = pp.shape_Data_to_LSTM_format(X, dropChannels, scale=DataScaling)
    # y = pp.reduceLabel(y).values           ## not needed if distributed Output in balance sliced Data == false
    y = y.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    class_weight = pp.getClassWeight_Dict(y)

    epochs = 1
    batch_size = 1

    m = Sequential()
    input_shape = (X.shape[1], X.shape[2])
    m = model_setup.modelDict[model](input_shape)

    callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=StopPatience, verbose=1, mode='auto')

    history = m.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=2,
                    class_weight=class_weight, callbacks=[callback])

    print('\n results of ' + RunName + '  on Model ' + model + '  with data set ' + filename)
    print('\n epochs: ' + str(epochs) + '\n batch size: ' + str(batch_size) + '\n stop patience:' + str(
        StopPatience) + ' \n scaling: ' + str(DataScaling))
    print('\n squeals are weighted with ' + str(class_weight[1]))
    d_Eval.get_overall_results([(X_test, y_test)], m)
    m_Eval.eval_all([history], epochs, RunName, m, Savepath, (X_test, y_test))



