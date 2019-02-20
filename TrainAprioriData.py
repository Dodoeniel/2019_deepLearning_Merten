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
import time

# list of names for Runs
RunNames = ['Bi']
# list of data set names
TrainfileNames = ['center8s_pad_TrainDataNumpy.p']
TestfileNames = ['center8s_pad_TestDataNumpy.p']
# list of models to run
models = ['Model5L_64bi']
path = '/media/computations/DATA/ExperimentalData/DataFiles/center8s_pad/'
Savepath = ''


DataScaling = True
StopPatience = 10


for i in range(len(RunNames)):
    t0 = time.time()
    RunName = RunNames[i]
    Trainfilename = TrainfileNames[0]
    Testfilename = TestfileNames[0]
    model = models[i]

    file = open(path+Trainfilename, 'rb')
    TrainData = pickle.load(file)
    file = open(path+Testfilename, 'rb')
    TestData = pickle.load(file)

    X_train = TrainData[0]
    y_train = pp.reduceNumpyTD(TrainData[1])

    X_test = TestData[0]
    y_test = pp.reduceNumpyTD(TestData[1])

    epochs = 15
    batch_size = 35

    class_weight = {0: 1.,
                    1: 1.
                    }

    m = Sequential()
    input_shape = (X_train.shape[1], X_train.shape[2])
    m = model_setup.modelDict[model](input_shape)

    callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=StopPatience, verbose=1, mode='auto')
    history = m.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[callback], class_weight=class_weight)

    print('\n results of ' + RunName + '  on Model ' + model + '  with data set ' + Testfilename)
    print('\n epochs: ' + str(epochs) + '\n batch size: ' + str(batch_size) + '\n stop patience:' + str(StopPatience) + ' \n scaling: ' + str(DataScaling))

    FP, FN, TP, TN = d_Eval.get_overall_results([(X_test, y_test)], m)
    m_Eval.eval_all([history], epochs, RunName, m, Savepath, TestData)
    MCC = d_Eval.get_MCC(FP, FN, TP, TN )
    print('&y&'+str(MCC)[0:4]+'&'+str(TP)+'&'+str(TN)+'&'+str(FP)+'&'+str(FN)+'\\'+'\\')
    print('\n' + str(t0-time.time()) + 's used')
