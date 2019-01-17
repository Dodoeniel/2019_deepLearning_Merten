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
RunNames = ['c1L_8', 'c1L_16', 'c1L_32', 'c1L_64', 'c1L_128', 'c1L_256', 'c2L_8', 'c2L_16', 'c2L_32', 'c2L_64', 'c2L_128', 'c2L_256', 'c3L_8', 'c3L_16', 'c3L_32', 'c3L_64', 'c3L_128', 'c3L_256','c4L_8', 'c4L_16', 'c4L_32', 'c4L_64', 'c4L_128', 'c4L_256','c5L_8', 'c5L_16', 'c5L_32', 'c5L_64', 'c5L_128', 'c5L_256','c6L_8', 'c6L_16', 'c6L_32', 'c6L_64', 'c6L_128', 'c6L_256']
# list of data set names
TrainfileNames = ['center8s_pad_TrainDataNumpy.p']
TestfileNames = ['center8s_pad_TestDataNumpy.p']
# list of models to run
models = ['Model1L_8', 'Model1L_16', 'Model1L_32', 'Model1L_64', 'Model1L_128', 'Model1L_256', 'Model2L_8', 'Model2L_16', 'Model2L_32', 'Model2L_64', 'Model2L_128', 'Model2L_256', 'Model3L_8', 'Model3L_16', 'Model3L_32', 'Model3L_64', 'Model3L_128', 'Model3L_256', 'Model4L_8', 'Model4L_16', 'Model4L_32', 'Model4L_64', 'Model4L_128', 'Model4L_256', 'Model5L_8', 'Model5L_16', 'Model5L_32', 'Model5L_64', 'Model5L_128', 'Model5L_256', 'Model6L_8', 'Model6L_16', 'Model6L_32', 'Model6L_64', 'Model6L_128', 'Model6L_256']
path = '/work/dyn/ctm9918/DataFiles/'
Savepath = '/work/dyn/ctm9918/ModelHistoryFiles/'


DataScaling = True
StopPatience = 30

for i in range(len(RunNames)):
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

    epochs = 300
    batch_size = 10

    m = Sequential()
    input_shape = (X_train.shape[1], X_train.shape[2])
    m = model_setup.modelDict[model](input_shape)

    callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=StopPatience, verbose=1, mode='auto')
    history = m.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[callback])

    print('\n results of ' + RunName + '  on Model ' + model + '  with data set ' + Testfilename)
    print('\n epochs: ' + str(epochs) + '\n batch size: ' + str(batch_size) + '\n stop patience:' + str(StopPatience) + ' \n scaling: ' + str(DataScaling))

    FP, FN, TP, TN = d_Eval.get_overall_results([(X_test, y_test)], m)
    m_Eval.eval_all([history], epochs, RunName, m, Savepath, TestData)
    MCC = d_Eval.get_MCC(FP, FN, TP, TN )
    print('&y&'+str(MCC)[0:4]+'&'+str(TP)+'&'+str(TN)+'&'+str(FP)+'&'+str(FN)+'\\'+'\\')

