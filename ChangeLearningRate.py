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
import math
from keras.regularizers import L1L2

# list of names for Runs
RunNames = ['test']
# list of data set names
TrainfileNames = ['center8s_pad_TrainDataNumpy.p']
TestfileNames = ['center8s_pad_TestDataNumpy.p']
# list of models to run
models = ['m5l_256_Rate1']
#path = '/work/dyn/ctm9918/DataFiles/'
#Savepath = '/work/dyn/ctm9918/ModelHistoryFiles/'
path = ''
Savepath = ''

DataScaling = True
StopPatience = 300

def step_decay(epoch):
	initial_lrate = 0.4
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

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
    batch_size = 20

    m = Sequential()
    input_shape = (X_train.shape[1], X_train.shape[2])
    m = model_setup.modelDict[model](input_shape)

    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=StopPatience, verbose=1, mode='auto')
    lrate = callbacks.LearningRateScheduler(step_decay)

    history = m.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[earlyStopping, lrate])

    print('\n results of ' + RunName + '  on Model ' + model + '  with data set ' + Testfilename)
    print('\n epochs: ' + str(epochs) + '\n batch size: ' + str(batch_size) + '\n stop patience:' + str(StopPatience) + ' \n scaling: ' + str(DataScaling))

    FP, FN, TP, TN = d_Eval.get_overall_results([(X_test, y_test)], m)
    m_Eval.eval_all([history], epochs, RunName, m, Savepath, TestData)
    MCC = d_Eval.get_MCC(FP, FN, TP, TN )
    print('&y&'+str(MCC)[0:4]+'&'+str(TP)+'&'+str(TN)+'&'+str(FP)+'&'+str(FN)+'\\'+'\\')

