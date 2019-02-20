import pickle
import numpy as np
import pandas as pd
from keras.models import load_model
from Libraries import data_preprocessing as pp
from Libraries import data_evaluation as d_Eval

ModelPath = '/media/computations/DATA/ExperimentalData/Runs/156417/'
ModelName = 'cross4L_16model'
m = load_model(ModelPath+ModelName+'.h5')

DataSetPath = '/media/computations/DATA/ExperimentalData/DataFiles/systemABCD/'
#TestDataSets = ['center8s_pad_B_TestDataPandas', 'center8s_pad_B_TrainDataPandas', 'center8s_pad_D_TestDataPandas', 'center8s_pad_D_TrainDataPandas', 'center8s_pad_C_TestDataPandas', 'center8s_pad_C_TrainDataPandas']
#TestDataSets = ['center8s_pad_TestDataPandas']
TestDataSets = ['center8s_pad_D_TestDataPandas']

TestData = pd.DataFrame()
TestLabel = pd.DataFrame()

dropChannels = ['time', 'stopId']

for name in TestDataSets:
    Data = pickle.load(open(DataSetPath + name + '.p', 'rb'))
    TestData = TestData.append(Data[0])
    TestLabel = TestLabel.append(Data[1])

TestData = pp.shape_Data_to_LSTM_format(TestData, dropChannels=dropChannels)
TestLabel = pp.reduceNumpyTD(pp.shape_Labels_to_LSTM_format(TestLabel))

FP, FN, TP, TN = d_Eval.get_overall_results([(TestData, TestLabel)], m)
MCC = d_Eval.get_MCC(FP, FN, TP, TN)