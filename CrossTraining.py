import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential
from Libraries import data_preprocessing as pp
from Libraries import data_evaluation as d_Eval
from Libraries import model_evaluation as m_Eval
from Libraries import model_setup
from keras import callbacks
from keras.models import load_model
from keras.regularizers import L1L2
import csv


# list of names for Runs
RunNames = ['cross1L_8', 'cross1L_16', 'coss1L_32', 'cross1L_64', 'cross1L_128', 'cross1L_256', 'cross2L_8', 'cross2L_16', 'cross2L_32', 'cross2L_64', 'cross2L_128', 'cross2L_256', 'cross3L_8', 'cross3L_16', 'cross3L_32', 'cross3L_64', 'cross3L_128', 'cross3L_256','cross4L_8', 'cross4L_16', 'cross4L_32', 'cross4L_64', 'cross4L_128', 'cross4L_256','cross5L_8', 'cross5L_16', 'cross5L_32', 'cross5L_64', 'cross5L_128', 'cross5L_256','cross6L_8', 'cross6L_16', 'cross6L_32', 'cross6L_64', 'cross6L_128', 'cross6L_256']
# list of data set names
TrainDataSets = ['center8s_pad_B_TrainDataPandas', 'center8s_pad_D_TrainDataPandas', 'center8s_pad_C_TrainDataPandas', 'center8s_pad_A_TrainDataPandas']

TestDataSets = ['center8s_pad_B_TestDataPandas', 'center8s_pad_D_TestDataPandas', 'center8s_pad_C_TestDataPandas', 'center8s_pad_A_TestDataPandas']

# list of models to run
models = ['Model1L_8', 'Model1L_16', 'Model1L_32', 'Model1L_64', 'Model1L_128', 'Model1L_256', 'Model2L_8', 'Model2L_16', 'Model2L_32', 'Model2L_64', 'Model2L_128', 'Model2L_256', 'Model3L_8', 'Model3L_16', 'Model3L_32', 'Model3L_64', 'Model3L_128', 'Model3L_256', 'Model4L_8', 'Model4L_16', 'Model4L_32', 'Model4L_64', 'Model4L_128', 'Model4L_256', 'Model5L_8', 'Model5L_16', 'Model5L_32', 'Model5L_64', 'Model5L_128', 'Model5L_256', 'Model6L_8', 'Model6L_16', 'Model6L_32', 'Model6L_64', 'Model6L_128', 'Model6L_256']
path = '/media/computations/DATA/ExperimentalData/DataFiles/systemABCD/'
Savepath = ''

SaveInfo = pd.DataFrame(index=RunNames, columns=['MCC', 'FP', 'TP', 'TN', 'FP', 'FN', 'model'])

TestData = pd.DataFrame()
TestLabel = pd.DataFrame()

TrainData = pd.DataFrame()
TrainLabel = pd.DataFrame()

dropChannels = ['time', 'stopId']

for name in TestDataSets:
    Data = pickle.load(open(path + name + '.p', 'rb'))
    TestData = TestData.append(Data[0])
    TestLabel = TestLabel.append(Data[1])
TestData = (TestData, TestLabel)

for name in TrainDataSets:
    Data = pickle.load(open(path + name + '.p', 'rb'))
    TrainData = TrainData.append(Data[0])
    TrainLabel = TrainLabel.append(Data[1])
TrainData = (TrainData, TrainLabel)


DataScaling = True
StopPatience = 15
for i in range(len(RunNames)):
    RunName = RunNames[i]
    model = models[i]

    X_train = pp.shape_Data_to_LSTM_format(TrainData[0], dropChannels=dropChannels)
    y_train = pp.reduceNumpyTD(pp.shape_Labels_to_LSTM_format(TrainData[1]))

    X_test = pp.shape_Data_to_LSTM_format(TestData[0], dropChannels=dropChannels)
    y_test = pp.reduceNumpyTD(pp.shape_Labels_to_LSTM_format(TestData[1]))

    epochs = 300
    batch_size = 10

    m = Sequential()
    input_shape = (X_train.shape[1], X_train.shape[2])
    m = model_setup.modelDict[model](input_shape)

    callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=StopPatience, verbose=1, mode='auto')
    history = m.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[callback])

    print('\n results of ' + RunName + '  on Model ' + model + '  with data set ' + 'all')
    print('\n epochs: ' + str(epochs) + '\n batch size: ' + str(batch_size) + '\n stop patience:' + str(StopPatience) + ' \n scaling: ' + str(DataScaling))

    FP, FN, TP, TN = d_Eval.get_overall_results([(X_test, y_test)], m)
    m_Eval.eval_all([history], epochs, RunName, m, Savepath, TestData)
    MCC = d_Eval.get_MCC(FP, FN, TP, TN )
    print('&y&'+str(MCC)[0:4]+'&'+str(TP)+'&'+str(TN)+'&'+str(FP)+'&'+str(FN)+'\\'+'\\')

    SaveInfo.loc[RunName, 'MCC'] = MCC
    SaveInfo.loc[RunName, 'TP'] = TP
    SaveInfo.loc[RunName, 'TN'] = TN
    SaveInfo.loc[RunName, 'FP'] = FP
    SaveInfo.loc[RunName, 'FN'] = FN
    SaveInfo.loc[RunName, 'model'] = model




###### find best Models
NumberOfModels = 2
ModelNameList = []
MCC_list = pd.to_numeric(SaveInfo.loc[:, 'MCC'])
while len(ModelNameList) < NumberOfModels:
    index = MCC_list.idxmax()
    ModelNameList.append(index)
    MCC_list = MCC_list.drop([index])

ModelPath = ''
fileName = 'center8s_pad_D_TestDataPandas'
Data = pickle.load(open(path+fileName+'.p', 'rb'))

X_ts = Data[0]
labels_td = Data[1]


COLUMN_ID = 'stopId'

label_classified = pp.reduceLabel(labels_td, indexStopID=True).to_frame().rename(columns={0 : 'label'})
emptyDF = pd.DataFrame(np.zeros((len(label_classified), 4)), index=label_classified.index, columns=['P', 'N', 'set', 'eval'])
label_classified = pd.concat([label_classified, emptyDF], axis=1, sort=False)
for ModelName in ModelNameList:
    m = load_model(ModelPath + ModelName + 'model.h5')
    dropChannels = ['time', 'stopId']
    for id in labels_td[COLUMN_ID].unique():
        currX = pp.shape_Data_to_LSTM_format(X_ts.loc[X_ts[COLUMN_ID] == id])
        y_pred = m.predict_classes(currX)
        if bool(y_pred):
            label_classified.loc[id, 'P'] = label_classified.loc[id, 'P']+1
        else:
            label_classified.loc[id, 'N'] = label_classified.loc[id, 'N']+1


thresholdP = 2


SavePath = ''
SaveName = 'Classified_StopIdInfo'
ofile = open(SavePath+SaveName+'.csv', 'w')
writer = csv.writer(ofile, delimiter=",")
Infos = ['stopId', 'label', 'P', 'N', 'set', 'eval']
writer.writerow(Infos)


for id in label_classified.index:
    if label_classified.loc[id, 'P'] >= thresholdP:
        label_classified.loc[id, 'set'] = 1
        if label_classified.loc[id, 'label'] == 1:
            label_classified.loc[id, 'eval'] = 'TP'
        elif label_classified.loc[id, 'label'] == 0:
            label_classified.loc[id, 'eval'] = 'FP'
    else:
        label_classified.loc[id, 'set'] = 0
        if label_classified.loc[id, 'label'] == 0:
            label_classified.loc[id, 'eval'] = 'TN'
        else:
            label_classified.loc[id, 'eval'] = 'FN'
    writer.writerow([id, label_classified.loc[id, 'label'], label_classified.loc[id, 'P'], label_classified.loc[id, 'N'], label_classified.loc[id, 'set'], label_classified.loc[id, 'eval'] ])

ofile.close()
