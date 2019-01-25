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
RunNames = ['A1L_8', 'A1L_16', 'A1L_32', 'A1L_64', 'A1L_128', 'A1L_256', 'A2L_8', 'A2L_16', 'A2L_32', 'A2L_64', 'A2L_128', 'A2L_256', 'A3L_8', 'A3L_16', 'A3L_32', 'A3L_64', 'A3L_128', 'A3L_256','A4L_8', 'A4L_16', 'A4L_32', 'A4L_64', 'A4L_128', 'A4L_256','A5L_8', 'A5L_16', 'A5L_32', 'A5L_64', 'A5L_128', 'A5L_256','A6L_8', 'A6L_16', 'A6L_32', 'A6L_64', 'A6L_128', 'A6L_256']
# list of data set names
TrainfileNames = ['center8s_pad_A_TrainDataNumpy.p']
TestfileNames = ['center8s_pad_A_TestDataNumpy.p']
# list of models to run
models = ['Model1L_8', 'Model1L_16', 'Model1L_32', 'Model1L_64', 'Model1L_128', 'Model1L_256', 'Model2L_8', 'Model2L_16', 'Model2L_32', 'Model2L_64', 'Model2L_128', 'Model2L_256', 'Model3L_8', 'Model3L_16', 'Model3L_32', 'Model3L_64', 'Model3L_128', 'Model3L_256', 'Model4L_8', 'Model4L_16', 'Model4L_32', 'Model4L_64', 'Model4L_128', 'Model4L_256', 'Model5L_8', 'Model5L_16', 'Model5L_32', 'Model5L_64', 'Model5L_128', 'Model5L_256', 'Model6L_8', 'Model6L_16', 'Model6L_32', 'Model6L_64', 'Model6L_128', 'Model6L_256']
path = '/work/dyn/ctm9918/DataFiles/'
Savepath = ''

SaveInfo = pd.DataFrame(index=RunNames, columns=['MCC', 'FP', 'TP', 'TN', 'FP', 'FN', 'model'])


DataScaling = True
StopPatience = 15
for i in range(len(RunNames)):
    RunName = RunNames[i]
    Trainfilename = TrainfileNames[0]
    Testfilename = TestfileNames[0]
    model = models[i]

    file = open(path + Trainfilename, 'rb')
    TrainData = pickle.load(file)
    file = open(path + Testfilename, 'rb')
    TestData = pickle.load(file)

    X_train = TrainData[0]
    y_train = pp.reduceNumpyTD(TrainData[1])

    X_test = TestData[0]
    y_test = pp.reduceNumpyTD(TestData[1])

    epochs = 1
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

    SaveInfo.loc[RunName, 'MCC'] = MCC
    SaveInfo.loc[RunName, 'TP'] = TP
    SaveInfo.loc[RunName, 'TN'] = TN
    SaveInfo.loc[RunName, 'FP'] = FP
    SaveInfo.loc[RunName, 'FN'] = FN
    SaveInfo.loc[RunName, 'model'] = model




###### find best Models
NumberOfModels = 10
ModelNameList = []
MCC_list = pd.to_numeric(SaveInfo.loc[:, 'MCC'])
while len(ModelNameList) < NumberOfModels:
    index = MCC_list.idxmax()
    ModelNameList.append(index)
    MCC_list = MCC_list.drop([index])

ModelPath = ''
fileName = 'center8s_pad_A_TestDataPandas'
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


thresholdP = 7


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
