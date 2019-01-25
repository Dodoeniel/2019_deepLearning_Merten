import pickle
from keras.models import load_model
from Libraries import configuration
from Libraries import log_setup as logSetup
from Libraries import prepare_csv as csv_read
from Libraries import data_evaluation as d_eval
from Libraries import data_preprocessing as pp
import numpy as np
import csv
import pandas as pd

path = '/media/computations/DATA/ExperimentalData/DataFiles/center8s_pad/'
fileName = 'center8s_pad_TestDataPandas'
#path = ''
#fileName = 'SmallWindowData'
Data = pickle.load(open(path+fileName+'.p', 'rb'))

## if difftruncated Data:
#X_ts = Data[0][0]
#labels_td = Data[0][1]
## if window data
X_ts = Data[0]
labels_td = Data[1]


ModelPath = '/media/computations/DATA/ExperimentalData/Runs/138798/'

ModelNameList = ['c6L_32', 'c6L_64', 'c5L_32', 'c6L_64', 'c3L_128', 'c2L_16', 'c1L_16']


COLUMN_ID = 'stopId'
#COLUMN_ID = 'sliceId'

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


thresholdP = 6


SavePath = 'Matlab/SingleWindow_AllSameId/'
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

v = 1