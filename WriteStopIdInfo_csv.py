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



COLUMN_ID = 'stopId'
ModelPath = '/media/computations/DATA/ExperimentalData/Runs/138798/'
ModelName = 'c6L_256model'

wantedSet = 'FP'

if not wantedSet == 'all':
    m = load_model(ModelPath + ModelName + '.h5')
    dropChannels = ['time', 'stopId']
    (FP_X, FP_y), (FN_X, FN_y), (TP_X, TP_y), (TN_X, TN_y) = d_eval.get_classified_dataSets(X_ts, labels_td, m, dropChannels=dropChannels, COLUMN_ID=COLUMN_ID)

if wantedSet == 'FP':
    labels_td = FP_y
elif wantedSet == 'FN':
    labels_td = FN_y
elif wantedSet == 'TN':
    labels_td = TN_y
elif wantedSet == 'TP':
    labels_td = TP_y



SavePath = 'Matlab/SingleWindow_AllSameId/'
SaveName = fileName+'_' + wantedSet + ModelName + '_StopIdInfo'
ofile = open(SavePath+SaveName+'.csv', 'w')
writer = csv.writer(ofile, delimiter=",")

if COLUMN_ID == 'sliceId':
    Infos = ['stopId', 'sliceNr', 'label']
else:
    Infos = ['stopId', 'label']

Param = ['v1', 'p1', 'torq1', 'frc1', 'tempg', 'tfld1', 'rh1']

Infos = Infos + Param
writer.writerow(Infos)

single_label = pp.reduceLabel(labels_td)

def shrinkStopId(stopId):
    DataSet = '1051'
    if COLUMN_ID == 'stopId':
        return stopId[0:len(stopId)-len(DataSet)-len('.')]
    elif COLUMN_ID == 'sliceId':
        return stopId[0:len(stopId)-len(DataSet)-len('.')-len('_')-len('1')]

def mean(ParamId, X_ts, stopId, COLUMN_ID):
    X_curr = X_ts[X_ts[COLUMN_ID] == stopId]
    ParamSeries = pd.Series(X_curr.loc[:, ParamId])
    nonZeroIndex = ParamSeries.nonzero()[0]
    if not len(nonZeroIndex) == 0:
        ParamSeries = ParamSeries.iloc[ParamSeries.first_valid_index():nonZeroIndex[-1]]
        return np.mean(ParamSeries.values)
    else:
        return 0


for i in range(len(single_label)):
    stopId = labels_td[COLUMN_ID].unique()[i]
    if COLUMN_ID == 'stopId':
        currInfos = [shrinkStopId(stopId), single_label.iloc[i, 0]]
    elif COLUMN_ID == 'sliceId':
        ### only valid for up to 10 slices
        sliceNr = stopId[-1]
        currInfos = [shrinkStopId(stopId), sliceNr, single_label.iloc[i, 0]]
    for curr_param in Param:
        currInfos = currInfos + [mean(curr_param, X_ts, stopId, COLUMN_ID)]
    writer.writerow(currInfos)

ofile.close()
