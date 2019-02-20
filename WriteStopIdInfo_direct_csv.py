from Libraries import data_import as dataImport
from Libraries import data_preprocessing as dataPreproc

from Libraries import configuration
from Libraries import log_setup as logSetup
from Libraries import prepare_csv as csv_read
from Libraries import data_preprocessing as pp
import numpy as np
import csv
import pandas as pd

projectName = 'tmp'
callDataset = '1051'

config = configuration.getConfig(projectName, callDataset)

# Setup Logger
logSetup.configureLogfile(config.logPath, config.logName)
logSetup.writeLogfileHeader(config)

# Import verified Time Series Data with Nadines Libraries
X_ts, labels = dataImport.loadVerifiedBrakeData(config.eedPath, config.eecPath, config.datasetNumber)

COLUMN_ID = 'stopId'


SavePath = 'Matlab/Feature_Histogram/'
SaveName = callDataset + '_features'

ofile = open(SavePath+SaveName+'.csv', 'w')
writer = csv.writer(ofile, delimiter=",")

if COLUMN_ID == 'sliceId':
    Infos = ['stopId', 'sliceNr', 'label']
else:
    Infos = ['stopId', 'label']

Param = ['v1', 'p1', 'torq1', 'frc1', 'tempg', 'tfld1', 'rh1']

Infos = Infos + Param
writer.writerow(Infos)

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


for i in range(len(labels)):
    stopId = X_ts[COLUMN_ID].unique()[i]
    if COLUMN_ID == 'stopId':
        currInfos = [shrinkStopId(stopId), labels.loc[stopId]]
    elif COLUMN_ID == 'sliceId':
        ### only valid for up to 10 slices
        sliceNr = stopId[-1]
        currInfos = [shrinkStopId(stopId), sliceNr, labels.loc[stopId]]
    for curr_param in Param:
        currInfos = currInfos + [mean(curr_param, X_ts, stopId, COLUMN_ID)]
    writer.writerow(currInfos)

ofile.close()
