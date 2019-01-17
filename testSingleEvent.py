import pickle
from keras.models import load_model
from Libraries import configuration
from Libraries import log_setup as logSetup

from Libraries import data_import as dataImport
from Libraries import data_preprocessing as pp
from Libraries import prepare_csv
from Libraries import data_evaluation as d_eval
import csv

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Dropout
from keras import backend as K


#projectName = 'Diff10Sec_trained2model'
#name = 'Diff10Sec_trained2model.h5'
#callDataset = '1051'
stopId = '1852.1051'

modelPath = '/media/computations/DATA/ExperimentalData/Runs/137805/'
modelName = 'c10s_td_padmodel.h5'
model = load_model(modelPath+modelName)

#config = configuration.getConfig(projectName, callDataset)
#logSetup.configureLogfile(config.logPath, config.logName)
#logSetup.writeLogfileHeader(config)
#X_ts, labels = dataImport.loadVerifiedBrakeData(config.eedPath, config.eecPath, config.datasetNumber)
#X_ts = pp.smoothingEedData(X_ts)
#eec = prepare_csv.eec_csv_to_eecData(config.eecPath, callDataset)
#labels = pp.getTimeDistributedLabels(eec, X_ts)

loadPath = '/media/computations/DATA/ExperimentalData/DataFiles/first10s_pad/'
dataSetName = 'first10s_pad'

Data = pickle.load(open(loadPath + dataSetName+'_TestDataPandas.p', 'rb'))
X_ts = Data[0]
labels = Data[1]


X_single = X_ts[X_ts['stopId'] == stopId]
labels_single = labels[labels['stopId'] == stopId]

#dropChannels = ['time', 'stopId', 'trg1', 'n1', 'trot1', 'tlin1', 'tlin2', 'tamb1']
dropChannels = ['time', 'stopId']

X = pp.shape_Data_to_LSTM_format(X_single, dropChannels)
y = pp.shape_Labels_to_LSTM_format(labels_single)
d_eval.countTD_MaxLabel([(X, y)], model)

SavePath = 'Matlab/'
SaveName = modelName+stopId+'_td_label'
ofile = open(SavePath+SaveName+'.csv', 'w')
labels_td_pred = model.predict_classes(X)

writer = csv.writer(ofile, delimiter=",")
writer.writerow([stopId])
for i in range(labels_td_pred.shape[1]):
    writer.writerow([labels_td_pred[0][i][0], y[0][i][0]])

ofile.close()
