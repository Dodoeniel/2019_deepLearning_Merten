import pickle
from keras.models import load_model
from Libraries import configuration
from Libraries import log_setup as logSetup
from Libraries import model_setup
from Libraries import data_import as dataImport
from Libraries import data_preprocessing as pp
from Libraries import prepare_csv as csv_read
from Libraries import data_evaluation as d_eval
from keras.regularizers import L1L2
import numpy as np
import csv

projectName = 'TestAufschwingen'
callDataset = '1051'
stopId = '1399.1051'
path = '/media/computations/DATA/ExperimentalData/DataFiles/center8s_pad/'
fileName = 'center8s_pad'
config = configuration.getConfig(projectName, callDataset)

# Setup Logger
logSetup.configureLogfile(config.logPath, config.logName)
logSetup.writeLogfileHeader(config)


Data = pickle.load(open(path+fileName+'.p', 'rb'))
X_ts = Data[0]
labels_td = Data[1]

COLUMN_ID = 'stopId'

#dropChannels = ['time', 'stopId', 'trg1', 'n1', 'trot1', 'tlin1', 'tlin2', 'tamb1']
dropChannels = ['time', 'stopId']

X_without, X_with, y_without, y_with = pp.getUpDownLabel_pandas(Data[0], Data[1], COLUMN_ID=COLUMN_ID)

y_with_np = pp.getUpDownLabels_np_v2(y_with, COLUMN_ID=COLUMN_ID)
y_without_np = pp.getUpDownLabels_np_v2(y_without, COLUMN_ID=COLUMN_ID)

X_with_np = pp.shape_Data_to_LSTM_format(X_with, dropChannels)
X_without_np = pp.shape_Data_to_LSTM_format(X_without, dropChannels)
input_shape = (None, X_with_np.shape[2])

m = model_setup.modelDict['m2l_16_UpDown'](input_shape)
epochs = 3
batch_size = 1

m.fit(X_with_np, y_with_np, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=2)


def checkTimeSeriesUpDown(model, X_without_np):
    label_pred = model.predict_classes(X_without_np)
    up_count = 0
    down_count = 0
    false_squeal = 0
    for i in range(label_pred.shape[0]):
        if (2 in label_pred[i]):
            up_count += 1
        if (3 in label_pred[i]):
            down_count += 1
        if (1 in label_pred[i]):
            false_squeal += 1
    return  up_count, down_count, false_squeal

v = checkTimeSeriesUpDown(m, X_without_np)
print('Up Count:' + str(v[0]))
print('Down Count:' + str(v[1]))
print('false Squeal:' + str(v[2]))