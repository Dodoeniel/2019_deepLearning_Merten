import pickle
from keras.models import load_model
from Libraries import configuration
from Libraries import log_setup as logSetup

from Libraries import data_import as dataImport
from Libraries import data_preprocessing as pp
from Libraries import prepare_csv as csv
from Libraries import data_evaluation as d_eval


runName = 'Diff10Sec_trained2'
model = load_model(runName+'model.h5')
Data = pickle.load(open(runName+'_TestData.p', 'rb'))

X_test = Data[0][0]
y_test = Data[0][1]

COLUMN_ID = 'stopId'
d_eval.countTD_MaxLabel([(X_test, y_test)], model)
#for stopId in X_test[COLUMN_ID].unique():
#    X_single = X_test[X_test['stopId'] == stopId]
#    labels = y_test[y_test['stopId'] == stopId]
#    dropChannels = ['time', 'stopId', 'trg1', 'n1', 'trot1', 'tlin1', 'tlin2', 'tamb1']
#    X = pp.shape_Data_to_LSTM_format(X_single, dropChannels)
#    y = pp.shape_Labels_to_LSTM_format(labels)
#    d_eval.countTD_MaxLabel([(X, y)], model)