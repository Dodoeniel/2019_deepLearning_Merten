import pickle
from keras.models import load_model
from Libraries import configuration
from Libraries import log_setup as logSetup

from Libraries import data_import as dataImport
from Libraries import data_preprocessing as pp
from Libraries import prepare_csv as csv_read
from Libraries import data_evaluation as d_eval
import csv

projectName = 'TestAufschwingen'
callDataset = '1051'
stopId = '1399.1051'
path = '/media/computations/DATA/ExperimentalData/'

config = configuration.getConfig(projectName, callDataset)

# Setup Logger
logSetup.configureLogfile(config.logPath, config.logName)
logSetup.writeLogfileHeader(config)


Data = pickle.load(open('SmallWindowData.p', 'rb'))
X_ts = Data[0]
labels_td = Data[1]

Model_name = 'Diff10Sec_trainedmodel.h5'
model = load_model(Model_name)
COLUMN_ID = 'sliceId'

dropChannels = ['time', 'stopId', 'trg1', 'n1', 'trot1', 'tlin1', 'tlin2', 'tamb1']

FP, FN, TP, TN = d_eval.get_classified_dataSets(X_ts, labels_td, model, dropChannels=dropChannels, COLUMN_ID=COLUMN_ID)

def getNrOfBeginEnding(ClassifiedDataSet_label, nrBegin=1, nrEnd=1):
    begin = 0
    end = 0

    if not ClassifiedDataSet_label.empty:
        for id in ClassifiedDataSet_label[COLUMN_ID].unique():
            currLabel = ClassifiedDataSet_label[ClassifiedDataSet_label[COLUMN_ID] == id]
            checkBegin = True
            for i in range(nrBegin):
                if int(currLabel.loc[currLabel.first_valid_index()+i, 'label']) == 0:
                    checkBegin = False
            if checkBegin:
                begin += 1
            checkEnd = True
            for i in range(nrEnd):
                if int(currLabel.loc[currLabel.last_valid_index()-i, 'label']) == 0:
                    checkEnd = False
            if checkEnd:
                end += 1
        return begin, end
    else:
        return begin, end

FN_BeginEnd = getNrOfBeginEnding(FN[1])
print('FN Begin with 1: ' + str(FN_BeginEnd[0]))
print('FN End with 1: ' + str(FN_BeginEnd[1]))
FP_BeginEnd = getNrOfBeginEnding(FP[1])
print('FP Begin with 1: ' + str(FP_BeginEnd[0]))
print('FP End with 1: ' + str(FP_BeginEnd[1]))
TP_BeginEnd = getNrOfBeginEnding(TP[1])
print('TP Begin with 1: ' + str(TP_BeginEnd[0]))
print('TP End with 1: ' + str(TP_BeginEnd[1]))
TN_BeginEnd = getNrOfBeginEnding(TN[1])
print('TN Begin with 1: ' + str(TN_BeginEnd[0]))
print('TN End with 1: ' + str(TN_BeginEnd[1]))