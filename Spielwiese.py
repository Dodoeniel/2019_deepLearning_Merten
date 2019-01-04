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


Data = pickle.load(open('smallDiff.p', 'rb'))
X_ts = Data[0][0]
labels_td = Data[0][1]

Model_name = 'Diff10Sec_trainedmodel.h5'
model = load_model(Model_name)
COLUMN_ID = 'stopId'

dropChannels = ['time', 'stopId', 'trg1', 'n1', 'trot1', 'tlin1', 'tlin2', 'tamb1']

FP, FN, TP, TN = d_eval.get_classified_dataSets(X_ts, labels_td, model, dropChannels=dropChannels, COLUMN_ID=COLUMN_ID)

eec = csv_read.eec_csv_to_eecData(config.eecPath, callDataset)

def getLengthOfClassified(ClassifiedDataSet, eec):
    if not ClassifiedDataSet.empty:
        timeList = list()
        for id in ClassifiedDataSet[COLUMN_ID].unique():
            timeList.append(eec.loc[eec[COLUMN_ID] == id, 'stoptime'])
        return timeList
    else:
        timeList = list()
        return timeList


ofile = open('LengthClassifiedDatasets.csv', 'w')
writer = csv.writer(ofile, delimiter=",")


header = ['FP', 'FN', 'TP', 'TN']
writer.writerow([header])
FN_time = getLengthOfClassified(FN[0], eec)
TN_time = getLengthOfClassified(TN[0], eec)
TP_time = getLengthOfClassified(TP[0], eec)
FP_time = getLengthOfClassified(FP[0], eec)

for i in range(max(len(FN_time), len(TN_time), len(TP_time), len(FP_time))):
    FN_one = ''
    TN_one = ''
    TP_one = ''
    FP_one = ''
    if i < len(FN_time):
        FN_one = FN_time[i]
    if i < len(TN_time):
        TN_one = TN_time[i]
    if i < len(TP_time):
        TP_one = TP_time[i]
    if i < len(FP_time):
        FP_one = FP_time[i]
    writer.writerow([FP_time, FN_time, TP_time, TN_time])

ofile.close()

v = 1