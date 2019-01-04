import pickle
from keras.models import load_model
from Libraries import configuration
from Libraries import log_setup as logSetup

from Libraries import data_import as dataImport
from Libraries import data_preprocessing as pp
from Libraries import prepare_csv as csv
from Libraries import data_evaluation as d_eval

projectName = 'TestAufschwingen'
callDataset = '1051'
stopId = '1399.1051'

Data = pickle.load(open('SmallWindowData.p', 'rb'))
X_ts = Data[0]
labels_td = Data[1]

Model_name = 'Diff10Sec_trained2model.h5'
model = load_model(Model_name)
COLUMN_ID = 'sliceId'

dropChannels = ['time', 'stopId', 'trg1', 'n1', 'trot1', 'tlin1', 'tlin2', 'tamb1']


v = d_eval.get_classified_dataSets(X_ts, labels_td, model, dropChannels=dropChannels, COLUMN_ID=COLUMN_ID)

v = 1