import pickle
from Libraries import data_preprocessing as pp

loadPath = '/media/computations/DATA/ExperimentalData/DataFiles/last8s_pad/'
dataSetName = 'last8s_pad'
SavePath = '/media/computations/DATA/ExperimentalData/DataFiles/last8s_pad/'

Data = pickle.load(open(loadPath + dataSetName+'.p', 'rb'))
X_ts = Data[0]
labels_td = Data[1]

#dropChannels = ['time', 'stopId', 'trg1', 'n1', 'trot1', 'tlin1', 'tlin2', 'tamb1']
dropChannels = ['time', 'stopId']

COLUMN_ID = 'stopId'
X, y = pp.balanceSlicedData(Data[0], Data[1], target=50, distributed_Output=True, COLUMN_ID=COLUMN_ID)
TrainData, TestData = pp.splitDataPandasFormat(X, y, split=0.3, COLUMN_ID=COLUMN_ID)
X_train = pp.shape_Data_to_LSTM_format(TrainData[0], dropChannels, scale=True)
y_train = pp.shape_Labels_to_LSTM_format(TrainData[1])
X_test = pp.shape_Data_to_LSTM_format(TestData[0], dropChannels, scale=True)
y_test = pp.shape_Labels_to_LSTM_format(TestData[1])
pickle.dump((X_train, y_train), open(SavePath + dataSetName + '_TrainDataNumpy.p', 'wb'))
pickle.dump((X_test, y_test), open(SavePath + dataSetName + '_TestDataNumpy.p', 'wb'))
pickle.dump(TrainData, open(SavePath + dataSetName + '_TrainDataPandas.p', 'wb'))
pickle.dump(TestData, open(SavePath + dataSetName + '_TestDataPandas.p', 'wb'))