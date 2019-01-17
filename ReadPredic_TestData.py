import pickle
from keras.models import load_model
from Libraries import data_evaluation as d_eval
from Libraries import data_preprocessing as pp

runName = 'l6s_td'
runNr = '135538'
path = '/media/computations/DATA/ExperimentalData/Runs/' + runNr + '/'
model = load_model(path+runName+'model.h5')
Data = pickle.load(open(path+runName+'_TestData.p', 'rb'))

X_test = Data[0]
y_test = Data[1]

## if test Data pandas
dropChannels = ['time', 'stopId']
X_test = pp.shape_Data_to_LSTM_format(Data[0], dropChannels, scale=True)
y_test = pp.shape_Labels_to_LSTM_format(Data[1])

FP, FN, TP, TN = d_eval.countTD_MaxLabel([(X_test, y_test)], model)
MCC = d_eval.get_MCC(FP, FN, TP, TN )
print('&'+str(MCC)[0:4]+'&'+str(TP)+'&'+str(TN)+'&'+str(FP)+'&'+str(FN)+'\\'+'\\')