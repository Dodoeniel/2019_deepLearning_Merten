import pickle
from keras.models import load_model
from Libraries import data_evaluation as d_eval


runName = 'm1l8td'
path = '/media/computations/DATA/ExperimentalData/Runs/122957/'
model = load_model(path+runName+'model.h5')
Data = pickle.load(open(path+runName+'_TestData.p', 'rb'))

X_test = Data[0]
y_test = Data[1]

d_eval.countTD_MaxLabel([(X_test, y_test)], model)