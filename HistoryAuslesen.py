import pickle
import keras

import matplotlib as mpl
#mpl.use("pgf")
mpl.use('TkAgg')

Run = 127964
Name = 'Diff10Sec_kReg'
PATH = '/media/computations/DATA/ExperimentalData/Runs/' + str(Run) + '/' + Name + '_history.p'

import matplotlib.pyplot as plt
import numpy

history = pickle.load( open( PATH, "rb"))
history = history[0]
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

#from matplotlib2tikz import save as tikz_save

#tikz_save(Run+".tex")