import numpy as np
import pickle
from keras.models import load_model
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import Input
from Libraries import data_preprocessing as pp

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.cm as cmx


name = 'WindowedData_Set1_Model_2_model.h5'
dataName = 'w11s_11hs_1051.p'
PATH = '/media/computations/DATA/ExperimentalData/Runs/116926/'
model = load_model(PATH + name)
layer = model.get_layer('lstm_2')
weights = layer.get_weights()

inputs1 = Input(shape=(None, 8))
lstm1, state_h, state_c = LSTM(8, return_sequences=True, return_state=True)(inputs1)
model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])
model.set_weights(weights)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


Data = pickle.load(open(PATH + dataName, 'rb'))
dropChannels = ['time', 'stopId', 'trg1', 'n1', 'trot1', 'tlin1', 'tlin2', 'tamb1']
X = pp.shape_Data_to_LSTM_format(Data[0], dropChannels)
Y = pp.shape_Labels_to_LSTM_format(Data[1])

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlim([-1, 9])
ax1.set_ylim([-1, 4])
cm = plt.get_cmap('viridis')
cNorm  = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

currX = X[0]
currY = Y[0]

plt.box(False)

animation_time = len(currX)
interval = 10

ax1.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)


ax1.annotate('cell output', xy=(-1, 3))
ax1.annotate('cell state', xy=(-1, 2))
ax1.annotate('hidden state', xy=(-1, 1))
ax1.annotate('', xy=(1, 0), xytext=(8, 0), arrowprops={'arrowstyle': '|-|'})
ax1.annotate('time [s]', xy=(8, 0))
ax1.annotate('0', xy=(1, -0.2))
ax1.annotate(str(animation_time/100), xy=(8, -0.2))


#ax1.annotate('', xy=(0.5, 0.5), xytext=(8.5, 0.5), arrowprops={'arrowstyle': '-'})
ax1.annotate('node', xy=(0, 0.5))
ax1.annotate('1', xy=(1, 0.5))
ax1.annotate('2', xy=(2, 0.5))
ax1.annotate('3', xy=(3, 0.5))
ax1.annotate('4', xy=(4, 0.5))
ax1.annotate('5', xy=(5, 0.5))
ax1.annotate('6', xy=(6, 0.5))
ax1.annotate('7', xy=(7, 0.5))
ax1.annotate('8', xy=(8, 0.5))


def animate(i):
    # get cell state at that time step (keras only gives the last one
    X_timestep = currX[0:i+1]
    yt, ht, ct = model.predict(X_timestep.reshape((1, X_timestep.shape[0], X_timestep.shape[1])))

    ax1.add_artist(plt.Rectangle(xy=(1, 0.1), width=10, height = 0.2, color='white', fill=True))
    ax1.annotate('', xy=(1, 0), xytext=(8, 0), arrowprops={'arrowstyle': '|-|'})

    currTimeInterval = 7/(interval*animation_time)*i
    ax1.annotate('curr', xy=(1+currTimeInterval, -0.1), xytext=(1+currTimeInterval,0.1))

    # get value from 0th data set, ith row (time step) and 0th feature
    for k in range(len(yt[0][i])):

        colorVal = scalarMap.to_rgba(yt[0][-1][k])
        circle1 = plt.Circle((1+k, 3), 0.2, color=colorVal)
        ax1.add_artist(circle1)
        colorVal = scalarMap.to_rgba(ct[0][k])
        circle1 = plt.Circle((1 + k, 2), 0.2, color=colorVal)
        ax1.add_artist(circle1)
        colorVal = scalarMap.to_rgba(ht[0][k])
        circle1 = plt.Circle((1 + k, 1), 0.2, color=colorVal)
        ax1.add_artist(circle1)


anim = animation.FuncAnimation(fig, animate, frames=np.arange(0, 100), interval=10)
anim.save(PATH+'ActivationMap.gif', dpi=80, writer='imagemagick')
#def animate(i):

