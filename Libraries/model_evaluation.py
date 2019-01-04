"""Contains Tools for Evaluating the Training Procedue and Helper Functions for Debugging and Documentation purpose"""
import matplotlib.pyplot as plt
from keras.models import Sequential
import math
import numpy as np
import pickle
from matplotlib2tikz import save as tikz_save

def calculateConfusionMatrix(labels, predictions):
    confusion = tf.confusion_matrix(labels, predictions)
    confusionmatrix = confusion.eval(session=tf.Session())
    return confusionmatrix

def calculateMCC(confusionMatrix):
    tp = confusionMatrix[0,0]
    tn = confusionMatrix[1,1]
    fp = confusionMatrix[0,1]
    fn = confusionMatrix[1,0]
    mcc = (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return mcc

def plot_loss_history(histories, epochs, name, path):
    vaL_loss = np.zeros((epochs, 1))
    train_loss = np.zeros((epochs, 1))
    for history in histories:
        vaL_loss = vaL_loss + np.array(history.history['val_loss'])
        train_loss = train_loss + np.array(history.history['loss'])
    plt.plot(train_loss)
    plt.plot(vaL_loss)
    plt.title('model_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    tikz_save(path + name+'.tex')

def eval_all(histories, epochs, name, model, path, test_data):
    pickle.dump(histories, open(name + '_history.p', 'wb'))
    model.save(name + 'model.h5')
    pickle.dump(test_data, open(name + '_TestData.p', 'wb'))