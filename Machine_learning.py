import copy
import functools
from logging import error
from mmap import ACCESS_COPY
from tkinter.constants import E
from keras.backend import conv1d, dropout, elu
from keras.layers.convolutional import Conv1D
from matplotlib.colors import cnames
import numpy as np
import typing
from typing import List,Tuple,Dict
import random
import pickle
import openpyxl
import pathlib
from code import InteractiveConsole
import openpyxl as xl
import tkinter
import tkinter.filedialog
from dataclasses import dataclass
import pydub
from pydub.playback import play
import os

import IPython.display as ipd
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

from os.path import dirname, join as pjoin
from scipy.io import wavfile
import scipy.io
import scipy as sp
import matplotlib.pylab as pylab
import image
import joblib
import dask.distributed


import sklearn
import sklearn.ensemble

from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.preprocessing import LabelEncoder
import sklearn.multiclass
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import BayesianGaussianMixture


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,LSTM,GRU,SimpleRNN,MaxPooling1D,Bidirectional
from keras.optimizers import Adam,SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU,PReLU,ELU
import keras
import tensorflow as tf
from tensorflow.keras import regularizers


import noisereduce as nr
#tf.python.client.device_lib.list_local_devices()

#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
# sess = tf.Session(config=config) 
# keras.backend.set_session(sess)

EPOCHS = 20
BATCH_SIZE = 100
MAX_ITER = 15
NUM_LABELS = 2
Segment = Tuple[float,float]
Segments = List[Segment]
CLASSES = 2
CORRECT = np.eye(CLASSES)[1].astype('float16') 
INCORRECT = np.eye(CLASSES)[0].astype('float16')
# @dataclass
# class recording_info:
#     rec_name:str
#     seqs: List[Tuple[float,float]]



def choose_directory_dialog()-> str : 
    root = tkinter.Tk()
    root.withdraw()
    selected = tkinter.filedialog.askdirectory(parent=root, title='Choose directory')
    return selected



def load_data(dir):
    xs = np.load(dir + "/xs")#"/data_pieces/augmentation/all_data.npy")
    ys = np.load(dir + "/ys")#"/data_pieces/augmentation/all_targets.npy")
    return xs, ys 

def load_tests(dir):
    xs = np.load(dir + "/xs_test")#"/data_pieces/all_tests.npy")
    ys = np.load(dir + "/ys_test")#"/data_pieces/all_test_targets.npy")
    return xs, ys        #xs.reshape([xs.shape[0],xs.shape[1],xs.shape[2],1])

def load_audios(dir:str)-> Tuple[List[object],List[bool]]:     # bool = target
    from pathlib import Path
    Path(dir + "/data_pieces/negative").glob('*.npy')
    neg = list(map(lambda p: (np.load(p),CORRECT),Path(dir + "/data_pieces/augmentation/positive").glob('*.npy')))
    pos = list(map(lambda p: (np.load(p),INCORRECT),Path(dir + "/data_pieces/augmentation/negative").glob('*.npy')))
        
    all=pos+neg
    random.shuffle(all)

    xs = [a[0].reshape((a[0].shape[0],a[0].shape[1],1)) for a in all]
    ys = [a[1] for a in all]
    return np.array(xs),np.array(ys)

def load_test_audios(dir:str)-> Tuple[List[object],List[bool]]:     # bool = target
    from pathlib import Path
    Path(dir + "/data_pieces/negative").glob('*.wav')
    neg = list(map(lambda p: (np.load(p),CORRECT),Path(dir + "/data_pieces/test/positive").glob('*.wav')))
    pos = list(map(lambda p: (np.load(p),INCORRECT),Path(dir + "/data_pieces/test/negative").glob('*.npy')))
    all=pos+neg
    xs = [a[0].reshape((a[0].shape[0],a[0].shape[1],1)) for a in all]
    ys = [a[1] for a in all]
    return np.array(xs),np.array(ys)



def MLP(xs, ys,xs_test,ys_test):
    model = Sequential()

    reg = regularizers.l2(l=1e-4)
    model.add(Dense(
        1024,
        input_dim=int(xs.shape[1]),
        activation=LeakyReLU(alpha=0.1),
        kernel_regularizer=reg
        ))
    model.add(Dense(
        256,
        activation=LeakyReLU(alpha=0.1),
        kernel_regularizer=reg
        ))
    model.add(Dense(
        256,
        activation=LeakyReLU(alpha=0.1),
        kernel_regularizer=reg
        ))
    model.add(Dense(
        256,
        activation=LeakyReLU(alpha=0.1),
        kernel_regularizer=reg
        ))
    model.add(Dense(
        256,
        activation=LeakyReLU(alpha=0.1),
        kernel_regularizer=reg
        ))
    model.add(Dense(
        256,
        activation=LeakyReLU(alpha=0.1),
        kernel_regularizer=reg
        ))
    model.add(Dense(
        128,
        activation=LeakyReLU(alpha=0.1),
        kernel_regularizer=reg
        ))
    # model.add(Dropout(0.2))
    model.add(Dense(NUM_LABELS,activation='sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy',tf.keras.metrics.AUC()], optimizer='adam') 
    
    model.summary()
    
    history = model.fit(xs,ys, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, verbose=1,use_multiprocessing=True)
    model.evaluate(xs_test,ys_test)
    return model,history

def Sklearn_model(xs, ys, xs_test,ys_test, model):
    X_train, X_eval, y_train, Y_eval = sklearn.model_selection.train_test_split(xs,ys,test_size=0.2)

    from joblib import parallel_backend
    with parallel_backend('threading', n_jobs=10):
        model.fit(X_train,y_train)
        
    print("eval accuracy:")
    predicted = model.predict(X_eval)
    print(sklearn.metrics.accuracy_score(Y_eval,predicted))

    print("test accuracy:")
    predicted = model.predict(xs_test)
    print(sklearn.metrics.accuracy_score(ys_test,predicted))


def CNN(xs, ys,xs_test,ys_test):

    xs = xs.reshape((None,xs.shape[1],xs.shape[2],1))
    xs_test = xs_test.reshape((None,xs_test.shape[1],xs_test.shape[2],1))
    reg = regularizers.l2(l=1e-4)
    act = LeakyReLU(alpha=0.1)
    model = Sequential([
        # Conv2D(96,(3,3),activation=act, kernel_regularizer=reg,padding='same'),
        Conv2D(96,(3,12),strides=(1,6),activation=act, kernel_regularizer=reg,padding='same'),
        # Conv2D(96,(5,5),activation=act, kernel_regularizer=reg,padding='same'),
        MaxPooling2D((3,2)),
        # Conv2D(96,(3,3),activation=act, kernel_regularizer=reg,padding='same'),  
        Conv2D(96,(9,9),activation=act, kernel_regularizer=reg,padding='same'),  
        MaxPooling2D((2,2)),
        Conv2D(96,(5,5),activation=act, kernel_regularizer=reg,padding='same'), 
        MaxPooling2D((2,2)),
        Conv2D(96,(3,3),activation=act, kernel_regularizer=reg,padding='same'), 
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(32, activation=act, kernel_regularizer=reg),
        # Dropout(0.2),
        Dense(16, activation=act, kernel_regularizer=reg),
        # Dropout(0.2),
        Dense(NUM_LABELS,activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy',tf.keras.metrics.AUC()], optimizer='adam')#
    in_shape = (None,xs.shape[1],xs.shape[2],1)
    model.build(input_shape=in_shape)
    model.summary()

    history = model.fit(xs,ys, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, verbose=1,workers=4,use_multiprocessing=True)


    model.evaluate(xs_test,ys_test)
    return model,history

def CNN1D(xs, ys,xs_test,ys_test):

    reg = regularizers.l2(l=1e-4)
    act = LeakyReLU(alpha=0.1)
    model = Sequential([
        Conv1D(512,3,activation=act, kernel_regularizer=reg,padding='same'),
        MaxPooling1D(2),
        Conv1D(512,3,activation=act, kernel_regularizer=reg,padding='same'),
        MaxPooling1D(2),
        Conv1D(512,3,activation=act, kernel_regularizer=reg,padding='same'),
        MaxPooling1D(2),
        Conv1D(512,3,activation=act, kernel_regularizer=reg,padding='same'),
        MaxPooling1D(2),
        Conv1D(512,3,activation=act, kernel_regularizer=reg,padding='same'),
        MaxPooling1D(3),
        Flatten(),
        Dense(32, activation=act, kernel_regularizer=reg),
        Dense(NUM_LABELS,activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy',tf.keras.metrics.AUC()])
    in_shape = (None,xs.shape[1],xs.shape[2])
    model.build(input_shape=in_shape)
    model.summary()

    history = model.fit(xs,ys, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, verbose=1,workers=4,use_multiprocessing=True)


    model.evaluate(xs_test,ys_test)
    #eeee = [l.get_weights() for l in model.layers[0]]

    return model,history

def RNN(xs, ys,xs_test,ys_test):
    reg = regularizers.l2(l=1e-4)
    act = LeakyReLU(alpha=0.1)
    model = Sequential([
        GRU(256,return_sequences=1),
        GRU(128,return_sequences=1),
        GRU(128),
        Dense(NUM_LABELS,activation='sigmoid')
    ])#

    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy',tf.keras.metrics.AUC()], optimizer='adam')#
    in_shape = (None,xs.shape[1],xs.shape[2])
    model.build(input_shape=in_shape)
    model.summary()

    history = model.fit(xs,ys, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, verbose=1,workers=4,use_multiprocessing=True)
    model.evaluate(xs_test,ys_test)

    return model,history

def Train_models(xs, ys,xs_test,ys_test):
    CRNN(xs, ys,xs_test,ys_test)
    RNN(xs, ys,xs_test,ys_test)
    CNN(xs, ys,xs_test,ys_test)
    out = CNN1D(xs, ys,xs_test,ys_test)
    return out

def CRNN(xs, ys,xs_test,ys_test):
    reg = regularizers.l2(l=1e-4)
    act = LeakyReLU(alpha=0.1)
    model = Sequential([
        GRU(128,return_sequences=1),
        Conv1D(256,3,activation=act, kernel_regularizer=reg,padding='same'),
        MaxPooling1D(2),
        GRU(128,return_sequences=1),
        MaxPooling1D(2),
        Conv1D(256,3,activation=act, kernel_regularizer=reg,padding='same'),
        MaxPooling1D(2),
        GRU(128,return_sequences=1),
        MaxPooling1D(2),
        Conv1D(256,3,activation=act, kernel_regularizer=reg,padding='same'),
        MaxPooling1D(2),
        GRU(128),
        Dense(NUM_LABELS,activation='sigmoid')
    ])#

    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy',tf.keras.metrics.AUC()], optimizer='adam')#
    in_shape = (None,xs.shape[1],xs.shape[2])
    model.build(input_shape=in_shape)
    model.summary()

    history = model.fit(xs,ys, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, verbose=1,workers=4,use_multiprocessing=True)
    model.evaluate(xs_test,ys_test)

    return model,history



if __name__ == "__main__":
    print()
    dir = choose_directory_dialog()
 
    xs,ys = load_data(dir)
    xs_test,ys_test = load_tests(dir)
    type = 'float32'
    xs,ys,xs_test, ys_test = xs.astype(type),ys.astype(type),xs_test.astype(type), ys_test.astype(type)
    # (samples, mel_features, time) to (samples, time, mel_features)
    # without it RNN caps at 75 acc, clearly not working 
    # CNN(xs, ys,xs_test,ys_test)

    # xs = np.swapaxes(xs,1,2)
    # xs_test = np.swapaxes(xs_test,1,2)

    # 4D data
    # CNN(xs, ys,xs_test,ys_test)


    #3D data
    CNN1D(xs, ys,xs_test,ys_test)
    CRNN(xs, ys,xs_test,ys_test)
    
    samples, a, b,c = xs.shape
    xs = xs.reshape([samples,a*b*c])

    samples, a, b,c = xs_test.shape
    xs_test = xs_test.reshape([samples,a*b*c])
    # 1D data
    # MLP(xs,ys,xs_test,ys_test)

    ys_test = np.array(list(map(lambda y: np.argmax(y)+1,ys_test)))
    ys = np.array(list(map(lambda y: np.argmax(y)+1,ys)))

    run_model = lambda m: Sklearn_model(xs,ys,xs_test,ys_test,m)
    
    # flat data, flat targets 

    # model = sklearn.linear_model.LogisticRegressionCV(verbose=0,max_iter=200)
    # run_model(model)

    # model = MLPClassifier([128,64,32], max_iter=15,verbose=1)
    # run_model(model)


    # model = KNeighborsClassifier()
    # run_model(model)


    # from sklearn.naive_bayes import GaussianNB,BernoulliNB
    # model = GaussianNB()
    # run_model(model)
    
    # model = BayesianGaussianMixture(n_components=1)
    # run_model(model)

    # n_estimators = 3
    # model = sklearn.ensemble.BaggingClassifier(sklearn.ensemble.GradientBoostingClassifier(verbose=1,min_samples_split=10), max_samples=0.5 / n_estimators, n_estimators=n_estimators,n_jobs=n_estimators)
    # model = sklearn.ensemble.RandomForestClassifier(n_estimators=250,verbose=1,min_samples_split=5)
    # run_model(model)
    
    # # model = sklearn.ensemble.BaggingClassifier(sklearn.svm.SVC(kernel='rbf'), max_samples=1.0 / n_estimators, n_estimators=n_estimators)
    # model = sklearn.svm.LinearSVC(verbose=1)   
    # run_model(model)
    
 
    # InteractiveConsole(locals=globals()).interact()

    # from matplotlib import pyplot as plt
    # plt.figure()
    # plt.plot(history.history['binary_accuracy'])
    # plt.plot(history.history['val_binary_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper right')
    # plt.show(block=False)


    # plt.figure()
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show(block=False)

