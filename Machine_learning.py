import copy
import functools
from logging import error
from mmap import ACCESS_COPY
from tkinter.constants import E
from keras.backend import dropout
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

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D,Conv1D,MaxPooling1D
from keras.optimizers import Adam,SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import keras
import tensorflow as tf
from tensorflow.keras import regularizers



import noisereduce as nr
#tf.python.client.device_lib.list_local_devices()

#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
# sess = tf.Session(config=config) 
# keras.backend.set_session(sess)

EPOCHS = 15
BATCH_SIZE = 30
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


# TODO: 
# bootstrapping negative data instead of uniform selection?
# randomise adjusting to set length
# data augumentation
#
#
#
#


def choose_directory_dialog()-> str : 
    root = tkinter.Tk()
    root.withdraw()
    selected = tkinter.filedialog.askdirectory(parent=root, title='Choose directory')
    return selected



def load_data(dir):
    xs = np.load(dir + "/data_pieces/augmentation/all_data.npy")
    ys = np.load(dir + "/data_pieces/augmentation/all_targets.npy")


    return xs.reshape([xs.shape[0],xs.shape[1],xs.shape[2],1]), ys

def load_data_pieces(dir:str)-> Tuple[List[object],List[bool]]:     # bool = target
    from pathlib import Path
    Path(dir + "/data_pieces/negative").glob('*.npy')
    neg = list(map(lambda p: (np.load(p),CORRECT),Path(dir + "/data_pieces/augmentation/positive").glob('*.npy')))
    pos = list(map(lambda p: (np.load(p),INCORRECT),Path(dir + "/data_pieces/augmentation/negative").glob('*.npy')))
        
    all=pos+neg
    random.shuffle(all)

    xs = [a[0].reshape((a[0].shape[0],a[0].shape[1],1)) for a in all]
    ys = [a[1] for a in all]
    return np.array(xs),np.array(ys)

def load_test_data_pieces(dir:str)-> Tuple[List[object],List[bool]]:     # bool = target
    from pathlib import Path
    Path(dir + "/data_pieces/negative").glob('*.npy')
    neg = list(map(lambda p: (np.load(p),CORRECT),Path(dir + "/data_pieces/test/positive").glob('*.npy')))
    pos = list(map(lambda p: (np.load(p),INCORRECT),Path(dir + "/data_pieces/test/negative").glob('*.npy')))
    all=pos+neg
    xs = [a[0].reshape((a[0].shape[0],a[0].shape[1],1)) for a in all]
    ys = [a[1] for a in all]
    return np.array(xs),np.array(ys)



# def MLP(xs, ys):
#     model = Sequential()

#     model.add(Dense(
#         128,
#         input_dim=int(xs.shape[1]),
#         activation='relu'
#         ))
#     model.add(Dense(
#         64,
#         activation='relu'
#         ))
#     #model.add(Dropout(0.2))
#     model.add(Dense(NUM_LABELS,activation='sigmoid'))

#     model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer='adam') 
    
#     # model.summary()

#     def label_loss(y_true,y_pred):
#          return tf.keras.losses.binary_crossentropy(y_true, y_pred) * y_true

#     return model.fit(xs,ys, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, verbose=1,use_multiprocessing=True)



def Logistic(xs,ys):
    # def train_logistic(data,t):
    # pipe = sklearn.pipeline.Pipeline(
    #     [("scaling", sklearn.preprocessing.StandardScaler())] +
    #     [("classifier", sklearn.linear_model.LogisticRegression(solver="saga", max_iter=MAX_ITER))]
    # )
    # pipe.fit(data, t)
    # return pipe
 

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(xs,ys,test_size=0.2)
    model = sklearn.linear_model.LogisticRegressionCV(verbose=1,max_iter=200)
    from joblib import parallel_backend
    with parallel_backend('threading', n_jobs=10):
        model.fit(X_train,y_train)
    predicted = model.predict(X_test)
    print()
    print(f"accuracy: {sklearn.metrics.accuracy_score(y_test,predicted)}")

def SVM(xs,ys):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(xs,ys,test_size=0.2)
    n_estimators = 3
    model = sklearn.ensemble.BaggingClassifier(sklearn.svm.SVC(kernel='rbf'), max_samples=1.0 / n_estimators, n_estimators=n_estimators)
   
    #model = sklearn.svm.LinearSVC(verbose=1)
# sklearn.ensemble.BaggingClassifier(, max_samples=1.0 / n_estimators, n_estimators=n_estimators,n_jobs=3)
    # from joblib import parallel_backend 
    #with parallel_backend('threading', n_jobs=10):
    model.fit(X_train,y_train)
    predicted = model.predict(X_test)
    print(sklearn.metrics.accuracy_score(y_test,predicted))

def Gradient_forest(xs,ys):
    import sklearn.ensemble
    xs, X_test, ys, y_test = sklearn.model_selection.train_test_split(xs,ys,test_size=0.2)
    n_estimators = 3
    model = sklearn.ensemble.BaggingClassifier(sklearn.ensemble.GradientBoostingClassifier(verbose=1,min_samples_split=10), max_samples=0.5 / n_estimators, n_estimators=n_estimators,n_jobs=n_estimators)
    # model = sklearn.ensemble.RandomForestClassifier(n_estimators=1000,verbose=1,min_samples_split=10)
    from joblib import parallel_backend
    with parallel_backend('threading', n_jobs=10):
        model.fit(xs,ys)
    predicted = model.predict(X_test)
    print(sklearn.metrics.accuracy_score(y_test,predicted))


# def CNN1D(xs, ys):
#     reg = None  #regularizers.l2(l=1e-4)
#     act = LeakyReLU(alpha=0.1)
#     model = Sequential([
#         Conv1D(64,32,activation=act, kernel_regularizer=reg),
#         MaxPooling1D(4),
#         Conv1D(32,9,strides=1,activation=act, kernel_regularizer=reg),
#         MaxPooling1D(4),
#         Conv1D(32,3,activation=act, kernel_regularizer=reg),
#         MaxPooling1D(2),
#         Flatten(),
#         Dense(16, activation=act, kernel_regularizer=reg),
#         Dense(NUM_LABELS,activation='sigmoid')
#     ])

#     model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer='adam')#,tf.keras.metrics.AUC()
#     in_shape = (None,xs.shape[1],xs.shape[2])
#     model.build(input_shape=in_shape)
#     model.summary()

#     history = model.fit(xs,ys, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, verbose=1,use_multiprocessing=True)
#     #eeee = [l.get_weights() for l in model.layers[0]]

#     return history
def CNN2D(xs, ys):
    


    reg = regularizers.l2(l=1e-4)
    act = LeakyReLU(alpha=0.1)
    model = Sequential([
        Conv2D(96,(15,15),strides=(3,3),activation=act, kernel_regularizer=reg,padding='same'),
        MaxPooling2D((2,2)),
        Conv2D(96,(3,3),activation=act, kernel_regularizer=reg,padding='same'),
        MaxPooling2D((2,2)),
        Conv2D(96,(3,3),activation=act, kernel_regularizer=reg,padding='same'),
        MaxPooling2D((2,2)),
        Conv2D(96,(3,3),activation=act, kernel_regularizer=reg,padding='same'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(32, activation=act, kernel_regularizer=reg),
        Dropout(0.2),
        Dense(16, activation=act, kernel_regularizer=reg),
        Dropout(0.2),
        Dense(NUM_LABELS,activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy',tf.keras.metrics.AUC()], optimizer='adam')#
    in_shape = (None,xs.shape[1],xs.shape[2],1)
    model.build(input_shape=in_shape)
    model.summary()

    history = model.fit(xs,ys, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, verbose=1,workers=4,use_multiprocessing=True)

    test_xs,test_ys = load_test_data_pieces(dir)
    model.evaluate(test_xs,test_ys)
    #eeee = [l.get_weights() for l in model.layers[0]]

    return history

if __name__ == "__main__":
    print()
    dir = choose_directory_dialog()

    
    xs,ys = load_data(dir)
    # history = CNN2D(xs, ys)




    samples, a, b,c = xs.shape
    xs = xs.reshape([samples,a*b*c])
    ys = np.array(list(map(lambda y: np.argmax(y)+1,ys))).astype('float16')
    # c = dask.distributed.Client()
    Gradient_forest(xs,ys)
    # SVM(x1,y1)   
    Logistic(xs,ys)
    # SVM(x1,y1)    
    
    
    InteractiveConsole(locals=globals()).interact()
    
   



    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show(block=False)


    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show(block=False)


# NOTES
# targets need to be one-hot encoded to work with accuracies in keras: otherwise use sparse accuracy
# 
# 
# 
# 
# 
# 
# 