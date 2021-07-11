import copy
import functools
from logging import error
from mmap import ACCESS_COPY
from tkinter.constants import E
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
from sklearn import metrics 
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

EPOCHS = 50
BATCH_SIZE = 30
MAX_ITER = 10
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





def process_excel(file_path: str)->Dict[str, Segments]:

    ws = xl.load_workbook(filename = file_path).active
    rec_infos : Dict[str, Segments] = {}

    line = 0

    for r in list(ws.rows)[1:]:                 # stuff is 0 indexed despite the documentation claiming its 1 indexed 
        rec_name = r[1].value
        start = r[ord('O') - ord('A')].value 
        end = r[ord('P') - ord('A')].value 
        if end is None and start is  None: continue
        end *= 1000
        start *= 1000
        line += 1
        
        add_or_append(rec_infos,rec_name , (start, end))
    return rec_infos


def load_recording(audio_file_path: str, segments:  Segments):
    song = pydub.AudioSegment.from_wav(audio_file_path)
    for [start, end] in  segments:
        #TODO  feature engineering stuff
        seg = song[start:end]
        #seg = seg.set_sample_width(2).set_channels(1)   
        #seg.export("data/" + info.rec_name + i.__str__() + ".wav", ".wav")
       
        #librosa.l
        play(seg)

def check_existence(s: str):
    if not os.path.exists(s): 
        b = os.path.exists(s)
        print(f"file not found: {b}")
        error(f"file not found: {b}")
    else: return





def add_or_append(dict: Dict[str, Segments], key:str, value:Segment):
    if key in dict:
        dict[key].append(value)
    else:
        dict[key] = [value]


def save_info(data):
    import pickle
    pickle.dump(data, open(dir + "/data_info.p", "wb" ) )

def load_stored_info(dir:str)-> Dict[str,Segments]:
    import pickle
    return pickle.load(open( dir + "/data_info.p", "rb" ) )

def load_data_pieces(dir:str)-> Tuple[List[object],List[bool]]:     # bool = target
    import pydub
    from pathlib import Path
    from scipy.io import wavfile
    from pathos.multiprocessing import ProcessingPool as Pool


    
    
    Path(dir + "/data_pieces/negative").glob('*.npy')
    neg = list(map(lambda p: (np.load(p),CORRECT),Path(dir + "/data_pieces/positive").glob('*.npy')))
    pos = list(map(lambda p: (np.load(p),INCORRECT),Path(dir + "/data_pieces/negative").glob('*.npy')))
        
    all=pos+neg
    random.shuffle(all)

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
    with parallel_backend('dask', n_jobs=10):
        model.fit(X_train,y_train)
    predicted = model.predict(X_test)
    print()
    print(f"accuracy: {sklearn.metrics.accuracy_score(y_test,predicted)}")

def SVM(xs,ys):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(xs,ys,test_size=0.2)
    n_estimators = 10
    # model = sklearn.svm.LinearSVC(verbose=1)
    model = sklearn.multiclass.OneVsRestClassifier(sklearn.ensemble.BaggingClassifier(sklearn.svm.LinearSVC(verbose=1), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
    # model = sklearn.multiclass.OneVsRestClassifier(sklearn.ensemble.BaggingClassifier(sklearn.svm.SVC(kernel='rbf', probability=True, class_weight='auto'), max_samples=1.0 / n_estimators, n_estimators=n_estimators))

    from joblib import parallel_backend
    with parallel_backend('dask', n_jobs=10):
        model.fit(X_train,y_train)
    predicted = model.predict(X_test)
    print(sklearn.metrics.accuracy_score(y_test,predicted))

def Gradient_forest(xs,ys):
    import sklearn.ensemble
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(xs,ys,test_size=0.2)
    
    #model = sklearn.ensemble.GradientBoostingClassifier(verbose=1)
    model = sklearn.ensemble.RandomForestClassifier(n_estimators=1000,verbose=1)
    from joblib import parallel_backend
    with parallel_backend('dask', n_jobs=10):
        model.fit(X_train,y_train)
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
        Conv2D(96,(5,1),activation=act, kernel_regularizer=reg,padding='same'),
        Conv2D(96,(1,5),activation=act, kernel_regularizer=reg,padding='same'),
        MaxPooling2D((4,2)),
        Conv2D(96,(3,1),strides=1,activation=act, kernel_regularizer=reg,padding='same'),
        Conv2D(96,(1,3),strides=1,activation=act, kernel_regularizer=reg,padding='same'),
        MaxPooling2D((2,2)),
        Conv2D(96,(3,1),strides=1,activation=act, kernel_regularizer=reg,padding='same'),
        Conv2D(96,(1,3),strides=1,activation=act, kernel_regularizer=reg,padding='same'),
        MaxPooling2D((2,2)),
        Conv2D(96,(3,1),strides=1,activation=act, kernel_regularizer=reg,padding='same'),
        Conv2D(96,(1,3),strides=1,activation=act, kernel_regularizer=reg,padding='same'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(16, activation=act, kernel_regularizer=reg),
        Dense(NUM_LABELS,activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy',tf.keras.metrics.AUC()], optimizer='adam')#
    in_shape = (None,xs.shape[1],xs.shape[2],1)
    model.build(input_shape=in_shape)
    model.summary()

    history = model.fit(xs,ys, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, verbose=1,use_multiprocessing=True)
    #eeee = [l.get_weights() for l in model.layers[0]]

    return history

if __name__ == "__main__":
    print()
    dir = choose_directory_dialog()

    
    xs,ys = load_data_pieces(dir)
    samples, a, b,c = xs.shape
    x1 = xs.reshape((samples,a*b*c))
    y1 = np.array(list(map(lambda y: np.argmax(y)+1,ys))).astype('float16')
    print('got data')
    c = dask.distributed.Client(processes=False)
    Gradient_forest(x1,y1)
    SVM(x1,y1)    
    InteractiveConsole(locals=globals()).interact()
    # Logistic(x1,y1)
    
    history = CNN2D(xs, ys)
   



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