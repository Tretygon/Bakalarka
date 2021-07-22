import numpy as np
import typing
from typing import List,Tuple,Dict
from dataclasses import dataclass

import numpy as np

from os.path import dirname, join as pjoin


import sklearn
import sklearn.ensemble


from keras.layers.convolutional import Conv1D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,GRU,MaxPooling1D,ConvLSTM2D
import tensorflow as tf
from tensorflow.keras import regularizers
import keras.metrics



EPOCHS = 20
BATCH_SIZE = 96
MAX_ITER = 15
NUM_LABELS = 2
VALIDATION = 0
Segment = Tuple[float,float]
Segments = List[Segment]
CLASSES = 2
CORRECT = np.eye(CLASSES)[1].astype('float16') 
INCORRECT = np.eye(CLASSES)[0].astype('float16')




def MLP(xs, ys,xs_test,ys_test):
    
    xs = xs.reshape(xs.shape[0],xs.shape[1]*xs.shape[2])
    xs_test = xs_test.reshape(xs_test.shape[0],xs_test.shape[1]*xs_test.shape[2])
    act = 'selu'
    reg = regularizers.l2(l=1e-4)
    model = Sequential([
        Dense(1024, input_dim=int(xs.shape[1]), activation=act, kernel_regularizer=reg),
        Dropout(0.2),
        Dense(1024, activation=act, kernel_regularizer=reg),
        Dropout(0.2),
        Dense(1024, activation=act, kernel_regularizer=reg),
        Dropout(0.2),
        Dense(1024, activation=act, kernel_regularizer=reg),
        Dropout(0.2),
        Dense(1024, activation=act, kernel_regularizer=reg),
        Dropout(0.2),
        Dense(1024, activation=act, kernel_regularizer=reg),
        Dropout(0.2),
        Dense(1024, activation=act, kernel_regularizer=reg),
        Dropout(0.2),
        Dense(1024, activation=act, kernel_regularizer=reg),
        Dropout(0.2),
        Dense(1024, activation=act, kernel_regularizer=reg),
        Dropout(0.2),
        Dense(1024, activation=act, kernel_regularizer=reg),
        Dropout(0.2),
        Dense(NUM_LABELS,activation='sigmoid'),
    ])

    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy',keras.metrics.Precision(),keras.metrics.Recall()], optimizer=tf.optimizers.Adam(clipnorm=2.0)) 
    
    model.summary()
    
    history = model.fit(xs,ys, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION, verbose=1,workers=8,use_multiprocessing=True)
    model.evaluate(xs_test,ys_test)
    return model,history



# def Sklearn_model(xs, ys, xs_test,ys_test, model):
#     X_train, X_eval, y_train, Y_eval = sklearn.model_selection.train_test_split(xs,ys,test_size=VALIDATION)

#     from joblib import parallel_backend
#     with parallel_backend('threading', n_jobs=10):
#         model.fit(X_train,y_train)
        
#     print("eval accuracy:")
#     predicted = model.predict(X_eval)
#     print(sklearn.metrics.accuracy_score(Y_eval,predicted))

#     print("test accuracy:")
#     predicted = model.predict(xs_test)
#     print(sklearn.metrics.accuracy_score(ys_test,predicted))


def CNN(xs, ys,xs_test,ys_test):

    xs = xs.reshape((xs.shape[0],xs.shape[1],xs.shape[2],1))
    xs_test = xs_test.reshape((xs_test.shape[0],xs_test.shape[1],xs_test.shape[2],1))
    reg = regularizers.l2(l=1e-4)
    act = 'selu'
    model = Sequential([
        Conv2D(128,(15,15),strides=(3,3),activation=act, kernel_regularizer=reg,padding='same'),
        MaxPooling2D((2,2)),
        Conv2D(128,(3,3),activation=act, kernel_regularizer=reg,padding='same'),  
        MaxPooling2D((2,2)),
        Conv2D(128,(3,3),activation=act, kernel_regularizer=reg,padding='same'), 
        MaxPooling2D((2,2)),
        Conv2D(128,(3,3),activation=act, kernel_regularizer=reg,padding='same'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(16, activation=act, kernel_regularizer=reg),
        Dense(NUM_LABELS,activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy',keras.metrics.Precision(),keras.metrics.Recall()], optimizer='adam')#
    in_shape = (None,xs.shape[1],xs.shape[2],1)
    model.build(input_shape=in_shape)
    model.summary()

    history = model.fit(xs,ys, batch_size=BATCH_SIZE, epochs=int(EPOCHS//1.5), validation_split=VALIDATION, verbose=1,workers=8,use_multiprocessing=True)


    model.evaluate(xs_test,ys_test)
    return model,history

def CNN1D(xs, ys,xs_test,ys_test):

    reg = regularizers.l2(l=1e-4)           
    act = 'selu'
    init = tf.keras.initializers.LecunNormal()

    model = Sequential([
        Conv1D(256,3,activation=act, kernel_regularizer=reg,padding='same',kernel_initializer=tf.keras.initializers.LecunNormal()),
        MaxPooling1D(3),
        Conv1D(256,3,activation=act, kernel_regularizer=reg,padding='same',kernel_initializer=tf.keras.initializers.LecunNormal()),
        MaxPooling1D(2),
        Conv1D(256,3,activation=act, kernel_regularizer=reg,padding='same',kernel_initializer=tf.keras.initializers.LecunNormal()),
        MaxPooling1D(2),
        Conv1D(256,3,activation=act, kernel_regularizer=reg,padding='same',kernel_initializer=tf.keras.initializers.LecunNormal()),
        MaxPooling1D(2),
        Conv1D(256,3,activation=act, kernel_regularizer=reg,padding='same',kernel_initializer=tf.keras.initializers.LecunNormal()),
        MaxPooling1D(2),
        Conv1D(256,3,activation=act, kernel_regularizer=reg,padding='same',kernel_initializer=tf.keras.initializers.LecunNormal()),
        MaxPooling1D(2),
        Conv1D(256,3,activation=act, kernel_regularizer=reg,padding='same',kernel_initializer=tf.keras.initializers.LecunNormal()),
        MaxPooling1D(2),
        Flatten(),
        Dense(16, activation=act, kernel_regularizer=reg),
        Dense(NUM_LABELS,activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy',keras.metrics.Precision(),keras.metrics.Recall()])
    in_shape = (None,xs.shape[1],xs.shape[2])
    model.build(input_shape=in_shape)
    model.summary()

    history = model.fit(xs,ys, batch_size=BATCH_SIZE, epochs=int(EPOCHS*1.5), validation_split=VALIDATION, verbose=1,workers=8,use_multiprocessing=True)


    model.evaluate(xs_test,ys_test)

    return model,history

def RNN(xs, ys,xs_test,ys_test):
    reg = regularizers.l2(l=1e-4)
    act = 'selu'
    model = Sequential([
        GRU(256,return_sequences=1),
        GRU(128,return_sequences=1),
        GRU(128),
        Flatten(),
        Dense(16, activation=act, kernel_regularizer=reg),
        Dense(NUM_LABELS,activation='sigmoid')
    ])#

    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy',keras.metrics.Precision(),keras.metrics.Recall()], optimizer='adam')#
    in_shape = (None,xs.shape[1],xs.shape[2])
    model.build(input_shape=in_shape)
    model.summary()

    history = model.fit(xs,ys, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION, verbose=1,workers=8,use_multiprocessing=True)
    model.evaluate(xs_test,ys_test)

    return model,history
###############################################################################################
def Train_models(xs, ys,xs_test,ys_test):
    
    MLP(xs, ys,xs_test,ys_test)
    CNN(xs, ys,xs_test,ys_test)
    # CNN1D(xs, ys,xs_test,ys_test)
    RNN(xs, ys,xs_test,ys_test)
    out = CRNN(xs, ys,xs_test,ys_test)
    return out
###############################################################################################

def CRNN(xs, ys,xs_test,ys_test):
    reg = regularizers.l2(l=1e-4)
    act = 'selu'
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
        Dense(16, activation=act, kernel_regularizer=reg),
        Dense(NUM_LABELS,activation='sigmoid')
    ])#

    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy',keras.metrics.Precision(),keras.metrics.Recall()], optimizer='adam')#
    in_shape = (None,xs.shape[1],xs.shape[2])
    model.build(input_shape=in_shape)
    model.summary()

    history = model.fit(xs,ys, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION, verbose=1,workers=8,use_multiprocessing=True)
    model.evaluate(xs_test,ys_test)

    
    return model,history


