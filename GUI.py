import copy
import functools
from logging import error
from mmap import ACCESS_COPY
from tkinter.constants import DISABLED, E
from keras.backend import conv1d, dropout, elu
from keras.layers.convolutional import Conv1D
from matplotlib.colors import cnames
import numpy as np
import typing
from typing import Callable, List,Tuple,Dict
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

import Machine_learning as ML
import process_data as Processing
import keras

def with_dir(f:Callable[[str],None]):
    dir = Processing.choose_directory_dialog()
    if dir:
        f(dir)
def with_dir_delayed(f:Callable[[str],None]):
    return lambda: with_dir(f)

# https://keras.io/guides/writing_your_own_callbacks/
class ModelCallbacks(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))

def app():
    import tkinter as tk
    from tkinter import ttk


    root = tk.Tk()
    root.title('Terminator nursery')
    btn_extract = tk.Button(root, text="Extract training data", command=with_dir_delayed(Processing.load_data))   #TODO: saving dialog?
    btn_extract.grid(row=0,column=0, sticky='nesw') 
    #popup with the number of extracted positive data

    btn_load = tk.Button(root, text="Load training data",command=with_dir_delayed(functools.comProcessing.load_data))
    btn_load.grid(row=0,column=1, sticky='nesw')

    training_data_label = tk.Label(root,bg='brown1', text = "              Training data               ")
    training_data_label.grid(row=0,column=3, sticky='nesw')
    

    btn_train_model = tk.Button(root, text="Train model", command=with_dir_delayed(Processing.load_data))
    btn_train_model.grid(row=1,column=0,columnspan=2,sticky='nesw')
    #popup with elapsed time      ?? and accuracy ??
    # name the model?
    #error if no data available

    model_label = tk.Label(root,bg='lime green', text = "AI model")
    model_label.grid(row=1,column=3, sticky='nesw')
    
    btn_load_model = tk.Button(root, text="Load model", command=with_dir_delayed(Processing.load_data))
    btn_load_model.grid(row=2,column=0, sticky='nesw')


    btn_save_model = tk.Button(root, text="Save model", command=with_dir_delayed(Processing.load_data))
    btn_save_model.grid(row=2,column=1, sticky='nesw')
    #error if no model available

    btn_apply_model = tk.Button(root, text="Apply model on data", command=with_dir_delayed(Processing.load_data))
    btn_apply_model.grid(row=3,column=0,columnspan=2, sticky='nesw')
    #error if no model available

    # stop button?
    # open console button?


    progress_bar = ttk.Progressbar(root, orient = tk.HORIZONTAL, length = 100, mode = 'determinate')
    progress_bar.grid(row=2,column=3,columnspan=1, sticky='nesw')
    progress_bar['value'] = 30

    buttons = [btn_extract,btn_load,btn_train_model,btn_load_model,btn_save_model,btn_apply_model]
    def disable_buttons():
        for button in buttons:
            button['state'] = tk.DISABLED
            
    def enable_buttons():
        for button in buttons:
            button['state'] = tk.NORMAL
            
    def update_progress(val:int):
        progress_bar['value'] = val
        root.update_idletasks()


    
    def start_hover(e):e.widget['bg']='LightBlue1'
    def end_hover(e):e.widget['bg']='white smoke'
    for button in buttons:
        button.bind("<Enter>", start_hover)
        button.bind("<Leave>", end_hover)
    

    root.mainloop()


if __name__ == "__main__":
    app()