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
import data_processing as Processing
import keras

from tkinter import *

from tkinter import messagebox


def with_dir(f:Callable[[str],None]):
    dir = Processing.choose_directory_dialog()
    if dir:
        f(dir)
def delayed(f:Callable[...,None], params):
    return lambda: f(*params)

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
    root.resizable(False, False)
    
    xs, xs_test, ys, ys_test = None, None, None, None 
    model = None

    training_data_label = tk.Label(root,bg='brown1', text = "              Training data               ")
    training_data_label.grid(row=0,column=3, sticky='nesw',columnspan=2)
    model_label = tk.Label(root,bg='brown1', text = "AI model")
    model_label.grid(row=1,column=3, sticky='nesw',columnspan=2)

    

    def extract_data():
        nonlocal model,xs, xs_test, ys, ys_test
        dial = tkinter.Tk()
        dial.withdraw()
        selected = tkinter.filedialog.askdirectory(parent=dial,title='Choose a directory')
        if selected is None or len(selected) == 0: return
        xs, xs_test, ys, ys_test = None, None, None, None       #free memory
        xs, xs_test, ys, ys_test = validate_and_use_inputs(lambda a,b,c,d: Processing.extract_training_data(update_progress,selected,a,b,c,d))
        training_data_label['bg'] = 'lime green'
        save_data()


    def load_data():
        nonlocal model,xs, xs_test, ys, ys_test
        dial = tkinter.Tk()
        dial.withdraw()
        selected = tkinter.filedialog.askopenfilename(parent=dial, title="Select training data", defaultextension='.npz',filetypes=[['npz files','*.npz']])
        if selected is None or len(selected) == 0: return
        loaded = np.load(selected)
        xs1, xs_test1, ys1, ys_test1 = loaded['arr_0.npy'],loaded['arr_1.npy'],loaded['arr_2.npy'],loaded['arr_3.npy']
        if xs1 is not None:
            xs, xs_test, ys, ys_test = xs1, xs_test1, ys1, ys_test1
            training_data_label['bg'] = 'lime green'


    def merge_data():
        nonlocal model,xs, xs_test, ys, ys_test
        dial = tkinter.Tk()
        dial.withdraw()
        selected = tkinter.filedialog.askopenfilenames(parent=dial, title="Select data to merge", defaultextension='.npz',filetypes=[['npz files','*.npz']])
        if selected is None or len(selected) == 0: return
        xs1, xs_test1, ys1, ys_test1  = Processing.merge_training_data(selected)
        if xs1 is not None:
            xs, xs_test, ys, ys_test = xs1, xs_test1, ys1, ys_test1
            training_data_label['bg'] = 'lime green'
            save_data()
            
    def save_data():
        nonlocal model, xs, xs_test, ys, ys_test
        if xs is None: return
        dial = tkinter.Tk()
        dial.withdraw()
        selected = tkinter.filedialog.asksaveasfilename(parent=dial, title="Save the training data", defaultextension='.npz',filetypes=[['npz files','*.npz']])
        if selected is None or len(selected) == 0: return
        np.savez(selected,xs, xs_test, ys, ys_test)



    def train_model():
        nonlocal model,xs, xs_test, ys, ys_test
        if xs is None or ys is None or ys_test is None or xs_test is None:
            messagebox.showinfo(f"Error", f"Training data not loaded")
            return

        model1,history = ML.Train_models(xs, ys, xs_test, ys_test)
        if model1 is not None:
            model = model1
            model_label['bg'] = 'lime green'
        print()

    def save_model():
        nonlocal model
        if model:
            dial = tkinter.Tk()
            dial.withdraw()
            selected = tkinter.filedialog.asksaveasfilename(parent=dial, title="Save model as")
            if selected:
                model.save(selected)
        else:
            messagebox.showinfo(f"Error", f"No model is loaded")
            return

    def load_model():
        nonlocal model
        dial = tkinter.Tk()
        dial.withdraw()
        selected = tkinter.filedialog.askopenfilename(parent=dial, title="Load model")
        if selected:
            model1 = keras.models.load_model(selected)
            if model1 != None:
                model = model1
                model_label['bg'] = 'lime green'

        print()

    def apply_model():
        nonlocal model
        import csv

        if model is None:
            messagebox.showinfo(f"Error", f"No model loaded") 
            return
        dial = tkinter.Tk()
        dial.withdraw()
        selected = tkinter.filedialog.askopenfilenames(parent=dial, title="Choose recordings",defaultextension='wav')
        if selected is None  or len(selected) == 0: return 

        for file in selected:
            
            with open(f'{file}_results.csv', mode='w') as results_file:
                fieldnames = ['file', 'start', 'end', 'activity']
                writer = csv.writer(results_file,fieldnames=fieldnames)#, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL
                pieces = Processing.partition_recording(file)
                rows = []
                for piece, start, end, file in pieces:
                    y = model.call(piece)#.predict(use_multiprocessing=True)
                    label = np.argmax(y)#[0]
                    rows.append([file,start,end,label])
                writer.writerows(rows)
    
    def partition_recording(file):
        mel,start,end,file = Processing.partition_recording()

    btn_extract = tk.Button(root, text="Extract data", command=extract_data)   #TODO: saving dialog?
    btn_extract.grid(row=0,column=0, sticky='nesw') 
    #popup with the number of extracted positive data

    btn_load = tk.Button(root, text="Load data",command=load_data)
    btn_load.grid(row=0,column=1, sticky='nesw')

    

    btn_save_data = tk.Button(root, text="Save data", command=save_data)
    btn_save_data.grid(row=1,column=0,sticky='nesw')

    btn_merge_data = tk.Button(root, text="Merge data", command=merge_data)
    btn_merge_data.grid(row=1,column=1,sticky='nesw')
    #popup with elapsed time      ?? and accuracy ??
    # name the model?
    #error if no data available

    
    btn_load_model = tk.Button(root, text="Load model", command=load_model)
    btn_load_model.grid(row=2,column=0, sticky='nesw')


    btn_save_model = tk.Button(root, text="Save model", command=save_model)
    btn_save_model.grid(row=2,column=1, sticky='nesw')
    #error if no model available

    btn_train_model = tk.Button(root, text="Train model", command=train_model)
    btn_train_model.grid(row=3,column=0,sticky='nesw')

    btn_apply_model = tk.Button(root, text="Apply model on data", command=apply_model)
    btn_apply_model.grid(row=3,column=1, sticky='nesw')
    #error if no model available

    # stop button?
    # open console button?



    

    tk.Label(root, text="Augmentations").grid(row=4)
    aug_col = tk.StringVar(root)
    e0 = tk.Entry(root,textvariable=aug_col)
    e0.insert(0,"3")
    e0.grid(row=4, column=1)

    tk.Label(root, text="Recording name column").grid(row=5)
    rec_col = tk.StringVar(root)
    e1 = tk.Entry(root,textvariable=rec_col)
    e1.insert(0,"B")
    e1.grid(row=5, column=1)
    
    tk.Label(root, text="Song start column").grid(row=6,column=0)
    start_col = tk.StringVar(root)
    e2 = tk.Entry(root,textvariable=start_col)
    e2.insert(0,"O")
    e2.grid(row=6, column=1)
    
    tk.Label(root, text="Song end column").grid(row=7,column=0)
    end_col = tk.StringVar(root)
    e3 = tk.Entry(root,textvariable=end_col)
    e3.insert(0,"P")
    e3.grid(row=7, column=1)
    
    # take data from the input fields in the UI, validate them and apply to a continuation
    def validate_and_use_inputs(f):
        vals = []
        for col in [rec_col,start_col,end_col]:
            val = col.get().strip()
            if  len(val) != 1 or ord(val[0]) < ord('A') or ord(val[0]) > ord('Z'):
                messagebox.showinfo(f"Error", f"Invalid text field input: {col.get()}") 
                return 
            vals.append(val[0])

        val = aug_col.get()
        if len(val) == 0 or not val.isnumeric():
            messagebox.showinfo(f"Error", f"Invalid text field input: {col.get()}") 
            return 
        vals.append(int(val))
        return f(*vals)
    
    progress_bar = ttk.Progressbar(root, orient = tk.HORIZONTAL, length = 100, mode = 'determinate')
    progress_bar.grid(row=2,column=3,columnspan=1, sticky='nesw')
    
    def update_progress(val:int):
        progress_bar['value'] = val
        root.update_idletasks()

    buttons = [btn_extract,btn_load,btn_train_model,btn_load_model,btn_save_model,btn_apply_model]
    def disable_buttons():
        for button in buttons:
            button['state'] = tk.DISABLED
            
    def enable_buttons():
        for button in buttons:
            button['state'] = tk.NORMAL
            


    
    def start_hover(e):e.widget['bg']='LightBlue1'
    def end_hover(e):e.widget['bg']='white smoke'
    for button in buttons:
        button.bind("<Enter>", start_hover)
        button.bind("<Leave>", end_hover)
    

    root.mainloop()


if __name__ == "__main__":
    app()