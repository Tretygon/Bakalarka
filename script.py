import copy
import functools
from tkinter.constants import E
import numpy as np
import sklearn
import typing
from typing import List,Tuple
import random
import pickle
import openpyxl
from code import InteractiveConsole
import openpyxl as xl
import tkinter
import tkinter.filedialog
from dataclasses import dataclass
import pydub
from pydub.playback import play


MAX_ITER = 10


@dataclass
class recording_info:
    rec_name:str
    seqs: List[Tuple[float,float]]


def load_files_dialog()-> 'tuple[List[recording_info],List[str]]' : 
    root = tkinter.Tk()
    root.withdraw()
    selected = tkinter.filedialog.askdirectory(parent=root, title='Choose directory')
    
    # files = root.tk.splitlist(selected)
    # for file in files: process_file(file)
    return load_data_recursively(selected)

def load_data_recursively(root:str) -> 'tuple[List[recording_info],List[str]]' : 
    from pathlib import Path


    all_recording_infos : List[recording_info] = []
    for p in Path(root).rglob('*.xlsx'):
        s = p.absolute().__str__()
        if "~$" in s: continue
        i = process_file(s)
        all_recording_infos += i

    all_audio_files : List[str] = []
    for p in Path(root).rglob('*.wav'):
        s = p.absolute().__str__()
        all_audio_files.append(s) 
 
    return all_recording_infos, all_audio_files

def process_file(file_path: str)->List[recording_info]:
    suffix = "DOPO" if "DOPO" in file_path else "RANO" if "RANO" in file_path else None
    if suffix is None : raise Exception("invalid file name:  "+file_path)

    ws = xl.load_workbook(filename = file_path).active
    info : recording_info = recording_info("", [])
    all_recording_infos : List[recording_info] = []

    def store():
        if info.rec_name != "" and len(info.seqs) != 0:
            all_recording_infos.append(info)

    for r in list(ws.rows)[1:]:                 # stuff is 0 indexed despite the documentation claiming its 1 indexed ?!!?
        if r[13].value == None: continue 
        if r[1].value:
            store()
            info = recording_info(r[1].value+suffix+".wav" ,[])
        info.seqs.append((r[13].value * 1000, r[14].value * 1000))
    store()
    return all_recording_infos


def load_recording(info:  recording_info, audio_file_path: str):
    song = pydub.AudioSegment.from_wav(info.rec_name)
    for [start, end] in  info.seqs:
        #TODO  feature engineering stuff
        seg = song[start:end].set_frame_rate(10_000).set_sample_width(8)
        play(seg)


# train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
#         dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)

def train_logistic(data,t):
    pipe = sklearn.pipeline.Pipeline(
        [("scaling", sklearn.preprocessing.StandardScaler())] +
        [("classifier", sklearn.linear_model.LogisticRegression(solver="saga", max_iter=MAX_ITER))]
    )
    pipe.fit(data, t)
    return pipe
 


def classify(model, data):
    return model.predict(data)







if __name__ == "__main__":
    infos, audios = load_files_dialog()

    for i in infos: 
        path = next(filter(lambda a: i.rec_name in a, audios))                # .next() == [0]
        load_recording(i,path)
    InteractiveConsole(locals=globals()).interact()


