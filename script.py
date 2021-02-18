import copy
import functools
import numpy as np
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

@dataclass
class recording_info:
    rec_name:str
    seqs: List[Tuple[float,float]]

all_recording_infos : List[recording_info] = []
all_recording_files: List[str] = []


def load_excels(): 
    root = tkinter.Tk()
    selected = tkinter.filedialog.askopenfilenames(parent=root, title='Choose files')
    files = root.tk.splitlist(selected)
    for file in files: process_file(file)

def process_file(file_name: str):
    suffix = "DOPO" if "DOPO" in file_name else "RANO" if "RANO" in file_name else None
    if suffix is None : raise Exception("invalid file name:  "+file_name)

    wb = xl.load_workbook(filename = file_name)
    info : recording_info = recording_info("", [])

    def store():
        if info.rec_name != "" and len(info.seqs) != 0:
            all_recording_infos.append(info)

    for r in wb.rows[2:]: 
        if r[14].value == None: continue 
        if r[2].value:
            store()
            info = recording_info(r[2].value+suffix+".wav" ,[])
        info.seqs.append((r[14].value * 1000, r[15].value * 1000))
    store()


def load_recording(info:  recording_info):
    song = pydub.AudioSegment.from_wav(info.file_name)
    for [start, end] in  info.seqs:
        #TODO  feature engineering
        seg = song[start:end].set_sample_width(8).set_channels(1).set_frame_rate(10_000)
        play(seg)




if __name__ == "__main__":
    InteractiveConsole(locals=globals()).interact()


