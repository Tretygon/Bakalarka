import copy
import functools
from mmap import ACCESS_COPY
from tkinter.constants import E
import numpy as np
import sklearn
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
import matplotlib.pyplot as plt

from scipy.io import wavfile as wav
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.optimizers import Adam
# from keras.utils import to_categorical

MAX_ITER = 10
SEGMENT_LEN = 2000

Segment = Tuple[float,float]
Segments = List[Segment]


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
#
#
#
#
#
#
def parse_excel(path:str)-> Dict[str,Segments]:
    infos, audios = load_data_recursively(path)

    data_info = {}

    for name,segs in infos.items(): 
        path = next(filter(lambda a: name in a, audios))                # next(...) == [0]
        check_existence(path)
        data_info[path] = segs

    return data_info

def choose_directory_dialog()-> str : 
    root = tkinter.Tk()
    root.withdraw()
    selected = tkinter.filedialog.askdirectory(parent=root, title='Choose directory')
    return selected


def load_data_recursively(root:str) -> Tuple[Dict[str, Segments],List[str]] : 
    from pathlib import Path


    for p in Path(root).glob('*.xlsx'):
        s = p.absolute().__str__()
        if ".~" in s or "~$" in s: continue              #temporary excel file thingy
        rec_info = process_file(s)
        break

    all_audio_files : List[str] = []
    for p in Path(root).rglob('*.wav'):
        s = p.absolute().__str__()
        all_audio_files.append(s) 
 
    return rec_info, all_audio_files


def process_file(file_path: str)->Dict[str, Segments]:

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
    else: return

# train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
#         dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)

def train_logistic(data,t):
    pipe = sklearn.pipeline.Pipeline(
        [("scaling", sklearn.preprocessing.StandardScaler())] +
        [("classifier", sklearn.linear_model.LogisticRegression(solver="saga", max_iter=MAX_ITER))]
    )
    pipe.fit(data, t)
    return pipe
 



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



def store_data_pieces(data: Dict[str, Segments], destination_path: str):
    data_ord = 1
    import pydub
    for path,segments in data.items(): 
        song = pydub.AudioSegment.from_wav(path)
        song_len = len(song)
        for [start, end] in  segments:
            start,end = align_to_set_len(SEGMENT_LEN, song_len, start, end)
            seg = song[start:end].set_sample_width(2).set_channels(1) 

            raw = np.array(seg.get_array_of_samples())#.astype(np.float32)
            if len(raw) != 88200:
                print()
            mn = seg.frame_width



            pathlib.Path(destination_path).mkdir(parents=True, exist_ok=True)
            seg.export(destination_path + data_ord.__str__() + ".wav", format="wav")
            data_ord += 1
            # play(seg)

def load_data_pieces(dir:str)-> Tuple[List[object],List[bool]]:     # bool = target
    import pydub
    from pathlib import Path

    xs = []
    ys = []
    for p in Path(dir + "/data_pieces/positive").glob('*.wav'):
        x = np.array(pydub.AudioSegment.from_wav(p).get_array_of_samples())
        if len(x) != 88200:
            print()
        xs.append(x)
        ys.append(1)

    for p in Path(dir + "/data_pieces/negative").glob('*.wav'):
        x = np.array(pydub.AudioSegment.from_wav(p).get_array_of_samples())
        if len(x) != 88200:
            print()
        xs.append(x)
        ys.append(0)

    mx = max([len(arr) for arr in xs])
    mn = min([len(arr) for arr in xs])
    for arr in xs:
        if len(arr) != 88200:
            print()
    return xs,ys


# enlarge or shrink each segment to make its lenght the desired constant
def align_to_set_len(const, song_len, start, end):
    adjust = (const  - end + start)/4      
    start = start - 3 * adjust
    end = end + adjust

    if end >= song_len:
        start = song_len - const
        end = song_len

    if start < 0:
        start = 0
        end = const

    dif = end - start
    if end - start != const:
        print()
    return start,end


def as_dict(lst : List[Tuple[str, Segments]])-> Dict[str, Segments]:
    return { path:segs for path,segs in lst}
    

def Make_negative_data(n: int,  info: Dict[str, Segments]):           #, files: List[str]
    files = list(info.keys())               # TODO: files without positive samples?
    per_file_samples =[n // len(files) for f in files]
    for _ in range(n % len(files)): 
        i = random.randrange(0, len(files))
        per_file_samples[i] = per_file_samples[i] + 1
    data = {f:Single_file_negative_data(f,info[f],samples) for f,samples in zip(files,per_file_samples)}
    return data 

          
    


def Single_file_negative_data(file: str,segs: Segments, n:int)-> Segments:
    file_len = len(pydub.AudioSegment.from_wav(file))

    def rec(rec_length: int, n: int, acc: List[Segment])-> List[Segment]:
        k = n
        if n == 0: return acc
        for i in range(k):
            start = random.randrange(0,rec_length-SEGMENT_LEN)
            end = start + SEGMENT_LEN
            for seg in segs:
                (a,b) = seg
                if (a < start and start < b ) or (a < end and end < b ):
                    return rec(rec_length,n,acc)
            acc.append((start,end))
            n -= 1
        return acc

    return rec(file_len, n, [])



if __name__ == "__main__":
    dir = choose_directory_dialog()

    #pos = load_stored_info(dir)
    # pos = parse_excel(dir)
    #save_info(pos)
    #c = [min([ k[1] - k[0] for k in b[1]]) for b in a]
    # num_of_data = sum([len(segs) for _,segs in pos.items()])
    # neg = Make_negative_data(num_of_data,pos)

    # store_data_pieces(neg, dir + "/data_pieces/negative/")
    #store_data_pieces(pos, dir + "/data_pieces/positive/")

    xs,ys = load_data_pieces(dir)
    # xs = np.array([x.get_array_of_samples() for x in xs])
    
    InteractiveConsole(locals=globals()).interact()

