from Machine_learning import CORRECT, INCORRECT
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
from pathlib import Path

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

SEGMENT_LEN = 3000  #in miliseconds
Segment = Tuple[float,float]
Segments = List[Segment]
TEST_SIZE=0.1

CORRECT = np.eye(2)[1].astype('float16') 
INCORRECT = np.eye(2)[0].astype('float16')
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
        len_song = len(pydub.AudioSegment.from_wav(path))
        new_segments = [align_to_set_len(SEGMENT_LEN, len_song, start, end) for start,end in segs] #length normalization
        data_info[path] = new_segments

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
        rec_info = process_excel(s)
        break

    all_audio_files : List[str] = []
    for p in Path(root).rglob('*.wav'):
        s = p.absolute().__str__()
        all_audio_files.append(s) 
 
    return rec_info, all_audio_files


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
def add_or_append_multiple(dict: Dict[str, Segments], key_values: List[Tuple[str, Segment]]):
    for key,value in key_values:
        if key in dict:
            dict[key].append(value)
        else:
            dict[key] = [value]
    return dict


def save_info(data):
    import pickle
    pickle.dump(data, open(dir + "/data_info.p", "wb" ) )

def load_stored_info(dir:str)-> Dict[str,Segments]:
    import pickle
    return pickle.load(open( dir + "/data_info.p", "rb" ) )

    
# enlarge or shrink each segment to make its lenght the desired constant  
def align_to_set_len(const, song_len, start, end):
    adjust = (const  - end + start)
    distribution  = random.random()      
    start = int(start - distribution*adjust)
    end = int(end + (1-distribution)*adjust)
    
    dif = const - (end - start)
    end = end + dif     #prevent rounding errors

    if end >= song_len:
        start = song_len - const
        end = song_len

    if start < 0:
        start = 0
        end = const

    return start,end

def store_data_pieces(data: Dict[str, Segments], destination_path: str,xs_dest_path: str,ys_dest_path: str,y):
    ret_xs,ret_ys = [],[]
    import pydub
    for path,segments in data.items(): 
        song = pydub.AudioSegment.from_wav(path)
        for [start, end] in  segments:
            seg = song[start:end]
            raw = np.array(seg.get_array_of_samples())
            sr = song.frame_rate
            mel = To_Mel(raw,sr)
            ret_xs.append(mel)
            ret_ys.append(y)
            # pathlib.Path(destination_path).mkdir(parents=True, exist_ok=True)
            # np.save(destination_path + data_ord.__str__(),mel)
            # data_ord += 1
    np.save(xs_dest_path,np.array(ret_xs))
    np.save(ys_dest_path,np.array(ret_ys))

def augment_and_store(data: Tuple[Dict[str, Segment],Dict[str, Segment]],xs_dest_path: str,ys_dest_path: str,y):
    data1,data2 = data
    ret_xs,ret_ys = [],[]
    import pydub
    items2_i = 0
    segments2_i = 0
    items2 = list(data2.items())
    random.shuffle(items2)
    (path2,segments2) = items2[items2_i]
    random.shuffle(segments2)
    song2 = pydub.AudioSegment.from_wav(path2)
    for (path1,segments1) in data1.items(): 
        song1 = pydub.AudioSegment.from_wav(path1)
        for [start1, end1] in  segments1:
            [start2, end2] = segments2[segments2_i]


            seg1 = song1[start1:end1]
            seg2 = song2[start2:end2]
            raw1 = np.array(seg1.get_array_of_samples())
            raw2 = np.array(seg2.get_array_of_samples())*0.5
            sr = song1.frame_rate
            mel = To_Mel(np.add(raw1,raw2),sr)

            ret_xs.append(mel)
            ret_ys.append(y)
            # pathlib.Path(destination_path).mkdir(parents=True, exist_ok=True)
            # np.save(destination_path + data_ord.__str__(),mel)

            data_ord += 1
            segments2_i += 1
            if segments2_i == len(segments2):
                segments2_i = 0
                items2_i += 1
                if items2_i == len(items2): return
                (path2,segments2) = items2[items2_i]
                random.shuffle(segments2)
                song2 = pydub.AudioSegment.from_wav(path2)
    np.save(xs_dest_path,np.array(ret_xs))
    np.save(ys_dest_path,np.array(ret_ys))

def as_dict(lst : List[Tuple[str, Segments]])-> Dict[str, Segments]:
    return { path:segs for path,segs in lst}
    

def Make_negative_data(n: int,  info: Dict[str, Segments])-> Dict[str, Segments]:           #, files: List[str]
    files = list(info.keys())               # TODO:?include files without positive samples?

    # spread evenly across files
    per_file_samples =[n // len(files) for f in files]
    for _ in range(n % len(files)): 
        i = random.randrange(0, len(files))
        per_file_samples[i] = per_file_samples[i] + 1

    data = {f:Single_file_choose_negative_data(f,info[f],samples) for f,samples in zip(files,per_file_samples)}
    return data 

        
def To_Mel(data,sr):  
    
    S = librosa.feature.melspectrogram(data.astype('float16'), sr=sr, n_fft=1028, hop_length=256, n_mels=128)
    
    log_mfcc = librosa.feature.mfcc(S=np.log(S+1e-6), sr=sr, n_mfcc=48)
    
    return log_mfcc.astype('float16')


def Single_file_choose_negative_data(file: str,segs: Segments, n:int)-> Segments:
    file_len = len(pydub.AudioSegment.from_wav(file))

    ret = []
    while True:
        if n == 0: return ret
        start = random.randrange(0,file_len-SEGMENT_LEN)
        end = start + SEGMENT_LEN
        for seg in segs:
            (a,b) = seg
            if (a < start and start < b ) or (a < end and end < b ):   # start or end is inside the segment <=> overlap
                break
        else:
            ret.append((start,end))
            n -= 1

def concat_small_files():
    all_pos = list(map(np.load,Path(dir + "/data_pieces/augmentation/positive").glob('*.npy')))
    all_neg = list(map(np.load,Path(dir + "/data_pieces/augmentation/negative").glob('*.npy')))
    all_data = np.array(all_pos+all_neg)
    targets = np.append(np.tile(CORRECT,[len(all_pos),1]),np.tile(INCORRECT,[len(all_neg),1]),axis=0)
    
    indices = np.arange(all_data.shape[0])
    np.random.shuffle(indices)
    all_data = all_data[indices]
    targets = targets[indices]
    
    np.save(dir +"/data_pieces/augmentation/all_data",all_data)
    np.save(dir +"/data_pieces/augmentation/all_targets",targets)


    all_pos = list(map(np.load,Path(dir + "/data_pieces/test/positive").glob('*.npy')))
    all_neg = list(map(np.load,Path(dir + "/data_pieces/test/negative").glob('*.npy')))
    all_data = np.array(all_pos+all_neg)
    targets = np.append(np.tile(CORRECT,[len(all_pos),1]),np.tile(INCORRECT,[len(all_neg),1]),axis=0)

    
    np.save(dir +"/data_pieces/all_tests",all_data)
    np.save(dir +"/data_pieces/all_test_targets",targets)
    
if __name__ == "__main__":
    print()
    dir = choose_directory_dialog()
    # concat_small_files()
    positive = load_stored_info(dir)
    positive = parse_excel(dir)
    all_positive = positive
    save_info(positive)
    flatten=lambda dict: [(file,s) for file,segs in dict.items() for s in segs]
    as_dict = lambda x: add_or_append_multiple({},x)

    pos_flat = flatten(positive)
    positive,positive_test = train_test_split(pos_flat,test_size=TEST_SIZE)
    num_of_all_data = len(pos_flat)
    num_of_test_data = len(positive_test)
    num_of_train_data = len(positive)
    positive,positive_test = as_dict(positive),as_dict(positive_test)
    
    
    k = 4  
    negative = Make_negative_data(num_of_all_data + num_of_train_data*(k-1),all_positive)  #take the dict before separating test data to not lose any files to make negative data
    negative = flatten(negative)
    random.shuffle(negative)
    negative,negative_test = train_test_split(negative,test_size=num_of_test_data)
    
    #split negatives into k buckets, because of augemtation, and 
    #convert back to dict to group together all the segments from the same file
    #saves a loooot of unnecessary loading later
    negative = [as_dict(negative[i::k]) for i in range(k)]
    negative_test = as_dict(negative_test)
    

    from pathos.multiprocessing import ProcessPool
    work = [
        [augment_and_store,(positive,negative[2]), dir + "/data_pieces/augmented/positive/aug", dir + "/data_pieces/augmented/positive/aug_t",CORRECT],
        [augment_and_store,(negative[0],negative[1]), dir + "/data_pieces/augmented/negative/aug", dir + "/data_pieces/augmented/negative/aug_t",INCORRECT],

        [store_data_pieces,positive, dir + "/data_pieces/augmented/positive/norm", dir + "/data_pieces/augmented/positive/norm_t",CORRECT],
        [store_data_pieces,negative[3], dir + "/data_pieces/augmented/negative/norm", dir + "/data_pieces/augmented/negative/norm_t",INCORRECT],

        [store_data_pieces,positive_test, dir + "/data_pieces/test/positive/pos", dir + "/data_pieces/augmented/positive/pos_t",CORRECT]
        [store_data_pieces,negative_test, dir + "/data_pieces/test/negative/neg", dir + "/data_pieces/augmented/negative/neg_t",INCORRECT]
    ]
    # ProcessPool(nodes=4).map(lambda args:args[0](args[1],args[2],args[3]),work)



