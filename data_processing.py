from Machine_learning import CORRECT, INCORRECT
import copy
import functools
from logging import error
from mmap import ACCESS_COPY
import numpy as np
import typing
from typing import List,Tuple,Dict
import random
from code import InteractiveConsole
import openpyxl as xl
from dataclasses import dataclass
import pydub
from pydub.playback import play
import os

import IPython.display as ipd
import numpy as np
# import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

# import scipy as sp
from pathlib import Path

from sklearn.model_selection import train_test_split

from pathlib import Path


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


def parse_excel(path:str,rec_col,start_col,end_col)-> Dict[str,Segments]:
    infos, audios = load_data_recursively(path,rec_col,start_col,end_col)

    data_info = {}
    sentinel = object() #https://stackoverflow.com/questions/3114252/one-liner-to-check-whether-an-iterator-yields-at-least-one-element
    for name,segs in infos.items(): 
        path = next(filter(lambda a: name in a, audios),sentinel)                # next(items) == items[0]
        if path == sentinel or not os.path.exists(path): 
            print(f"file not found: {name}")
            error(f"file not found: {name}")
        else:
            len_song = len(pydub.AudioSegment.from_wav(path))
            new_segments = [align_to_set_len(SEGMENT_LEN, len_song, start, end) for start,end in segs] #length normalization
            data_info[path] = new_segments

    return data_info


def load_data_recursively(root:str,rec_col,start_col,end_col) -> Tuple[Dict[str, Segments],List[str]] : 

    rec_info = {}
    for p in Path(root).rglob('*.xlsx'):
        s = p.absolute().__str__()
        if ".~" in s or "~$" in s: continue              #temporary excel file thingy
        process_excel(s,rec_info,rec_col,start_col,end_col)

    all_audio_files : List[str] = []
    for p in Path(root).rglob('*.wav'):
        s = p.absolute().__str__()
        all_audio_files.append(s) 
 
    return rec_info, all_audio_files


def process_excel(file_path: str,rec_info:Dict[str, Segments],rec_col,start_col,end_col):

    ws = xl.load_workbook(filename = file_path).active

    for r in list(ws.rows)[1:]:                 # stuff is 0 indexed despite the documentation claiming its 1 indexed 
        rec_name = r[ord(rec_col) - ord('A')].value
        start = r[ord(start_col) - ord('A')].value 
        end = r[ord(end_col) - ord('A')].value 
        
        if end is None or start is None or not isinstance(start,float) or not isinstance(end,float): continue

        end *= 1000
        start *= 1000
        
        add_or_append(rec_info,rec_name, (float(start), float(end)))


def load_recording(audio_file_path: str, segments:  Segments):
    song = pydub.AudioSegment.from_wav(audio_file_path)
    for [start, end] in  segments:
        seg = song[start:end]
        play(seg)





def add_or_append(dict: Dict[str, Segments], key:str, value:Segment):
    if key in dict:
        if not (value in dict[key]):
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

def no_aug_data_pieces(data: Dict[str, Segments], y):
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
    return ret_xs,ret_ys

# generator that chops a recording into SEGMENT_LEN sized pieces and transforms them to MFCC
# 75% overlap
def partition_recording(file):
    from itertools import chain
    import pydub
    song = pydub.AudioSegment.from_wav(file)
    ln = len(song)
    max = ((ln//SEGMENT_LEN)*(SEGMENT_LEN)) 
    for start in chain(range(0,max,SEGMENT_LEN//4), [ln-SEGMENT_LEN]):
        end = start+SEGMENT_LEN
        part = song[start:end] 
        sr = part.frame_rate
        mel = To_Mel(part ,sr)

        yield mel,start,end,file

def augment_data_pieces(data1: Dict[str, Segment],data2: Dict[str, Segment],y):
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
            raw2 = np.array(seg2.get_array_of_samples())*random.random()
            sr = song1.frame_rate
            mel = To_Mel(np.add(raw1,raw2),sr)

            ret_xs.append(mel)
            ret_ys.append(y)

            segments2_i += 1
            if segments2_i == len(segments2):
                segments2_i = 0
                items2_i += 1
                if items2_i == len(items2): return ret_xs, ret_ys
                (path2,segments2) = items2[items2_i]
                random.shuffle(segments2)
                song2 = pydub.AudioSegment.from_wav(path2)
    return ret_xs, ret_ys

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
    
    S = librosa.feature.melspectrogram(data.astype('float32'), sr=sr, n_fft=1028, hop_length=256, n_mels=128)
    
    log_mfcc = librosa.feature.mfcc(S=np.log(S+1e-6), sr=sr, n_mfcc=48)
    
    return log_mfcc.astype('float16').T


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

# def concat_small_files():
#     all_pos = list(map(np.load,Path(dir + "/data_pieces/augmentation/positive").glob('*.npy')))
#     all_neg = list(map(np.load,Path(dir + "/data_pieces/augmentation/negative").glob('*.npy')))
#     all_data = np.array(all_pos+all_neg)
#     targets = np.append(np.tile(CORRECT,[len(all_pos),1]),np.tile(INCORRECT,[len(all_neg),1]),axis=0)
    
   
    
#     np.save(dir +"/data_pieces/augmentation/all_data",all_data)
#     np.save(dir +"/data_pieces/augmentation/all_targets",targets)


#     all_pos = list(map(np.load,Path(dir + "/data_pieces/test/positive").glob('*.npy')))
#     all_neg = list(map(np.load,Path(dir + "/data_pieces/test/negative").glob('*.npy')))
#     all_data = np.array(all_pos+all_neg)
#     targets = np.append(np.tile(CORRECT,[len(all_pos),1]),np.tile(INCORRECT,[len(all_neg),1]),axis=0)

    
#     np.save(dir +"/data_pieces/all_tests",all_data)
#     np.save(dir +"/data_pieces/all_test_targets",targets)
    
def extract_training_data(report_progress,dir,rec_col,start_col,end_col,augmentations):
    from pathos.multiprocessing import Pool
    pool =  Pool(2+2*augmentations) 

    positive = parse_excel(dir,rec_col,start_col,end_col)
    report_progress(25)
    all_positive = positive
    flatten=lambda dict: [(file,s) for file,segs in dict.items() for s in segs]
    as_dict = lambda x: add_or_append_multiple({},x)

    pos_flat = flatten(positive)
    positive,positive_test = train_test_split(pos_flat,test_size=TEST_SIZE)
    num_of_all_data = len(pos_flat)
    num_of_test_data = len(positive_test)
    num_of_train_data = len(positive)
    positive,positive_test = as_dict(positive),as_dict(positive_test)
    
    k = 1 + 3*augmentations  # number of 'buckets of negatives)
    negative = Make_negative_data(num_of_all_data + num_of_train_data*(k-1),all_positive)  #take the dict before separating test data to not lose any files to make negative data
    negative = flatten(negative)
    random.shuffle(negative)
    negative,negative_test = train_test_split(negative,test_size=num_of_test_data)
    report_progress(50)
    #split negatives into k buckets, because of augmentation, and 
    #convert back to dict to group together all the segments from the same file
    #saves a loooot of unnecessary loading later
    negative = [as_dict(negative[i::k]) for i in range(k)]
    negative_test = as_dict(negative_test)
    work = [
        [no_aug_data_pieces,[positive,CORRECT]],
        [no_aug_data_pieces,[negative[k-1],INCORRECT]],

        *[[augment_data_pieces,[positive,negative[i],CORRECT]] for i in range(augmentations)],

        *[[augment_data_pieces,[negative[i],negative[i+1],INCORRECT]] for i in range(augmentations,augmentations*3,2)]
    ]
    data = list(pool.map(lambda work: work[0](*(work[1])),work))

    xs = []
    ys = []
    for d in data:
        for x in d[0]:
            xs.append(x)
        for y in d[1]:
            ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)


    report_progress(75)
    
    indices = np.arange(xs.shape[0])
    np.random.shuffle(indices)
    xs = xs[indices]
    ys = ys[indices]



    data = list(pool.map(lambda work: work[0](*(work[1])),
        [
            [no_aug_data_pieces,[positive_test,CORRECT]],
            [no_aug_data_pieces,[negative_test,INCORRECT]]
        ]))

    pool.close()
    pool.terminate()
    xs_test = []
    ys_test = []
    for d in data:
        for x in d[0]:
            xs_test.append(x)
        for y in d[1]:
            ys_test.append(y)
    xs_test = np.array(xs_test)
    ys_test = np.array(ys_test)

    report_progress(100)

    return xs, xs_test, ys, ys_test
# pathlib.Path(destination_path).mkdir(parents=True, exist_ok=True)

def merge_training_data(files):
    all = [np.load(f) for f in files]
    xs = np.concatenate([a[0] for a in all], axis=0)
    xs_test = np.concatenate([a[1] for a in all], axis=0)
    ys = np.concatenate([a[2] for a in all], axis=0)
    ys_test = np.concatenate([a[3] for a in all], axis=0)
    return xs,xs_test,ys,ys_test
    



if __name__ == "__main__":
    extract_training_data('B','O','P',lambda _: None)



