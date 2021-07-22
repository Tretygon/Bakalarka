from Machine_learning import BATCH_SIZE
from logging import error
import numpy as np
import typing
from typing import List,Tuple,Dict
import random
import openpyxl as xl
from dataclasses import dataclass
import os
from scipy.io.wavfile import read
import numpy as np
# import pandas as pd
import librosa
import librosa.display

# import scipy as sp
from pathlib import Path

from sklearn.model_selection import train_test_split

from pathlib import Path

CORRECT = np.eye(2)[1].astype('float16') 
INCORRECT = np.eye(2)[0].astype('float16')



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

def slice_wav(data, sr,start, end):
    start_i = int((start//10)*(sr//100))
    end_i = int((end//10)*(sr//100))
    # if len()
    res = data[start_i:end_i]
    return res

def parse_excel(path:str,rec_col:int,start_col:int,end_col:int)-> Dict[str,Segments]:
    infos, audios = load_data_recursively(path,rec_col,start_col,end_col)

    data_info = {}
    sentinel = object() #https://stackoverflow.com/questions/3114252/one-liner-to-check-whether-an-iterator-yields-at-least-one-element
    for name,segs in infos.items(): 
        path = next(filter(lambda a: name in a, audios),sentinel)                # next(items) == items[0]
        if path == sentinel or not os.path.exists(path): 
            print(f"file not found: {name}")
            error(f"file not found: {name}")
        else:
            sr,rec = read(path,True)
            len_song = int((rec.shape[0]/sr)*1000)
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


def process_excel(file_path: str,rec_info:Dict[str, Segments],rec_col:str,start_col:str,end_col:str):

    ws = xl.load_workbook(filename = file_path).active

    for r in list(ws.rows)[1:]:                 # stuff is 0 indexed despite the documentation claiming its 1 indexed 
        rec_name = r[ord(rec_col) - ord('A')].value
        start = r[ord(start_col) - ord('A')].value 
        end = r[ord(end_col) - ord('A')].value 
        
        if end is None or start is None or not isinstance(start,float) or not isinstance(end,float): continue

        end *= 1000
        start *= 1000
        
        add_or_append(rec_info,rec_name, (float(start), float(end)))


def Make_negative_data(n: int,  info: Dict[str, Segments])-> Dict[str, Segments]:           
    files = list(info.keys())               
    # spread evenly across files
    per_file_samples =[n // len(files) for f in files]
    for _ in range(n % len(files)): 
        i = random.randrange(0, len(files))
        per_file_samples[i] = per_file_samples[i] + 1

    data = {f:Single_file_choose_negative_data(f,info[f],samples) for f,samples in zip(files,per_file_samples)}
    return data 


def Single_file_choose_negative_data(file: str,segs: Segments, n:int)-> Segments:
    sr,rec = read(file,True)
    file_len = int((rec.shape[0]/sr)*1000)

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

    
# enlarge or shrink each segment to make its lenght the desired constant  
def align_to_set_len(const:int, song_len:int, start:int, end:int):
    adjust = (const  - end + start)
    distribution  = random.random()      
    start = ((start - distribution*adjust)//10)*10
    end = ((end + (1-distribution)*adjust)//10)*10
    start
    dif = const - (end - start)
    end = end + dif     #prevent rounding errors

    if end >= song_len:
        start = song_len - const - 1
        end = song_len - 1

    if start < 0:
        start = 0
        end = const

    return start,end

def no_aug_data_pieces(data: Dict[str, Segments], y:np.ndarray):
    ret_xs,ret_ys = [],[]
    for path,segments in data.items(): 
        sr,song = read(path,True)
        for [start, end] in  segments:
            slice = slice_wav(song, sr, start, end)
            mel = To_Mel(slice,sr)
            ret_xs.append(mel)
            ret_ys.append(y)
    return ret_xs,ret_ys

# generator that chops a recording into SEGMENT_LEN sized pieces and transforms them to MFCC
# 75% overlap
def partition_recording(file:str):
    from itertools import chain
    sr,rec = read(file,True)
    len_song = int((rec.shape[0]/sr)*1000)
    max = ((len_song//SEGMENT_LEN)*(SEGMENT_LEN)) 
    ret,ret_info = [],[]

    for start in chain(range(0,max,SEGMENT_LEN//4), [len_song-SEGMENT_LEN]):
        end = start+SEGMENT_LEN
        slice = slice_wav(rec,sr, start, end)
        mel = To_Mel(slice,sr)

        ret.append(mel)
        ret_info.append((start,end))
        if len(ret) == BATCH_SIZE:
            yield (np.array(ret),ret_info)
            ret, ret_info = [],[] 
    yield (ret,ret_info)


def augment_data_pieces(data1: Dict[str, Segment],data2: Dict[str, Segment],y:np.ndarray):
    ret_xs,ret_ys = [],[]
    items2_i = 0
    segments2_i = 0
    items1 = list(data1.items())
    items2 = list(data2.items())
    random.shuffle(items1)
    random.shuffle(items2)
    (path2,segments2) = items2[items2_i]
    random.shuffle(segments2)
    sr2,recording2 = read(path2)
    for (path1,segments1) in items1: 
        sr1,recording1 = read(path1)
        for [start1, end1] in  segments1:
            [start2, end2] = segments2[segments2_i]


            slice1 = slice_wav(recording1,sr1,start1,end1) 
            slice2 = slice_wav(recording2,sr2,start2,end2) * random.random()
            assert(sr1 == sr2)
            added = np.add(slice1,slice2)
            mel = To_Mel(added,sr1)

            ret_xs.append(mel)
            ret_ys.append(y)
            segments2_i += 1

            if segments2_i == len(segments2):
                segments2_i = 0
                items2_i += 1
                if items2_i == len(items2): return ret_xs, ret_ys
                (path2,segments2) = items2[items2_i]
                random.shuffle(segments2)
                sr2,recording2 = read(path2)
    return ret_xs, ret_ys

def as_dict(lst : List[Tuple[str, Segments]])-> Dict[str, Segments]:
    return { path:segs for path,segs in lst}
    


        
def To_Mel(data:np.ndarray,sr:int)->np.ndarray:  
    S = librosa.feature.melspectrogram(data.astype('float32'), sr=sr, n_fft=1028, hop_length=256, n_mels=128)
    
    log_mfcc = librosa.feature.mfcc(S=np.log(S+1e-6), sr=sr, n_mfcc=48)
    return log_mfcc.astype('float16').T


    
def extract_training_data(dir:str,rec_col:int,start_col:int,end_col:int,augmentations:int) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
     

    positive = parse_excel(dir,rec_col,start_col,end_col)
    # report_progress(10)
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
    # report_progress(25)

    #split negatives into k buckets, because of augmentation, then 
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
    from pathos.multiprocessing import Pool,cpu_count
    pool =  Pool(min(2+2*augmentations,cpu_count()))
    data = list(pool.map(lambda work: work[0](*(work[1])),work))

    random.shuffle(data)
    xs = []
    ys = []
    for d in data:
        for x in d[0]:
            xs.append(x)
        for y in d[1]:
            ys.append(y)


    # report_progress(75)
    



    data = list(pool.map(lambda work: work[0](*(work[1])),
        [
            [no_aug_data_pieces,[positive_test,CORRECT]],
            [no_aug_data_pieces,[negative_test,INCORRECT]]
        ]))
    random.shuffle(data)
    pool.close()
    pool.terminate()
    xs_test = []
    ys_test = []
    for d in data:
        for x in d[0]:
            xs_test.append(x)
        for y in d[1]:
            ys_test.append(y)

    # report_progress(100)

    return np.array(xs), np.array(xs_test), np.array(ys), np.array(ys_test)

def merge_training_data(files):
    all = [np.load(f) for f in files]
    all = [ [a[f] for f in a.files] for a in all]
    xs = np.concatenate([a[0] for a in all], axis=0)
    xs_test = np.concatenate([a[1] for a in all], axis=0)
    ys = np.concatenate([a[2] for a in all], axis=0)
    ys_test = np.concatenate([a[3] for a in all], axis=0)
    return xs,xs_test,ys,ys_test
    



