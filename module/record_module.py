import pyaudio
import numpy as np
import librosa
from time import time
import os
import pandas as pd

def time_recording(rate, chunk, n_fft):
    s = time()
    y = np.array([])
    numpydata = np.array([])
    rec_data = np.array([])
    p = pyaudio.PyAudio()
    
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
                    
    for j in range(0, int(rate / chunk * 3)):
        numpydata = np.frombuffer(stream.read(chunk), dtype=np.int16)
        y = np.append(y,numpydata)

    rec_data = librosa.amplitude_to_db(np.abs(librosa.stft(y=np.abs(y),n_fft=n_fft)))
    t = time()
    print("record time :", t - s)
    return rec_data

def save_data(record_data, i, path):
    s = time()
    head = [str(i) for i in range(record_data.shape[1])]
    df = pd.DataFrame(record_data, columns= head)
    file_name = ''.join(["id01_",'{:0>5}'.format(i),"_.parquet"])
    file_path_name = os.path.join(path,file_name)
    df.to_parquet(file_path_name)
    t = time()
    print("save time :", t - s)