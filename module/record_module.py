import pyaudio
import numpy as np
import librosa
from time import time
import os
import pandas as pd
import tensorflow as tf
from scipy.io import wavfile

model = tf.saved_model.load('./model/DTLN_norm_500h_saved_model')
infer = model.signatures["serving_default"]
print(infer)
def time_recording(rate, chunk, n_fft):
    s = time()
    frames = []
    y = np.array([])
    data = np.array([])
    rec_data = np.array([])
    p = pyaudio.PyAudio()
    
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
                    
    for _ in range(0, int(rate / chunk * 3)):
        data = stream.read(chunk)
        frames.append(np.frombuffer(data, dtype=np.float32))
    
    y = np.hstack(frames)

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

# time_recording(16000,1024,512)