import pyaudio
import tensorflow as tf
import numpy as np
import librosa
from time import time
import os
import pandas as pd

def time_recording(rate, chunk, n_fft, infer):
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

    # y = np.array(frames)
    y = np.hstack(frames)

    block_len = 512
    block_shift = 128
    # load model

    # load audio file at 16k fs (please change)
    audio = y

    # preallocate output audio
    out_file = np.zeros((len(audio)))
    # create buffer
    in_buffer = np.zeros((block_len))
    out_buffer = np.zeros((block_len))
    # calculate number of blocks
    num_blocks = (audio.shape[0] - (block_len-block_shift)) // block_shift
    # iterate over the number of blcoks        
    for idx in range(num_blocks):
        # shift values and write to buffer
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = audio[idx*block_shift:(idx*block_shift)+block_shift]
        # create a batch dimension of one
        in_block = np.expand_dims(in_buffer, axis=0).astype('float32')
        # process one block
        out_block= infer(tf.constant(in_block))['conv1d_1']
        # shift values and write to buffer
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer  += np.squeeze(out_block)
        # write block to output file
        out_file[idx*block_shift:(idx*block_shift)+block_shift] = out_buffer[:block_shift]

    rec_data = librosa.amplitude_to_db(np.abs(librosa.stft(y=np.abs(out_file),n_fft=n_fft)))
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
