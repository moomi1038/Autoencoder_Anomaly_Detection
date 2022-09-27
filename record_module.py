import os
import csv
import pyaudio
import numpy as np
import librosa
import librosa.display
import sys
import yaml
from time import sleep, time

with open("param.yaml") as f:
    param = yaml.load(f, Loader=yaml.FullLoader)


def time_recording(boolean, graph):
    global param
    global stream
    s = time()
    if boolean:
        y = np.array([])
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=param["PYAUDIO_CHANNELS"],
                        rate=param["AUDIO_SAMPLERATE"],
                        input=True,
                        frames_per_buffer=param["PYAUDIO_CHUNK"])
                        
        for j in range(0, int(param["AUDIO_SAMPLERATE"] / param["PYAUDIO_CHUNK"] * param["PYAUDIO_SECONDS"])):
            data = stream.read(param["PYAUDIO_CHUNK"])
            numpydata = np.frombuffer(data, dtype=np.int16)
            y = np.append(y,numpydata)

        if graph == 'Log Mel':
            mel_spectrogram = librosa.feature.melspectrogram(y=y,sr=param["AUDIO_SAMPLERATE"],n_fft=param["LIBROSA_N_FFT"], hop_length=param["LIBROSA_HOP_LENGTH"], n_mels= param["LIBROSA_N_MELS"], power=param["LIBROSA_POWER"])
            rec_data = 20.0 / param["LIBROSA_POWER"] * np.log10(mel_spectrogram + sys.float_info.epsilon)

        e = time()
        print("record_time  : ",e-s)
        
        return rec_data

def save_csv(record_data, i, graph):
    if graph == "Log Mel":
        dir_path_file = os.path.abspath('./{dir_name}'.format(dir_name = param["DIR_NAME_TRAIN_LOGMEL"]))

    file_name = dir_path_file + '/id01_' + '{:0>5}'.format(i) + '_.csv'
    
    f = open(file_name, 'w')
    wr = csv.writer(f)
    wr.writerows(record_data)
    f.close()
