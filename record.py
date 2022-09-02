import pyaudio
import numpy as np
import librosa
import librosa.display
import sys
import yaml

with open("param.yaml") as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

def time_recording(boolean, graph):
    global param
    if boolean:
        y = np.array([])
        frames = []
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=param["PYAUDIO_CHANNELS"],
                        rate=param["AUDIO_SAMPLERATE"],
                        input=True,
                        frames_per_buffer=param["PYAUDIO_CHUNK"])

        for j in range(0, int(param["AUDIO_SAMPLERATE"] / param["PYAUDIO_CHUNK"] * param["PYAUDIO_SECONDS"])):
            data = stream.read(param["PYAUDIO_CHUNK"])
            frames.append(data)
            numpydata = np.frombuffer(data, dtype=np.int16)
            y = np.append(y,numpydata)

        if graph == 'Log Mel':
            mel_spectrogram = librosa.feature.melspectrogram(y=y,sr=param["AUDIO_SAMPLERATE"],n_fft=param["LIBROSA_N_FFT"], hop_length=param["LIBROSA_HOP_LENGTH"], n_mels=param["LIBROSA_N_MELS"],power=param["LIBROSA_POWER"])
            rec_data = 20.0 / param["LIBROSA_POWER"] * np.log10(mel_spectrogram + sys.float_info.epsilon)
        
        elif graph == 'STFT':
            rec = librosa.stft(y=y,n_fft=param["LIBROSA_N_FFT"],hop_length=param["LIBROSA_HOP_LENGTH"],win_length=param["LIBROSA_WIN_LENGTH"])
            rec_data = 20.0 / param["LIBROSA_POWER"] * np.log10(rec + sys.float_info.epsilon)
        
        return rec_data
