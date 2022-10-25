import pyaudio
import numpy as np
import librosa
import onnxruntime
# from time import time
import os
import pandas as pd

path = os.getcwd()
inter1_path = os.path.join(path, "model/model_1.onnx")
inter2_path = os.path.join(path, "model/model_2.onnx")

interpreter_1 = onnxruntime.InferenceSession(inter1_path)
interpreter_2 = onnxruntime.InferenceSession(inter2_path)

def time_recording(rate, chunk, n_fft):
    # s = time()
    frames = []
    audio = []
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

    audio = np.hstack(frames)

    out_file = denoise(audio)

    result = audio - out_file

    rec_data = librosa.amplitude_to_db(np.abs(librosa.stft(y=np.abs(result),n_fft=n_fft)),top_db=100)

    # t = time()

    # print("record time :", t - s)

    return rec_data

def denoise(audio):
    ##########################
    # the values are fixed, if you need other values, you have to retrain.
    # The sampling rate of 16k is also fix.
    block_len = 512
    block_shift = 128
    # load models
    
    model_input_names_1 = [inp.name for inp in interpreter_1.get_inputs()]
    # preallocate input
    model_inputs_1 = {
                inp.name: np.zeros(
                    [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                    dtype=np.float32)
                for inp in interpreter_1.get_inputs()}
    # load models
    
    model_input_names_2 = [inp.name for inp in interpreter_2.get_inputs()]
    # preallocate input
    model_inputs_2 = {
                inp.name: np.zeros(
                    [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                    dtype=np.float32)
                for inp in interpreter_2.get_inputs()}

    out_file = np.zeros((len(audio)))
    # create buffer
    in_buffer = np.zeros((block_len)).astype('float32')
    out_buffer = np.zeros((block_len)).astype('float32')
    # calculate number of blocks
    num_blocks = (audio.shape[0] - (block_len-block_shift)) // block_shift
    # iterate over the number of blcoks  
    for idx in range(num_blocks):
        # shift values and write to buffer
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = audio[idx*block_shift:(idx*block_shift)+block_shift]
        # calculate fft of input block
        in_block_fft = np.fft.rfft(in_buffer)
        in_mag = np.abs(in_block_fft)
        in_phase = np.angle(in_block_fft)
        # reshape magnitude to input dimensions
        in_mag = np.reshape(in_mag, (1,1,-1)).astype('float32')
        # set block to input
        model_inputs_1[model_input_names_1[0]] = in_mag
        # run calculation 
        model_outputs_1 = interpreter_1.run(None, model_inputs_1)
        # get the output of the first block
        out_mask = model_outputs_1[0]
        # set out states back to input
        model_inputs_1[model_input_names_1[1]] = model_outputs_1[1]  
        # calculate the ifft
        estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex)
        # reshape the time domain block
        estimated_block = np.reshape(estimated_block, (1,1,-1)).astype('float32')
        # set tensors to the second block
        # interpreter_2.set_tensor(input_details_1[1]['index'], states_2)
        model_inputs_2[model_input_names_2[0]] = estimated_block
        # run calculation
        model_outputs_2 = interpreter_2.run(None, model_inputs_2)
        # get output
        out_block = model_outputs_2[0]
        # set out states back to input
        model_inputs_2[model_input_names_2[1]] = model_outputs_2[1]
        # shift values and write to buffer
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer  += np.squeeze(out_block)
        # write block to output file
        out_file[idx*block_shift:(idx*block_shift)+block_shift] = out_buffer[:block_shift]
        
    
    return out_file

def save_data(record_data, i, path):
    # s = time()
    head = [str(i) for i in range(record_data.shape[1])]
    df = pd.DataFrame(record_data, columns= head)
    file_name = ''.join(["id01_",'{:0>5}'.format(i),"_.parquet"])
    file_path_name = os.path.join(path,file_name)
    df.to_parquet(file_path_name)
    # t = time()
    # print("save time :", t - s)
