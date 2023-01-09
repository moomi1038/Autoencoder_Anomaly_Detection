# Autoencoder Anomaly Detection project

## 1. Outline

Autoencoder Anomaly Detection project with Teraleader by 2022.03 ~ 2022.12.

This project published in https://db.koreascholar.com/article.aspx?code=417571

## 2. Goal

Using Deep learning architecture for industrial safety development.

We choose CNC machine, specific example of industrial equipment.

## 3. Component

Macbook air 2020 (M1)

PLC (LS DN32S)

CNC

## 4. WorkFlow

<img width="80%" src="https://user-images.githubusercontent.com/61678329/211233816-a6648730-58a6-41d2-ad4c-5414f8995575.png"/>

<img width="80%" src="https://user-images.githubusercontent.com/61678329/211239914-f6d6e5b2-d52f-4363-8657-e804268926b1.png"/>

## 5. Explanation

### Used Library

PyQt6, Pyaudio, Pyserial, Tensorflow, librosa, onnxruntime, parquet

### Sound feature

This project use Pyaudio to record the sound.

parameter

'''

AUDIO_SAMPLERATE: 16000

PYAUDIO_CHUNK: 1024

LIBROSA_N_FFT: 512

'''

### Autoencoder

Project use Autoencoder because of the characteristic of Autoencoder.

key point of success of Ai program is find the feature of data.

Autoencoder is fit for find the feature of data and compare with new data.

Autoencoder : https://en.wikipedia.org/wiki/Autoencoder


parameter

'''

EPOCHS: 100

LOSS: mean_squared_error

OPTIMIZER: adam

PATIENCE: 3

SHUFFLE: true

VALIDATION_SPLIT: 0.1

BATCH_SIZE: 128

'''


### Threshold

<img width="80%" src="https://user-images.githubusercontent.com/61678329/211240078-43bce3a3-9c05-4d54-bb82-b30b3f859eaf.png"/>

We use Threshold value is " 1 : 1 = precision : recall ".

### Denoise

This project Denoise program from DTLN(https://github.com/breizhn/DTLN)

### PLC

Please check sputter (https://github.com/YEUNU/Sputter)
