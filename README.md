# Make Sputter by Raspberrypi, PLC(DN32S) 

## 1. Outline

Autoencoder Anomaly Detection project with Teraleader by 2022.03 ~ 2022.12.


## 2. Goal

Using Deep learning architecture for industrial safety development.

We choose CNC machine, specific example of industrial equipment.

## 3. Component

Macbook air 2020 (M1)

PLC (LS DN32S)

CNC

## 4. WorkFlow

<img width="80%" src="https://user-images.githubusercontent.com/61678329/211233816-a6648730-58a6-41d2-ad4c-5414f8995575.png"/>


## 5. Explanation

### Used Library

PyQt6, Pyserial, Tensorflow, librosa, onnxruntime, parquet

### Signal Rules

Digital Signal Rules

1. binary number -> hex number # fill 0 to make the number length is 4
2. '\x0501WSS0106%PW012'+ number +'\x04'

Analog Signal Rules

1. number -> hex number # fill 0 to make the number length is 4
2. '\x0501WSS0106%MW'+ address + number +'\x04'

manual : https://www.ls-electric.com/ko/product/view/P01121


