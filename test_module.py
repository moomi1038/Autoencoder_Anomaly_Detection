import numpy as np
import keras_model
import yaml

with open("param.yaml") as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

def test_run(data, graph, threshold):
    global param
    global model_stft
    global model_logmel

    test_data = data
    if graph == "Log Mel":
        model = model_logmel
        threshold = param["THRESHOLD_LOGMEL"]
    elif graph == "STFT":
        model = model_stft
        threshold = param["THRESHOLD_STFT"]

    dims = param["LIBROSA_N_MELS"] * param["FRAMES"]
    vector_array_size = len(test_data[0, :]) - param["FRAMES"] + 1

    vector_array = np.zeros((vector_array_size, dims))
    for t in range(param["FRAMES"]):
        vector_array[:, param["LIBROSA_N_MELS"] * t: param["LIBROSA_N_MELS"] * (t + 1)] = test_data[:, t: t + vector_array_size].T

    final_data = vector_array
    errors = np.mean(np.square(final_data - model.predict(final_data)), axis=1)
    reconstruction_error = np.mean(errors)
    
    # print("RECONSTRUCION ERROR :", reconstruction_error)
    if reconstruction_error > threshold:
        return False
    else:
        return True

def model_load():
    global model_stft
    global model_logmel
    try:
        model_logmel = keras_model.load_model("./model/model_logmel.hdf5")   
        model_stft = keras_model.load_model("./model/model_stft.hdf5")

    except Exception as e:
        print(e)

model_load()