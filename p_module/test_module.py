import numpy as np
import module.keras_model as keras_model
import yaml
from time import time
with open("param.yaml") as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

model = keras_model.load_model("./model/model_logmel.hdf5")   

def test_run(data, graph, threshold):
    global model
    s = time()
    # test_data = data

    dims = param["LIBROSA_N_MELS"] * param["FRAMES"]
    vector_array_size = len(data[0, :]) - param["FRAMES"] + 1

    vector_array = np.zeros((vector_array_size, dims))
    for t in range(param["FRAMES"]):
        vector_array[:, param["LIBROSA_N_MELS"] * t: param["LIBROSA_N_MELS"] * (t + 1)] = data[:, t: t + vector_array_size].T

    final_data = vector_array
    errors = np.mean(np.square(final_data - model.predict(final_data)), axis=1)
    reconstruction_error = np.mean(errors)
    e = time()
    print("test_time  : ",e-s)
    # print("RECONSTRUCION ERROR :", reconstruction_error)
    if reconstruction_error > threshold:
        return False
    else:
        return True
   
def model_load():
    global model
    model = keras_model.load_model("./model/model_logmel.hdf5")   