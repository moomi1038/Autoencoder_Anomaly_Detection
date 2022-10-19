import numpy as np
from time import time

def test_run(data, model, threshold):
    s = time()

    final_data = data.T
    errors = np.mean(np.square(final_data - model.predict(final_data)), axis=1)
    reconstruction_error = np.round(np.mean(errors),2)

    e = time()
    print("test_time  : ",e-s)

    if reconstruction_error > threshold:
        return [reconstruction_error, False]
    else:
        return [reconstruction_error, True]
     

# model = keras_model.load_model("./model/model_stft.hdf5")   
