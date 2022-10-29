import numpy as np
# from time import time

def test_run(data, model, threshold):
    # s = time()

    final_data = data.T
    errors = np.mean(np.square(final_data - model.predict(final_data)), axis=1)
    reconstruction_error = np.round(np.mean(errors),2)
    print("RECONSTRUCTION ERROR : ",reconstruction_error)
    # e = time()
    # print("test_time  : ",e-s)

    if reconstruction_error > threshold:
        return False
    else:
        return True     
