import os
import glob
import numpy as np
from tqdm.auto import tqdm 
import module.keras_model as keras_model
import module.utils as utils
import yaml
import pandas as pd

path = os.getcwd()
param_path = os.path.join(path,"param.yaml")

with open(param_path) as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

def test_file_list_generator(normal_dir,abnormal_dir,
                             prefix_normal = "id01",
                             ext="parquet"):

    normal_files = sorted(glob.glob("{dir}/{prefix_normal}*.{ext}".format(dir=normal_dir,
                                                                          prefix_normal=prefix_normal,
                                                                          ext=ext)))
    normal_labels = np.zeros(len(normal_files))
    
    anomaly_files = sorted(glob.glob("{dir}/id0[!1]*.{ext}".format(dir=abnormal_dir,
                                                                    ext=ext)))
    anomaly_labels = np.ones(len(anomaly_files))

    files = np.concatenate((normal_files, anomaly_files), axis=0)
    labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
    
    return files, labels

def validation_run():

    model_file = os.path.join(path,param["DIR_NAME_MODEL"])
    normal_dir = os.path.join(path,param["DIR_NAME_TRAIN_STFT"])
    abnormal_dir = os.path.join(path,param["DIR_NAME_TEST_STFT"])

    model = keras_model.load_model(model_file)

    test_files, y_true = test_file_list_generator(normal_dir,abnormal_dir)

    y_pred = [0. for k in test_files]

    for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):
        try:
            data = pd.read_parquet(file_path).to_numpy().T
            errors = np.mean(np.square(data - model.predict(data)), axis=1)
            reconstruction_error = np.mean(errors)
            y_pred[file_idx] = reconstruction_error

        except Exception as e:
            print(e)

    utils.plotting(y_true,y_pred, "valid")
