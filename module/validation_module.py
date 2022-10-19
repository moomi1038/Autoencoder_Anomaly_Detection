import os
import glob
import csv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import numpy as np
from tqdm import tqdm 
import module.keras_model as keras_model
import yaml
import pandas as pd
import matplotlib.pyplot as plt

with open("param.yaml") as f:
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

def validation_run(GRAPH):
    os.makedirs(param["DIR_NAME_RESULT"], exist_ok=True)

    csv_lines = []

    model_file = "./model/model_stft.hdf5"
    normal_dir = param["DIR_NAME_TRAIN_STFT"]
    abnormal_dir = param["DIR_NAME_TEST_STFT"]

    model = keras_model.load_model(model_file)

    performance = []

    test_files, y_true = test_file_list_generator(normal_dir,abnormal_dir)

    y_pred = [0. for k in test_files]

    for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):
        try:
            data = pd.read_parquet(file_path).to_numpy().T
            errors = np.mean(np.square(data - model.predict(data)), axis=1)
            reconstruction_error = np.mean(errors)
            y_pred[file_idx] = reconstruction_error

        except:
            print("file broken")

        auc = roc_auc_score(y_true, y_pred)

        csv_lines.append([auc])
        performance.append([auc])

    # threshold_fixed = []
    # precision_rt, recall_rt, threshold_rt = precision_recall_curve(y_true,y_pred)
    # best_cnt_dic = abs(precision_rt-recall_rt)
    # threshold_fixed.append([threshold_rt[np.argmin(best_cnt_dic)]])
    # threshold_file_path = os.path.abspath("{dir_name}/threshold.csv".format(dir_name=param["DIR_NAME_RESULT"]))
    
    # save_csv(threshold_file_path, threshold_fixed)

    # param["THRESHOLD_LOGMEL"] = round(float(threshold_rt[np.argmin(best_cnt_dic)]),1)

    # with open('param.yaml', 'w') as file:
    #     yaml.dump(param, file, default_flow_style=False)

        
    # print("\n============ END OF TEST FOR A MACHINE ID ============")

    # averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
    # csv_lines.append(["Average"] + list(averaged_performance))
    
    # csv_lines.append([])
    # result_path = "{result}/result.csv".format(result=param["DIR_NAME_RESULT"])
    # save_csv(save_file_path=result_path, save_data=csv_lines)

    # del_file()

    # return False