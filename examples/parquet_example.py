import pandas as pd
import glob
import os
from tqdm import tqdm
path = "tr/test_stft"
training_list_path = os.path.abspath("{dir_name}/{id}_*.{ext}".format(dir_name=path, id = "id04", ext="csv"))

files = sorted(glob.glob(training_list_path))
record_data = pd.read_csv(files[0],header=None)
head = [str(i) for i in range(record_data.shape[1])]

for idx, i in tqdm(enumerate(files)):
    data = pd.read_csv(i,header=None).to_numpy()
    df = pd.DataFrame(data, columns= head)
    file_name = ''.join(["id04_",'{:0>5}'.format(idx),"_.parquet"])
    path_name = './tr/test_stft_parquet'

    file_path_name = os.path.join(path_name,file_name)
    df.to_parquet(file_path_name)

# def save_data(record_data, i):
#     head = [str(i) for i in range(record_data.shape[1])]
#     df = pd.DataFrame(record_data, columns= head)
#     file_name = ''.join(["id01_",'{:0>5}'.format(i),"_.parquet"])
#     path_name = './tr/train_stft'
#     file_path_name = os.path.join(path_name,file_name)
#     df.to_parquet(file_path_name)