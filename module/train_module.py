import os
import glob
from tqdm import tqdm
import module.keras_model as keras_model
import pandas as pd
import yaml

with open("param.yaml") as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

def file_list_generator(id = "id01",
                        ext = "parquet"):
    dir_name = param["DIR_NAME_TRAIN_STFT"]

    training_list_path = os.path.abspath("{dir_name}/{id}_*.{ext}".format(dir_name=dir_name, id = id, ext=ext))
    files = sorted(glob.glob(training_list_path))

    for i in tqdm(files,total=len(files)):
        data = pd.read_parquet(i).to_numpy().T
        
    return data

def train():
    train_data = file_list_generator()
    model = keras_model.get_model(train_data.shape[1], name = 'model')
    model.summary()
    model.compile(loss=param["LOSS"], optimizer=param["OPTIMIZER"])
    
    history = model.fit(train_data,
                        train_data,
                        epochs=param["EPOCHS"],
                        batch_size=param["BATCH_SIZE"],
                        shuffle=param["SHUFFLE"],
                        validation_split=param["VALIDATION_SPLIT"],
                        verbose=param["VERBOSE"])

    model_file_path = os.path.abspath("./model/model_stft.hdf5")
    model.save(model_file_path)

