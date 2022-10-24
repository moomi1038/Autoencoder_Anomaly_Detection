import os
import glob
from tqdm import tqdm
import module.keras_model as keras_model
import pandas as pd
# import yaml

path = os.getcwd()
# param_path = os.path.join(path,"param.yaml")

# with open(param_path) as f:
#     param = yaml.load(f, Loader=yaml.FullLoader)

def file_list_generator(id,ext, normal_dir, abnormal_dir):
    # if id == "id01":
    #     dir_name = os.path.join(path,param["DIR_NAME_TRAIN_STFT"])
    # else:
    #     dir_name = os.path.join(path,param["DIR_NAME_TEST_STFT"])
    if id == "id01":
        dir_name = os.path.join(path,normal_dir)
    else:
        dir_name = os.path.join(path,abnormal_dir)

    training_list_path = os.path.abspath("{dir_name}/{id}_*.{ext}".format(dir_name=dir_name, id = id, ext=ext))
    files = sorted(glob.glob(training_list_path))

    for i in tqdm(files,total=len(files)):
        data = pd.read_parquet(i).to_numpy().T
    return data



def train(normal_dir, abnormal_dir, loss, optimizer, echochs, batch_size, shuffle, validation_split, verbose, model_path):
    train_data = file_list_generator("id01", "parquet", normal_dir, abnormal_dir)

    model = keras_model.get_model(train_data.shape[1], name = 'model')
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    
    history = model.fit(train_data,
                        train_data,
                        epochs=echochs,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        validation_split=validation_split,
                        verbose=verbose)

    model_file_path = os.path.join(path, model_path)
    model.save(model_file_path)

    return True, history
