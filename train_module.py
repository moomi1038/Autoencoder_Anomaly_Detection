import os
import csv
import glob
import keras_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

with open("param.yaml") as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

fig = plt.figure(figsize=(30, 10))

def loss_plot(loss, val_loss):
    ax = fig.add_subplot(1, 1, 1)
    ax.cla()
    ax.plot(loss)
    ax.plot(val_loss)
    ax.set_title("Model loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(["Train", "Validation"], loc="upper right")

def save_figure(name):
    plt.savefig(name)

def file_list_generator(graph,
                        id = "id01",
                        ext = "csv"):
    global param
    if graph == "Log Mel":
        dir_name = param["DIR_NAME_TRAIN_LOGMEL"]
    elif graph == "STFT":
        dir_name = param["DIR_NAME_TRAIN_STFT"]

    training_list_path = os.path.abspath("{dir_name}/{id}_*.{ext}".format(dir_name=dir_name, id = id, ext=ext))
    files = sorted(glob.glob(training_list_path))

    return files

def file_to_vector_array(FILES, graph):
    global param
    if graph == "Log Mel":
        dims = param["LIBROSA_N_MELS"] * param["FRAMES"]
        for i in range(len(FILES)):
            temp = pd.read_csv(FILES[i],header=None).to_numpy()
            
            vector_array_size = len(temp[0, :]) - param["FRAMES"] + 1
            if vector_array_size < 1:
                return np.empty((0, dims))

            vector_array = np.zeros((vector_array_size, dims))
            for t in range(param["FRAMES"]):
                vector_array[:, param["LIBROSA_N_MELS"] * t: param["LIBROSA_N_MELS"] * (t + 1)] = temp[:, t: t + vector_array_size].T

            if i == 0:
                dataset = np.zeros((vector_array.shape[0] * len(FILES), dims), float)
            dataset[vector_array.shape[0] * i: vector_array.shape[0] * (i + 1), :] = vector_array

        return dataset


def train(train_data,
        graph,
        dir_name = param["DIR_NAME_MODEL"],
        ext = "hdf5",
        result_dir_name = param["DIR_NAME_RESULT"],
        result_ext = "png"):
    global param

    if graph == "Log Mel":
        model_file_path = os.path.abspath("./{dir_name}/model_logmel.{ext}".format(dir_name=dir_name, ext=ext))
    elif graph == "STFT":
        model_file_path = os.path.abspath("./{dir_name}/model_stft.{ext}".format(dir_name=dir_name, ext=ext))

    history_img = "./{result_dir_name}/loss.{result_ext}".format(result_dir_name=result_dir_name, result_ext=result_ext)
    
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
    
    loss_plot(history.history["loss"], history.history["val_loss"])
    save_figure(history_img)

    model.save(model_file_path)


def train_run(graph):
    global param
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    os.makedirs("./" + param["DIR_NAME_RESULT"], exist_ok=True)
    FILES = file_list_generator(graph)
    train_data = file_to_vector_array(FILES, graph)
    train(train_data, graph)


def save_csv(record_data, i, graph):
    global param
    if graph == "Log Mel":
        dir_path_file = os.path.abspath('./{dir_name}'.format(dir_name = param["DIR_NAME_TRAIN_LOGMEL"]))
    elif graph == "STFT":
        dir_path_file = os.path.abspath('./{dir_name}'.format(dir_name = param["DIR_NAME_TRAIN_STFT"]))

    file_name = dir_path_file + '/id01_' + '{:0>5}'.format(i) + '_.csv'
    
    f = open(file_name, 'w')
    wr = csv.writer(f)
    wr.writerows(record_data)
    f.close()

