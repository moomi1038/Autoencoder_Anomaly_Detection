import os
import matplotlib.pyplot as plt
import csv
import yaml

with open("param.yaml") as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

def save_csv(record_data, i, graph):
    if graph == "Log Mel":
        dir_path_file = os.path.abspath('./{dir_name}'.format(dir_name = param["DIR_NAME_TRAIN_LOGMEL"]))
    elif graph == "STFT":
        dir_path_file = os.path.abspath('./{dir_name}'.format(dir_name = param["DIR_NAME_TRAIN_STFT"]))
    file_name = dir_path_file + '/id01_' + '{:0>5}'.format(i) + '_.csv'
    
    f = open(file_name, 'w')
    wr = csv.writer(f)
    wr.writerows(record_data)
    f.close()

def loss_plot(loss, val_loss):
    fig = plt.figure(figsize=(30, 10))
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

def del_file():
    global param
    for file in os.scandir(param["DIR_NAME_TRAIN_LOGMEL"]):
        # print('file.path', file.path)
        os.remove(file.path)

