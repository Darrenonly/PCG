import os
from config.config import *


def del_file(datapath):
    train_path = ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']
    val_path = 'validation'
    val_data = []
    for dataset in os.listdir(datapath):
        data_dir = os.path.join(datapath, dataset)
        for data in os.listdir(data_dir):
            data_path = os.path.join(data_dir, data)
            if os.path.isfile(data_path):
                if data.startswith("."):
                    os.remove(data_path)
                    print(data + 'del done!')
                elif not data.endswith(".wav"):
                    if data.endswith(".csv"):
                        continue
                    os.remove(data_path)
                    print(data + 'del done!')

    val_data_path = os.path.join(datapath, val_path)
    for _data in os.listdir(val_data_path):
        if not _data.endswith('.csv'):
            val_data.append(_data)
    for train_path in train_path:
        train_data_path = os.path.join(datapath, train_path)
        for train_data in os.listdir(train_data_path):
            if train_data in val_data:
                # print(train_data)
                train_data = os.path.join(train_data_path, train_data)
                os.remove(train_data)
    print('Del File Done!')


if __name__ == "__main__":
    del_file(DATAPATH)
