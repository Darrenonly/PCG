"""
@author: Hulk
"""
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.config import *
from process.data_process1 import read_wav_data, get_fbank_feature


def load_data(type='train'):
    if type == "":
        print("The type should be give that train test validation")
        return
    data_input = []
    data_label = []
    if type == 'train':
        train = pd.read_csv(TRAIN_FILE, header=None)

        for index, train_row in tqdm(train.iterrows(), total=len(train)):
            y, fs = read_wav_data(train_row[2])
            data_input.append(get_fbank_feature(y, fs))
            data_label.append(train_row[1])

    elif type == 'test':
        test = pd.read_csv(TEST_FILE, header=None)
        for index, test_row in tqdm(test.iterrows(), total=len(test)):
            y, fs = read_wav_data(test_row[2])
            data_input.append(get_fbank_feature(y, fs))
            data_label.append(test_row[1])
    else:
        validation = pd.read_csv(VALIDATION_FILE, header=None)
        for index, validation_row in tqdm(validation.iterrows(), total=len(validation)):
            y, fs = read_wav_data(validation_row[2])
            data_input.append(get_fbank_feature(y, fs))
            data_label.append(validation_row[1])
    print(np.shape(data_input))
    return pd.DataFrame([data_label, data_input])


def extract_feature_to_csv():
    for file in os.listdir(FEATPATH):
        if file == "train5.txt":
            df = load_data(type='train').T
            # print(df.T)
            df.to_pickle(TRAIN_data)
        elif file == 'test5.txt':
            df = load_data(type='test').T
            # print(df.T)
            df.to_pickle(TEST_data)
        elif file == 'validation5.txt':
            df = load_data(type='validation').T
            # print(df.T)
            df.to_pickle(VALIDATION_data)
        else:
            continue
    print("Extract feature and save to csv === done！")


def read_feature_from_csv(type='train'):
    if type not in ['train', 'test', 'validation']:
        print("The type should be give that train test validation")
        return
    if type == 'train':
        dat = pd.read_pickle(TRAIN_data)

    elif type == 'test':
        dat = pd.read_pickle(TEST_data)
    else:
        dat = pd.read_pickle(VALIDATION_data)
    return np.ndarray(dat.loc[:, 0]), np.ndarray(dat.loc[:, 1])


if __name__ == "__main__":
    extract_feature_to_csv()
