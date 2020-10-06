#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据处理
@author Hulk
"""
import wave
import pandas as pd
from glob import glob
from python_speech_features import fbank, delta
from config.config import *
import matplotlib.pyplot as plt
# plt.figure(figsize=(2.56, 2.56))

import pywt
import os
import numpy as np
from scipy.signal import butter, lfilter, resample, stft
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


# 巴特沃斯通带滤波
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def read_wavfile(filepath):
    # samplerate, wav_data = wavfile.read(filepath)
    wav_data, samplerate = librosa.load(filepath, sr=2000)
    wav_data = wav_data / np.max(wav_data)
    return wav_data, samplerate


# 巴特沃斯滤波、小波分解
def preprocess(filepath, outpath=None):
    if not os.path.isfile(filepath):
        print(filepath + "is not a file, please check it!!! ")
    wav_data, samplerate = read_wavfile(filepath)
    wav_data = librosa.resample(wav_data, 2000, 1000)
    wav_data_after_filter = butter_bandpass_filter(wav_data, 25, 400, 1000, 4)
    # 构建小波函数
    w = pywt.Wavelet('db6')
    # maxlev = pywt.dwt_max_level(len(wav_data_after_filter), w.dec_len)
    # print("maximum level is" + str(maxlev))
    threshold = 0.1  # 阈值
    coffs = pywt.wavedec(wav_data_after_filter, w, level=6)
    print(coffs)
    print(coffs.shape)
    coffs[len(coffs) - 1].fill(0)
    for i in range(1, len(coffs) - 1):
        coffs[i] = pywt.threshold(coffs[i], threshold * max(coffs[i]))
    final_data = pywt.waverec(coffs, 'db6')

    return final_data


def find_files(directory, pattern='**/*.wav'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)


def normalize_frames(m, epsilon=1e-12):
    x = [(v - np.mean(v)) / max(np.std(v), epsilon) for v in m]
    return x


def read_wav_data(filename):
    """
    读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
    """
    wav = wave.open(filename, "rb")  # 打开一个wav格式的音频文件流
    num_frame = wav.getnframes()  # 获取帧数
    framerate = wav.getframerate()  # 获取帧速率
    str_data = wav.readframes(num_frame)  # 读取全部的帧
    wav.close()  # 关闭流
    wave_data = np.frombuffer(str_data, dtype=np.short)  # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, 1
    wave_data = wave_data.T  # 将矩阵转置
    # wave_data = wave_data
    return wave_data, framerate


def get_fbank_feature(wavsignal, fs):
    # 获取输入特征
    feat, energy = fbank(wavsignal, samplerate=fs, nfilt=23)
    feat_d = delta(feat, 2)
    feat_dd = delta(feat_d, 2)

    wav_feature = np.column_stack((normalize_frames(feat), normalize_frames(feat_dd), normalize_frames(feat_d)))
    wav_feature = np.reshape(np.array(wav_feature), (3, len(wav_feature), 23))
    return wav_feature


def data_catalog(datapath):
    datalibr = pd.DataFrame()
    datalibr['filename'] = find_files(datapath)
    # print(datalibr)
    datalibr['filename'] = datalibr['filename'].apply(lambda x: x.replace('\\', '/'))  # normalize windows paths
    datalibr['dataset_id'] = datalibr['filename'].apply(lambda x: x.split('/')[-2])
    datalibr['wav_id'] = datalibr['filename'].apply(lambda x: x.split('/')[-1])
    for dateset in os.listdir(datapath):
        label_file = os.path.join(datapath, dateset, 'REFERENCE.csv')
        lab = pd.read_csv(label_file, header=None)
        for j in range(len(lab)):
            for i in range(len(datalibr)):
                if datalibr.loc[i]['dataset_id'] == dateset and datalibr.loc[i]['wav_id'] == lab.loc[j][0]:
                    datalibr.loc[i]['label'] = lab.loc[j][1]

    return datalibr


def enframe(signal, nw, inc):
    '''将音频信号转化为帧。
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    '''
    signal_length = len(signal)  # 信号总长度
    if signal_length <= nw:  # 若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
    else:  # 否则，计算帧的总长度
        nf = int(np.ceil((1.0 * signal_length - nw + inc) / inc))
    pad_length = int((nf - 1) * inc + nw)  # 所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal = np.concatenate((signal, zeros))  # 填补后的信号记为pad_signal
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
                                                           (nw, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
    frames = pad_signal[indices]  # 得到帧信号
    #    win=np.tile(winfunc(nw),(nf,1))  # window窗函数，这里默认取1
    #    return frames*win   #返回帧信号矩阵
    return frames


def stft_specgram(x, picname=None, **params):  # picname是给图像的名字，为了保存图像
    f, t, zxx = stft(x, **params)
    plt.pcolormesh(t, f, np.abs(zxx))
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    if picname is not None:
        plt.savefig('..\\picture\\' + str(picname) + '.jpg')  # 保存图像
    plt.clf()  # 清除画布
    return t, f, zxx


def prep(datapath):
    np.random.seed(42)
    datasall = []
    for dataset in os.listdir(datapath):
        data_dir = os.path.join(datapath, dataset)
        label_file = os.path.join(data_dir, 'REFERENCE.csv')
        lab = pd.read_csv(label_file, header=None)
        dat = pd.DataFrame(lab, index=lab[0], copy=True)

        for data in os.listdir(data_dir):
            if not data.endswith(".wav"):
                continue
            if "." in data:
                dataname = data.split(".")[0]
            out_dir = os.path.join(FEATPATH, dataset + "/" + dataname)
            # if os.path.exists(out_dir):
            #     continue
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            # y, fs = read_wav_data(os.path.join(data_dir, data))
            # y = y * 1.0 / np.max(abs(y))
            # y_res = resample(y[0], len(y[0]) // 2)
            y_res = preprocess(os.path.join(data_dir, data))
            y_enf = enframe(y_res, 3 * 1000, 2 * 1000)
            data_y = []
            for idx in range(0, len(y_enf)):
                # f, t, zxx = stft(y_enf[idx], fs=2000,nfft=512,noverlap=90,window='hamming')
                #
                # plt.pcolormesh(t, f, np.abs(zxx))
                # wav_feat = get_fbank_feature(y_enf[idx], fs=2000)
                # plt.colorbar()
                # plt.specgram(y_enf[idx], Fs=2000, scale_by_freq=True, sides='default')
                # plt.axis('off')
                # plt.axes().get_xaxis().set_visible(False)
                # plt.axes().get_yaxis().set_visible(False)
                # fig = plt.gcf()
                # save_dir = out_dir + '/'+dataname+"_"+str(idx)+'.npy'
                # np.save(save_dir, y_enf[idx])
                x = y_enf[idx].tolist()
                x.append(lab.loc[lab.loc[:, 0] == dataname, 1].values[0].astype("float"))
                # data_y.append(x)
                datasall.append(x)
                # plt.savefig(save_dir, bbox_inches='tight', pad_inches=0)
                # plt.clf()
                # reference = dataname+"_"+str(idx) + " " + lab.loc[lab.loc[:, 0] == dataname, 1].astype(str) + " " + save_dir
                # if dataset in TRAIN_DICT:
                #
                #     # reference = dataname+"_"+str(idx)+ " " + lab.ioc[lab.ioc[:,0]==dataname, 1]
                #     write_to_txt(os.path.join(FEATPATH, 'train4.txt'), reference)
                # elif dataset == 'validation':
                # #     # reference = dataname+"_"+str(idx)+ " " + lab.ioc[lab.ioc[:,0]==dataname, 1]
                # #     write_to_txt(os.path.join(FEATPATH, 'test.txt'), reference)
                # # else:
                #     write_to_txt(os.path.join(FEATPATH, 'validation.txt'), reference)
                #

    np.save(os.path.join(FEATPATH, "train"), np.array(datasall))


def prep3(datapath='../data'):
    dict = {'N': 0, 'MVP': 1, 'MS': 2, 'MR': 3, 'AS': 4}
    for dataset in os.listdir(datapath):
        print(dataset)
        data_dir = os.path.join(datapath, dataset)
        for data in os.listdir(data_dir):
            if not data.endswith(".wav"):
                continue
            if "." in data:
                dataname = data.split(".")[0]
            out_dir = os.path.join(FEATPATH, dataset + "/" + dataname)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            y_res = preprocess(os.path.join(data_dir, data))
            y_enf = enframe(y_res, 2 * 2000, 1 * 2000)
            for idx in range(0, len(y_enf)):
                save_dir = out_dir + '/' + dataname + "_" + str(idx) + '.npy'
                np.save(save_dir, y_enf[idx])

                reference = dataname + "_" + str(idx) + save_dir

                write_to_txt(os.path.join(FEATPATH, 'train.txt'), reference)


def write_to_txt(filepath, reference, mode="a+", code="gbk"):
    with open(filepath, mode=mode) as f:
        f.writelines(reference + "\n")


def del_file():
    for data in os.listdir("../data/validation/"):
        if data.startswith("a"):
            os.remove("../data/training-a/" + data)
        elif data.startswith("b"):
            os.remove("../data/training-b/" + data)
        elif data.startswith("c"):
            os.remove("../data/training-c/" + data)
        elif data.startswith("d"):
            os.remove("../data/training-d/" + data)
        elif data.startswith("e"):
            os.remove("../data/training-e/" + data)
        elif data.startswith("f"):
            os.remove("../data/training-f/" + data)


from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle


def train_split(filedir=os.path.join(FEATPATH, 'train.npy'), ratio=0.3):
    data = np.load(filedir, allow_pickle=True)
    # print(data)
    # split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # # print(data[])
    # for train_index, test_index in split.split(data, data.iloc[:,2]):
    #     train = data.loc[train_index]
    #     test = data.loc[test_index]
    from imblearn.over_sampling import ADASYN
    sm = ADASYN(random_state=42)
    data = shuffle(data)
    train, test = train_test_split(data, test_size=ratio, random_state=20)
    x_train, y_train = sm.fit_resample(train[:, 0:-1], train[:, -1])
    y_train = np.reshape(y_train, (-1, 1))
    train = np.concatenate([x_train, y_train], axis=1)
    test = shuffle(test)
    test, val = train_test_split(test, test_size=0.3333333, random_state=20)

    np.save(os.path.join(FEATPATH, 'train5'), train)  # delimiter=“，” 才能一个数据一个格
    np.save(os.path.join(FEATPATH, 'test5'), test)
    np.save(os.path.join(FEATPATH, 'validation5'), val)


if __name__ == "__main__":
    # y, fs = read_wav_data('../data/training-a/a0058.wav')
    # l = get_fbank_feature(y, fs)
    # print(np.shape(l))
    # del_file()
    prep(DATAPATH)
    # prep3()
    train_split()

