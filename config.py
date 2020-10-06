
"""
一些列常量参数
@author: Hulk
"""

DATAPATH = "../data/"

FEATPATH = "../feat_sr1k_5/"

TRAIN_DICT = ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f', 'validation']

TRAIN_FILE = FEATPATH + 'train5.txt'
TEST_FILE = FEATPATH + 'test5.txt'
VALIDATION_FILE = FEATPATH + 'validation5.txt'

TRAIN_data = FEATPATH + 'train_data.pkl'
TEST_data = FEATPATH + 'test_data.pkl'
VALIDATION_data = FEATPATH + 'validation_data.pkl'


NUM_CLASS = 2
BATCH_SIZE = 1
EPOCH = 10
CHECKPOINT_DIR = '..\\checkpoint\\'
