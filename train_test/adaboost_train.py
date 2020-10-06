import time
import numpy as np
import torch
from prefetch_generator import BackgroundGenerator
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torch.autograd import Variable


from config.config import FEATPATH
from models.densenet_stem import densenet_cifar
from train_test.my_dataset1 import My_Dataset


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def train2():
    feat_model = densenet_cifar()
    if torch.cuda.is_available():
        model = feat_model.cuda()
    model.load_state_dict(torch.load('../checkpoint\chk4/best1.pth'))
    model.eval()
    train_data = My_Dataset(filepath=FEATPATH + 'train1.txt', transform=None)
    # test_data = My_Dataset(filepath= FEATPATH + 'validation.txt', transform=test_transforms)
    print(len(train_data))
    # train_db, val_db = torch.utils.data.random_split(train_data, [32724, 3636])
    # val1_db, test_db = torch.utils.data.random_split(val_db, [1818, 1818])
    train_db, val_db = torch.utils.data.random_split(train_data, [63775, 7086])
    val1_db, test_db = torch.utils.data.random_split(val_db, [3543, 3543])
    # train_db, val_db = torch.utils.data.random_split(train_data, [2828, 314])
    # val1_db, test_db = torch.utils.data.random_split(val_db, [157, 157])/
    # 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
    train_loader = DataLoaderX(dataset=train_db, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoaderX(dataset=val1_db, batch_size=128, shuffle=True, num_workers=1)
    test_loader = DataLoaderX(dataset=test_db, batch_size=128, shuffle=False, num_workers=1)
    train_features=[]
    train_labels = []
    test_features =[]
    test_labels =[]
    # clf = AdaBoostClassifier(n_estimators=1000, algorithm='SAMME')
    clf = ExtraTreesClassifier()
    test_predict=[]
    for features, label in train_loader:
        with torch.no_grad():
            feat = Variable(features.cuda())
            out2,out1 = feat_model(feat)
            out1 = out1.cpu()
            clf.fit(out1, label)
        train_features.append(np.array(out1))
        train_labels.append(np.array(label))
    for features, label in test_loader:
        with torch.no_grad():
            feat = Variable(features.cuda())
            out2, out1 = feat_model(feat)
            out1 = out1.cpu()
            test_predict.extend(clf.predict(out1).astype(int))
        test_features.append(out1)
        test_labels.extend(np.array(label).astype(int))
    time_2 = time.time()
    print('Start training...')
    # n_estimators表示要组合的弱分类器个数；
    # algorithm可选{‘SAMME’, ‘SAMME.R’}，默认为‘SAMME.R’，表示使用的是real boosting算法，‘SAMME’表示使用的是discrete boosting算法
    # clf = AdaBoostClassifier(n_estimators=100, algorithm='SAMME.R')
    #
    # clf.fit(train_features, train_labels)
    # time_3 = time.time()
    # print('training cost %f seconds' % (time_3 - time_2))
    #
    # print('Start predicting...')
    # test_predict = clf.predict(test_features)
    # time_4 = time.time()
    # print('predicting cost %f seconds' % (time_4 - time_3))

    score = accuracy_score(test_labels, test_predict)


    print("The accruacy score is %f" % score)
    print(classification_report(test_labels, test_predict))


if __name__ == "__main__":
    train2()