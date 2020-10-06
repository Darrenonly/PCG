########k折划分############
import os

import torch

from train_test.pytorchtool import EarlyStopping
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from config.config import FEATPATH
# from models.densenet import densenet_cifar
from models.densenet_stem import densenet_cifar
from train_test.eval import eval_net
from train_test.my_dataset2 import My_Dataset
import numpy as np

val_best_losss = 0


##########定义dataset##########
class TraindataSet(Dataset):
    def __init__(self, train_features, train_labels):
        self.x_data = train_features
        self.y_data = train_labels
        self.len = len(train_labels)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def get_k_fold_data(k, i, X, y):  # 此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = len(X) // k  # 每份的个数:数据总条数/折数（组数）

    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        # idx 为每组 valid
        X_part, y_part = torch.Tensor(X[idx]), torch.Tensor(y[idx])
        if j == i:  # 第i折作valid
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)  # dim=0增加行数，竖着连接
            y_train = torch.cat((y_train, y_part), dim=0)
    # print(X_train.size(),X_valid.size())
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs=150, learning_rate=0.2, weight_decay=0.0001, batch_size=128):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0

    for i in range(k):
        X_train, y_train, X_valid, y_valid = get_k_fold_data(k, i, X_train, y_train)  # 获取k折交叉验证的训练和验证数据
        net = densenet_cifar()  # 实例化模型
        if torch.cuda.is_available():
            net = net.cuda()
        # 每份数据进行训练,体现步骤三
        losses, val_losses, train_acc, val_acc = train(net, X_train, y_train, X_valid, y_valid, num_epochs,
                                                       learning_rate, weight_decay, batch_size)

        print('*' * 25, '第', i + 1, '折', '*' * 25)
        print('train_loss:%.6f' % losses[-1], 'train_acc:%.4f\n' % train_acc[-1],
              'valid loss:%.6f' % val_losses[-1], 'valid_acc:%.4f' % val_acc[-1])
        train_loss_sum += losses[-1]
        valid_loss_sum += val_losses[-1]
        train_acc_sum += train_acc[-1]
        valid_acc_sum += val_acc[-1]
    print('#' * 10, '最终k折交叉验证结果', '#' * 10)
    # 体现步骤四
    print('train_loss_sum:%.4f' % (train_loss_sum / k), 'train_acc_sum:%.4f\n' % (train_acc_sum / k),
          'valid_loss_sum:%.4f' % (valid_loss_sum / k), 'valid_acc_sum:%.4f' % (valid_acc_sum / k))
    torch.save(net.state_dict(), "../checkpoint/desenet_stem.pth")

    # eval_net(weight_decay=weight_decay, learning_rate=learning_rate,model=net)


# 训练函数
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay,
          batch_size):
    # train_ls, test_ls = [], []  ##存储train_loss,test_loss
    dataset = TraindataSet(train_features, train_labels)
    train_iter = DataLoader(dataset, batch_size, shuffle=True)
    dataset1 = TraindataSet(test_features, test_labels)
    test_iter = DataLoader(dataset1, batch_size, shuffle=True)
    early_stopping = EarlyStopping(patience=20, verbose=True)
    # if os.path.exists('checkpoint.pt'):
    #     net.load_state_dict(torch.load('checkpoint.pt'))
    global val_best_losss
    # 将数据封装成 Dataloder 对应步骤（2）

    # 这里使用了Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    for epoch in range(num_epochs):
        net.train()
        correct = 0
        for i, (X, y) in enumerate(train_iter):  # 分批训练
            X = X.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            output, _ = net(X)
            loss = loss_func(output, y.long())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # 计算正确率
            y_hat = output
            pred = y_hat.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()

            if (i + 1) % 100 == 0:
                # 每10个batches打印一次loss
                print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch + 1, num_epochs,
                                                                    i + 1, len(train_features) // batch_size,
                                                                    loss.item()))
        accuracy = 100. * correct / len(train_features)
        print('Epoch: {}, Loss: {:.5f}, Training set accuracy: {}/{} ({:.3f}%)'.format(
            epoch + 1, loss.item(), correct, len(train_features), accuracy))
        train_acc.append(accuracy)

        # 每个epoch计算测试集accuracy
        net.eval()
        val_loss = val_best_losss
        correct = 0
        with torch.no_grad():
            for x, y in test_iter:
                x = x.cuda()
                y = y.cuda()
                optimizer.zero_grad()
                output, _ = net(x)
                loss = loss_func(output, y.long()).item()
                # optimizer.step()
                y_hat = output
                val_loss += loss * len(y)  # sum up batch loss
                pred = y_hat.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(y.view_as(pred)).sum().item()

        val_losses.append(val_loss / len(test_features))
        accuracy = 100. * correct / len(test_features)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            val_loss / len(test_features), correct, len(test_features), accuracy))
        val_acc.append(accuracy)
        early_stopping(val_loss / len(test_features), net)

        if early_stopping.early_stop:
            print("Early stopping")
            val_best_losss = val_loss / len(test_features)
            break
    return losses, val_losses, train_acc, val_acc


train_data = My_Dataset(filepath=FEATPATH + 'train6.txt', transform=None)
# val_data = My_Dataset(filepath=FEATPATH + "validation5.txt", transform=None)

train_size = len(train_data)
# 因为得到的dataset是一个数组字典，所以只能一个个往数组里添加
train_dataset = []
teat_train = []
for i in range(train_size):
    train_dataset.append(train_data[i])
train_d = [x for x, _ in train_dataset]
train_l = [y for _, y in train_dataset]
# loss_func = nn.CrossEntropyLoss().cuda()  ###申明loss函
loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([3.0, 1.0])).float()).cuda()  # 申明loss函数
# print(train_d[:])
k_fold(10, train_d, train_l)  # k=10,十折交叉验证
