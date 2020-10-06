"""
@suthor: Hulk
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from config.config import *
import numpy as np
from models.densenet_stem import densenet_cifar
from sklearn.metrics import classification_report, roc_curve, auc, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from models.densenet_stem_with_stem_block import densenet_cifar_normal
from train_test.CB import FocalLoss
from train_test.my_dataset1 import My_Dataset


def eval_net(test_loader=None, datanum=None, weight_decay=0.0001, learning_rate=0.02, net=None, ):
    val_losses = []
    val_acc = []

    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # criterion = FocalLoss(2, alpha=0.4, gamma=2, size_average=True).cuda()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.2, 0.8])).float()).cuda()

    net.eval()
    val_loss = 0
    correct = 0
    pred1 = []
    y1 = []
    # plot ROC curve and area the curve
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.cuda()
            y = y.cuda()
            output, out1 = net(x)
            # net.eval()
            #
            loss = criterion(output, y.long()).item()
            optimizer.zero_grad()
            # optimizer.step()
            y_hat = output
            val_loss += loss * len(y)  # sum up batch loss
            pred = y_hat.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()

            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y.cpu().numpy(), pred.cpu().numpy())
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            # plt.plot(fpr, tpr, lw=1, alpha=0.3,
            #          label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
            pred1.extend(pred.cpu().numpy())
            y1.extend(y.cpu().numpy())
    # # TP predict 和 label 同时为1
    # TP += ((pred == 1) & (y1 == 1)).cpu().sum().item()
    # # TN predict 和 label 同时为0
    # TN += ((pred == 0) & (y == 0)).cpu().sum().item()
    # # FN predict 0 label 1
    # FN += ((pred == 0) & (y == 1)).cpu().sum().item()
    # # FP predict 1 label 0
    # FP += ((pred == 1) & (y == 0)).cpu().sum().item()
    #
    # p = TP / (TP + FP)
    # r = TP / (TP + FN)
    # F1 = 2 * r * p / (r + p)
    # acc = (TP+ TN) / (TP + TN + FP + FN)
    # print('Eval set: p: {:.4f}, recall: {:.4f}, F1: {:.4f}, Accuracy: {:.3f}%\n'.format(
    #     p, r, F1, acc))
    val_losses.append(val_loss / datanum)
    accuracy = 100. * correct / datanum
    print('Eval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        val_loss / datanum, correct, datanum, accuracy))
    val_acc.append(accuracy)
    # return val_losses, val_acc
    print(classification_report(pred1, y1))
    print("Macc{:.4f}".format(balanced_accuracy_score(y1, pred1)))
    print(confusion_matrix(y1, pred1))
    # plt.show()
    # 画图
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    #
    # mean_tpr = np.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    # plt.plot(mean_fpr, mean_tpr, color='b',
    #          label=r'ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
    #          lw=2, alpha=.8)
    #
    # std_tpr = np.std(tprs, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                  label=r'$\pm$ 1 std. dev.')
    #
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic curve')
    # plt.legend(loc="lower right")
    # plt.show()


if __name__ == "__main__":

    model = densenet_cifar()
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('../checkpoint/chk_sr1k_0/best.pth'))
    test_data = My_Dataset(filepath=FEATPATH + "test5.txt", transform=None)
    t_loader = DataLoader(test_data, shuffle=False, batch_size=64)
    eval_net(test_loader=t_loader, datanum=len(test_data), net=model)
