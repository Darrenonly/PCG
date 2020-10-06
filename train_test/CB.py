
import torch as t
import torch.nn.functional as F
import torch.nn as nn




class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = t.ones(class_num, 1)
        else:
            if alpha:
#                 self.alpha = t.ones(class_num, 1, requires_grad=True)
                self.alpha = t.tensor(alpha, requires_grad=True)
                # print('alpha初始\n', self.alpha)
                # print('alpha shape\n', self.alpha.shape)
#             else:
#                 self.alpha = t.ones(class_num, 1*alpha).cuda()
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        
    def forward(self, inputs, targets):
        #input.shape = (N, C)
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1) #經過softmax 概率
        #---------one hot start--------------#
        class_mask = inputs.data.new(N, C).fill_(0)  #生成和input一样shape的tensor
        # print('依照input shape制作:class_mask\n', class_mask)
        class_mask = class_mask.requires_grad_() #需要更新， 所以加入梯度计算
        ids = targets.view(-1, 1) #取得目标的索引
        # print('取得targets的索引\n', ids)
        class_mask.data.scatter_(1, ids.data, 1.) #利用scatter将索引丢给mask
        # print('targets的one_hot形式\n', class_mask) #one-hot target生成
        #---------one hot end-------------------#
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
#         alpha = self.alpha[ids.data.view(-1, 1)]
#         alpha = self.alpha[ids.view(-1)]
        alpha = self.alpha
        # print('alpha值\n', alpha)
        # print('alpha shape\n', alpha.shape)
        
        probs = (P*class_mask).sum(1).view(-1, 1) 
        # print('留下targets的概率（1的部分），0的部分消除\n', probs)
        #将softmax * one_hot 格式，0的部分被消除 留下1的概率， shape = (5, 1), 5就是每个target的概率
        
        log_p = probs.log()
        # print('取得对数\n', log_p)
        #取得对数
        
        batch_loss = -alpha*(t.pow((1-probs), self.gamma))*log_p #對應下面公式
        # print('每一个batch的loss\n', batch_loss)
        #batch_loss就是取每一个batch的loss值
        
        
        #最终将每一个batch的loss加总后平均
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        # print('loss值为\n', loss)
        return loss