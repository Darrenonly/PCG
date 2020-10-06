import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np


# 定义读取文件的格式
def default_loader(path):
    return Image.open(path).convert('RGB')


class My_Dataset(Dataset):
    def __init__(self, filepath=None, transform=None, target_transform=None, loader=default_loader):
        super(My_Dataset, self).__init__()
        # fh = open(filepath, 'r')
        # imgs = []
        # for line in fh:
        #     line = line.strip('\n')
        #     line = line.rstrip('\n')
        #     content = line.split(" ")
        #     if(int(content[1]) == -1):
        #         imgs.append((content[2], 0))
        #     else:
        #         imgs.append((content[2], int(content[1])))
        self.imgs = np.load(filepath, allow_pickle=True)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        img = self.imgs[index][:2999]

        label = self.imgs[index][3000]
        if label == -1:
            label=0
        # img = self.loader(fn)
        # img = np.load(fn)
        img = torch.from_numpy(img)
        img = img.type(torch.FloatTensor)  # 转Float
        # img = img.cuda()  # 转cuda
        # if self.transform is not None:
        #     img = self.transform(img)

        img = torch.reshape(img, (1, len(img)))
        return img, label

    def __len__(self):
        return len(self.imgs)
