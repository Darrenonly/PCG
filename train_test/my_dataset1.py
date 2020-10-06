import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np


# import keras

# 定义读取文件的格式
def default_loader(path):
    return Image.open(path).convert('RGB')


class My_Dataset(Dataset):
    def __init__(self, filepath=None, transform=None, target_transform=None, loader=default_loader):
        super(My_Dataset, self).__init__()
        fh = open(filepath, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')
            content = line.split(" ")
            if content[1] == '':
                imgs.append((content[2], 0))
            else:
                imgs.append((content[2], int(content[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]

        # label = keras.utils.to_categorical(label, 2).astype("long")
        # print(label)
        # img = self.loader(fn)
        img = np.load(fn)
        img = torch.from_numpy(img)
        img = img.type(torch.FloatTensor)  # 转Float
        # img = img.cuda()  # 转cuda
        # if self.transform is not None:
        #     img = self.transform(img)
        # label = torch.Tensor(label)
        img = torch.reshape(img, (1, len(img)))

        return img, label

    def __len__(self):
        return len(self.imgs)
