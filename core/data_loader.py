import torch
import torch.utils.data as data
import cv2
import numpy as np


class SVHNLoader(data.Dataset):

    def __init__(self, file):
        super(SVHNLoader, self).__init__()

        with open(file, 'r') as f:
            self.data = f.readlines()

    def __getitem__(self, idx):
        data = self.data[idx].strip().split()
        img = cv2.imread(data[0])
        left, top, right, bottom = map(lambda x: (0, int(x))[int(x) > 0], [data[1], data[2], data[3], data[4]])
        img = img[top:bottom, left:right, :]
        img = cv2.resize(img, (64, 64))
        labels = torch.ones(5)*10
        length = int(data[5])
        for i in range(length):
            labels[i] = int(data[6+i])
        # print('leng:{}, labels:{}'.format(length, labels))
        # cv2.imshow('cv', img)
        # cv2.waitKey(0)
        return torch.from_numpy(img / 255.0).permute(2, 0, 1).float(), torch.tensor(labels).long(), length

    def __len__(self):
        r"""
        :return: total number of samples
        """
        return len(self.data)
