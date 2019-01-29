import argparse
import torch
import os
import cv2

parser = argparse.ArgumentParser('scrip to check SVHN data')
parser.add_argument('--file', default='data/train.txt', type=str, help='path to mat file')

args = parser.parse_args()


def read_list(path):
    file_list = []
    for root, _, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] == '.png':
                file_list.append(os.path.join(root, file))
    return file_list


def run():
    with open(args.file, 'r') as f:
        total = f.readlines()
    for idx in range(len(total)):
        try:
            data = total[idx].strip().split()
            img = cv2.imread(data[0])
            left, top, right, bottom = map(lambda x: (0, int(x))[int(x) > 0], [data[1], data[2], data[3], data[4]])
            img = img[top:bottom, left:right, :]
            img = cv2.resize(img, (64, 64))
            labels = torch.ones(5) * 10
            length = int(data[5])
            for i in range(length):
                labels[i] = int(data[6 + i])
        except TypeError:
            print(data[0])
            print('label:{}, length:{}'.format(labels, length))
            cv2.imshow('cv', img)
            cv2.waitKey(0)


if __name__ == '__main__':
    run()
