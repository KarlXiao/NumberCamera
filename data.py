"""
process data
"""
import os
import scipy.io as sio
import numpy as np
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt

train_datapath = 'data/train_32x32.mat'
test_datapath = 'data/test_32x32.mat'
load_train = sio.loadmat(train_datapath)
load_test = sio.loadmat(test_datapath)


train_data = load_train['X']
image_size = train_data.shape[0]
num_channels = train_data.shape[2]
num_train = train_data.shape[3]
train_data = train_data.transpose((3, 0, 1, 2))
train_labels= np.reshape(load_train['y'], [num_train])

test_data = load_test['X']
num_test = test_data.shape[3]
test_data = test_data.transpose((3, 0, 1, 2))
test_labels = np.reshape(load_test['y'], [num_test])


def data_shuffle(data, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_data = data[permutation, :, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_data, shuffled_labels

train_data, train_labels = data_shuffle(train_data, train_labels)
test_data, test_labels = data_shuffle(test_data, test_labels)

def make_validation(data, labels, percent):
    breakpoint = int(labels.shape[0] * percent)
    train_data = data[0:breakpoint, :, :, :]
    train_labels = labels[0:breakpoint]
    valid_data = data[breakpoint:labels.shape[0], :, :, :]
    valid_labels = labels[breakpoint:labels.shape[0]]
    return train_data, train_labels, valid_data, valid_labels

train_data, train_labels, valid_data, valid_labels = make_validation(train_data, train_labels, 0.8)

pickle_file = 'data/number.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_data': train_data,
        'train_labels': train_labels,
        'valid_data': valid_data,
        'valid_labels': valid_labels,
        'test_data': test_data,
        'test_labels':test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

print('Compressed pickle size:', os.stat(pickle_file).st_size)

for i in range(20):
    plt.imshow(valid_data[i, :, :, :])
    print(valid_labels[i])
    plt.show()