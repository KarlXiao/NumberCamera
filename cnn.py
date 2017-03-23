"""
train cnn model for number recognition
"""
import numpy as np
from six.moves import cPickle as pickle
import time
import tensorflow as tf

try:
    with open('data/number.pickle', 'rb') as f:
        pickle_file = pickle.load(f)
        train_data = pickle_file['train_data']
        train_labels = pickle_file['train_labels']
        train_mean = pickle_file['train_mean']
        valid_data = pickle_file['valid_data']
        valid_labels = pickle_file['valid_labels']
        valid_mean = pickle_file['valid_mean']
        test_data = pickle_file['test_data']
        test_labels = pickle_file['test_labels']
        del pickle_file
except Exception as e:
    print('Unable to read data:', pickle_file, ':', e)
    raise

print('Training dataset:', train_data.shape, train_labels.shape)
print('Validation dataset:', valid_data.shape, valid_labels.shape)
print('Test dataset:', test_data.shape, test_labels.shape)

image_size = 32
num_channels = 3
num_labels = 10

def label_reformat(labels):
    new_labels = (np.arange(num_labels) == labels[:, None]).astype(np.float)
    return new_labels

train_labels = label_reformat(train_labels)
valid_labels = label_reformat(valid_labels)
test_labels = label_reformat(test_labels)

print('Training dataset:', train_data.shape, train_labels.shape)
print('Validation dataset:', valid_data.shape, valid_labels.shape)
print('Test dataset:', test_data.shape, test_labels.shape)

graph = tf.Graph()
with graph.as_default():

    # input
    tf_data = tf.placeholder(tf.float32, shape=[None, image_size, image_size, num_channels])
    tf_labels = tf.placeholder(tf.float32, shape=[None, num_labels])
    keep_prob = tf.placeholder(tf.float32)

    def batch_norm(data, name):
        mean, var = tf.nn.moments(data, axes=[0])
        return tf.nn.batch_normalization(data, mean, var, None, None, 0.001)

    def conv2d(data, weight, bias, stride, padding, name=None):
        conv_out = tf.nn.conv2d(data, weight, [1, stride, stride, 1], padding=padding, name=name)
        add_bias = tf.nn.bias_add(conv_out, bias)
        return tf.nn.relu(add_bias)
        
    # model
    def model(data, weight, bias, dropout=1):
        conv1 = conv2d(data, weight['w1'], bias['b1'], 5, 1, 'VALID', 'conv1')
        pool1 = tf.nn.max_pool(conv1, 2, 2, 'VALID')

        conv2 = conv2d(pool1, weight['w2'], bias['b2'], 1, 'VALID', 'conv2')
        pool2 = tf.nn.max_pool(conv2, 2, 2, 'VALID')

        shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [shape[0], shape[1]*shape[2]*shape[3]])
        fc1 = tf.nn.relu(tf.matmul(pool2, weight['w3'])+bias['b3'])
        out = tf.matmul(fc1, weight['w4'])+bias['b4']
        return out

    # parameter
    weight = {
        'w1': tf.Variable(tf.random_normal([5, 5, num_channels, 6])),
        'w2': tf.Variable(tf.random_normal([5, 5, 6, 16])),
        'w3': tf.Variable(tf.random_normal([16*5*5, 120])),
        'w4': tf.Variable(tf.random_normal([120, 84]))
    }

    bias = {
        'b1': tf.Variable(tf.random_normal([6])),
        'b2': tf.Variable(tf.random_normal([16])),
        'b3': tf.Varibale(tf.random_normal([120])),
        'b4': tf.Variable(tf.random_normal([84]))
    }

    logits = model(tf_data, weight, bias)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_labels))

    # optimizer
    global_setp = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.1, global_setp, 100, 0.95, staircase=True)
    tf.summary.scale('Learning rate', learning_rate)
    optimizer = tf. train.GradientDescentOptimizer(learning_rate).minimize(loss, global_setp=global_setp)

    correct_pre = tf.