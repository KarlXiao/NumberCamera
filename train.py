"""
train cnn model for number recognition
"""
import math
import time
import numpy as np
from six.moves import cPickle as pickle
import tensorflow as tf
from SVHNmodel import Model

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

image_size = 54
num_channels = 3
num_labels = 11


def label_reformat(labels):
    new_labels = np.ndarray([labels.shape[0], labels.shape[1], num_labels])
    labels = np.reshape(labels, [labels.shape[0], labels.shape[1], 1])
    for i in range(labels.shape[1]):
        new_labels[:, i, :] = np.reshape((np.arange(num_labels) == labels[:, i, None]).astype(
            np.float), [labels.shape[0], num_labels])
    return new_labels


train_labels = label_reformat(train_labels)
valid_labels = label_reformat(valid_labels)
test_labels = label_reformat(test_labels)

print('Training dataset:', train_data.shape, train_labels.shape)
print('Validation dataset:', valid_data.shape, valid_labels.shape)
print('Test dataset:', test_data.shape, test_labels.shape)

with tf.Graph().as_default():
    # input
    tf_data = tf.placeholder(
        tf.float32, shape=[None, image_size, image_size, num_channels])
    tf_digits = tf.placeholder(tf.float32, shape=[None, num_labels])
    tf_length = tf.placeholder(tf.float32, shape=[None, 1])
    dropout = tf.placeholder(tf.float32)

    digits_length, digits = Model.forward(tf_data, dropout)
    loss = Model.loss(digits_length, digits, tf_length, tf_digits)

    # optimizer
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(
        0.01, global_step, 100, 0.9, staircase=True)

    optimizer = tf. train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)

    tf.summary.image('image', tf_data)
    tf.summary.scalar('Training loss', loss)
    tf.summary.scalar('Learning rate', learning_rate)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs')
    saver = tf.train.Saver()

num_steps = 20000
batch_size = 256

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    writer.add_graph(graph=session.graph)
    print('Initialized')
    start_time = time.time()
    for step in range(num_steps):
        train_offset = (step * batch_size) % (train_data.shape[0] - batch_size)
        batch_data = train_data[train_offset:(train_offset + batch_size), :, :, :] - train_mean
        batch_label = train_labels[train_offset:(train_offset + batch_size), :, :]
        feed_dict = {tf_data: batch_data,tf_digits: batch_label, dropout: 0.2}
        _, l, summary = session.run([optimizer, loss, merged], feed_dict=feed_dict)
        writer.add_summary(summary, step)
        if step % 1000 == 0:
            print('Batch loss at step %d: %f' % (step, l))
    print('Training costs %f seconds.' % (time.time() - start_time))
    saver.save(session, './models/cnn_numbercamera.tfmodel')

with tf.Session() as session:
    saver.restore(session, './models/cnn_numbercamera.tfmodel')
    test_accuracy = 0
    for i in range(math.ceil(test_data.shape[0] / batch_size)):
        test_offset = (i * batch_size) % (test_data.shape[0] - batch_size)
        feed_dict = {tf_data: test_data[test_offset:(test_offset + batch_size), :, :, :] - train_mean,
                     tf_digits: test_labels[test_offset:(test_offset + batch_size), :, :], dropout: .2}
        print('Test accuray: %f%%' % (100 * test_accuracy /
                                      math.floor(test_data.shape[0] / batch_size)))
