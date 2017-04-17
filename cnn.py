"""
train cnn model for number recognition
"""
import math
import time
import numpy as np
from six.moves import cPickle as pickle
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
num_labels = 11


def label_reformat(labels):
    new_labels = np.ndarray([labels.shape[0], labels.shape[1], num_labels])
    labels = np.reshape(labels, [labels.shape[0], labels.shape[1], 1])
    for i in range(labels.shape[1]):
        new_labels[:, i, :] = np.reshape((np.arange(num_labels) == labels[:, i, None]).astype(np.float), [labels.shape[0], num_labels])
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
    tf_data = tf.placeholder(
        tf.float32, shape=[None, image_size, image_size, num_channels])
    tf_labels = tf.placeholder(tf.float32, shape=[None, num_labels])
    keep_prob = tf.placeholder(tf.float32)

    def batch_norm(data, name):
        mean, var = tf.nn.moments(data, axes=[0])
        return tf.nn.batch_normalization(data, mean, var, None, None, 0.001)

    def conv2d(data, weight, bias, stride, padding, name=None):
        conv_out = tf.nn.conv2d(
            data, weight, [1, stride, stride, 1], padding=padding, name=name)
        add_bias = tf.nn.bias_add(conv_out, bias)
        return tf.nn.relu(add_bias)

    # dnn_model
    def dnn_model(data, weight, bias, dropout=1):
        conv1 = conv2d(data, weight['w1'], bias['b1'], 1, 'VALID', 'conv1')
        pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        norm1 = batch_norm(pool1, 'norm1')

        conv2 = conv2d(norm1, weight['w2'], bias['b2'], 1, 'VALID', 'conv2')
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        norm2 = batch_norm(pool2, 'norm2')

        shape = norm2.get_shape().as_list()
        reshape = tf.reshape(norm2, [-1, shape[1] * shape[2] * shape[3]])
        out = tf.matmul(reshape, weight['w3']) + bias['b3']
        return out

    weight = {
        'w1': tf.Variable(tf.random_normal([5, 5, num_channels, 20])),
        'w2': tf.Variable(tf.random_normal([5, 5, 20, 50])),
        'w3': tf.Variable(tf.random_normal([50 * 29 * 29, 4096])),
    }

    bias = {
        'b1': tf.Variable(tf.random_normal([20])),
        'b2': tf.Variable(tf.random_normal([50])),
        'b3': tf.Variable(tf.random_normal([4096])),
    }

    logits_dnn = dnn_model(tf_data, weight, bias, keep_prob)

    # softmax design
    weight_softmax = {
        'l': tf.Variable(tf.random_normal([4096, 7])),
        's1': tf.Variable(tf.random_normal([4096, 11])),
        's2': tf.Variable(tf.random_normal([4096, 11])),
        's3': tf.Variable(tf.random_normal([4096, 11])),
        's4': tf.Variable(tf.random_normal([4096, 11])),
        's5': tf.Variable(tf.random_normal([4096, 11]))
    }

    bias_softmax = {
        'l': tf.Variable(tf.random_normal([7])),
        's1': tf.Variable(tf.random_normal([11])),
        's2': tf.Variable(tf.random_normal([11])),
        's3': tf.Variable(tf.random_normal([11])),
        's4': tf.Variable(tf.random_normal([11])),
        's5': tf.Variable(tf.random_normal([11])),
    }

    def softmax_loss(logits_dnn, weight, bias, labels):
        logits = tf.matmul(logits_dnn, weight) + bias
        return logits, tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    # loss
    logits_l, loss_l = softmax_loss(logits_dnn, weight_softmax['l'], bias_softmax['l'], tf_labels[:, 1, :])
    logits_s1, loss_s1 = softmax_loss(logits_dnn, weight_softmax['s1'], bias_softmax['s1'], tf_labels[:, 2, :])
    logits_s2, loss_s2 = softmax_loss(logits_dnn, weight_softmax['s2'], bias_softmax['s2'], tf_labels[:, 3, :])
    logits_s3, loss_s3 = softmax_loss(logits_dnn, weight_softmax['s3'], bias_softmax['s3'], tf_labels[:, 4, :])
    logits_s4, loss_s4 = softmax_loss(logits_dnn, weight_softmax['s4'], bias_softmax['s4'], tf_labels[:, 5, :])
    logits_s5, loss_s5 = softmax_loss(logits_dnn, weight_softmax['s5'], bias_softmax['s5'], tf_labels[:, 6, :])
    loss = loss_l + loss_s1 + loss_s2 + loss_s3 + loss_s4 + loss_s5

    tf.summary.scalar('Training loss', loss)
    tf.summary.scalar('Softmax L loss', loss_l)
    tf.summary.scalar('Softmax S1 loss', loss_s1)
    tf.summary.scalar('Softmax S2 loss', loss_s2)
    tf.summary.scalar('Softmax S3 loss', loss_s3)
    tf.summary.scalar('Softmax S4 loss', loss_s4)
    tf.summary.scalar('Softmax S5 loss', loss_s5)

    # optimizer
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.01, global_step, 100, 0.9, staircase=True)
    tf.summary.scalar('Learning rate', learning_rate)
    optimizer = tf. train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)

    # accuracy
    def log_pro(logits):
        return tf.log(tf.nn.softmax(logits))

    def log_index(logits, axis):
        log = tf.log(tf.nn.softmax(logits))
        return log, tf.argmax(log, axis)

    l = log_pro(logits_l)
    ind1, s1 = log_index(logits_s1, 1)
    ind2, s2 = log_index(logits_s2, 1)
    ind3, s3 = log_index(logits_s3, 1)
    ind4, s4 = log_index(logits_s4, 1)
    ind5, s5 = log_index(logits_s5, 1)

    correct_pre = tf.equal(tf.argmax(logits, axis=1),
                           tf.argmax(tf_labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))
    tf.summary.scalar('Training accuracy', accuracy)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs')
    saver = tf.train.Saver()

num_steps = 20000
batch_size = 256

with tf.Session(graph=graph) as session:
    session.run(tf.global_variables_initializer())
    writer.add_graph(graph=session.graph)
    print('Initialized')
    start_time = time.time()
    for step in range(num_steps):
        train_offset = (step * batch_size) % (train_data.shape[0] - batch_size)
        batch_data = train_data[train_offset:(
            train_offset + batch_size), :, :, :] - train_mean
        batch_label = train_labels[train_offset:(train_offset + batch_size), :, :]
        feed_dict = {tf_data: batch_data,
                     tf_labels: batch_label, keep_prob: 1.}
        _, l, train_accuracy, summary = session.run(
            [optimizer, loss, accuracy, merged], feed_dict=feed_dict)
        writer.add_summary(summary, step)
        if step % 1000 == 0:
            valid_accuracy = 0
            for i in range(math.ceil(valid_data.shape[0] / batch_size)):
                valid_offset = (
                    i * batch_size) % (valid_data.shape[0] - batch_size)
                feed_dict = {tf_data: valid_data[valid_offset:(valid_offset + batch_size), :, :, :] - valid_mean,
                             tf_labels: valid_labels[valid_offset:(valid_offset + batch_size), :, :], keep_prob: 1.}
                step_accuracy = session.run(accuracy, feed_dict=feed_dict)
                valid_accuracy = valid_accuracy + step_accuracy
            print('Batch loss at step %d: %f' % (step, l))
            print('Training Batch accuray: %f%%' % (train_accuracy * 100))
            print('Valid accuracy: %f%%' % (100 * valid_accuracy /
                                            math.floor(valid_data.shape[0] / batch_size)))
    print('Training costs %f seconds.' % (time.time() - start_time))
    saver.save(session, './models/cnn_numbercamera.tfmodel')

with tf.Session(graph=graph) as session:
    saver.restore(session, './models/cnn_numbercamera.tfmodel')
    test_accuracy = 0
    for i in range(math.ceil(test_data.shape[0] / batch_size)):
        test_offset = (i * batch_size) % (test_data.shape[0] - batch_size)
        feed_dict = {tf_data: test_data[test_offset:(test_offset + batch_size), :, :, :] - train_mean,
                     tf_labels: test_labels[test_offset:(test_offset + batch_size), :, :], keep_prob: 1.}
        step_accuracy = session.run(accuracy, feed_dict=feed_dict)
        test_accuracy = test_accuracy + step_accuracy
        print('Test accuray: %f%%' % (100 * test_accuracy /
                                  math.floor(test_data.shape[0] / batch_size)))
