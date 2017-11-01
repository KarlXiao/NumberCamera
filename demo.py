import tensorflow as tf
from donkey import Donkey
from SVHNmodel import Model
from meta import Meta
from pylab import *

path_to_eval_tfrecords_file = 'data/test.tfrecords'
batch_size = 16

meta = Meta()
meta.load('data/meta.json')

image_batch, length_batch, digits_batch = Donkey.build_batch(path_to_eval_tfrecords_file, batch_size=batch_size, num_examples=meta.num_test_examples, shuffled=False)
length_logits, digits_logits = Model.forward(image_batch, dropout=1.0)
length_predictions = tf.argmax(length_logits, axis=1)
digits_predictions = tf.argmax(digits_logits, axis=2)
digits_batch_string = tf.reduce_join(tf.as_string(digits_batch), axis=1)
digits_predictions_string = tf.reduce_join(tf.as_string(digits_predictions), axis=1)

sess = tf.InteractiveSession()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

restorer = tf.train.Saver()
checkpoint_path = tf.train.latest_checkpoint('logs/train')
restorer.restore(sess, checkpoint_path)

length_predictions_val, digits_predictions_string_val, image_batch_val = sess.run([length_predictions, digits_predictions_string, image_batch])
image_batch_val = (image_batch_val / 2.0) + 0.5

idx = 12
image_val = image_batch_val[idx]
length_prediction_val = length_predictions_val[idx]
digits_prediction_string_val = digits_predictions_string_val[idx]

print ('length: %d' % length_prediction_val)
print ('digits: %s' % digits_prediction_string_val[0:length_prediction_val])
imshow(image_val)
show()

coord.request_stop()
coord.join(threads)
sess.close()