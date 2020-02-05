#!/usr/bin/python3

import os;
import tensorflow as tf;
from create_dataset import segment_parse_function;
from models import SilNet;

batch_size = 8;
input_shape = (256,256,3);

def main():

  silnet = SilNet(input_shape);
  optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3, beta_1 = 0.5);
  trainset = iter(tf.data.TFRecordDataset(os.path.join('datasets','synthesis.tfrecord')).repeat(-1).map(segment_parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE));
  checkpoint = tf.train.Checkpoint(model = silnet, optimizer = optimizer);
  log = tf.summary.create_file_writer('checkpoints');
  avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
  while True:
    data, mask = next(trainset);
    with tf.GradientTape() as tape:
      output = silnet(data);
      loss = tf.keras.losses.SparseCategoricalCrossentropy()(mask, output);
    grads = tape.gradients(loss, silnet.trainable_variables);
    avg_loss.update_state(loss);
    optimizer.apply_gradients(zip(grads, silnet.trainable_variables));
    if tf.equal(optimizer.iterations % 100, 0):
      with log.as_default():
        tf.summary.scalar('loss', avg_loss.result(), step = optimizer.iterations);
      print('Step #%d loss: %.6f' % (optimizer.iterations, avg_loss.result()));
      if avg_loss.result() < 0.01: break;
      avg_loss.reset_states();
    if tf.equal(optimizer.iterations % 1000, 0):
      checkpoint.save(os.path.join('checkpoint', 'ckpt'));
  silnet.save('silnet.h5');
  
if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();
