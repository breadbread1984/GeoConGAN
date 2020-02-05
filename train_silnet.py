#!/usr/bin/python3

import os;
import tensorflow as tf;
from create_dataset import synthetic_parse_function, real_parse_function;
from models import SilNet;

batch_size = 8;
input_shape = (256,256,3);
dataset_size = 93476;

def main(is_synth = True):

  silnet = SilNet(input_shape);
  optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3, beta_1 = 0.5);
  if is_synth:
    trainset = iter(tf.data.TFRecordDataset(os.path.join('datasets','synthetic.tfrecord')).repeat(-1).map(synthetic_parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE));
  else:
    trainset = iter(tf.data.TFRecordDataset(os.path.join('datasets','real.tfrecord')).repeat(-1).map(real_parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE))
  checkpoint = tf.train.Checkpoint(model = silnet, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  log = tf.summary.create_file_writer('checkpoints');
  avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
  while True:
    data, mask = next(trainset);
    with tf.GradientTape() as tape:
      output = silnet(data);
      loss = tf.keras.losses.SparseCategoricalCrossentropy()(mask, output);
    grads = tape.gradient(loss, silnet.trainable_variables);
    avg_loss.update_state(loss);
    optimizer.apply_gradients(zip(grads, silnet.trainable_variables));
    if tf.equal(optimizer.iterations % 100, 0):
      image = tf.clip_by_value((data + 1.) * 127.5, clip_value_min = 0., clip_value_max = 255.);
      image = tf.cast(image, dtype = tf.uint8);
      output = tf.clip_by_value(output[..., 0:1] * 255., clip_value_min = 0., clip_value_max = 255.);
      output = tf.tile(tf.cast(output, dtype = tf.uint8), (1, 1, 1, 3));
      visualize = tf.concat([image[0:1,...], output[0:1,...]], axis = 2);
      with log.as_default():
        tf.summary.scalar('loss', avg_loss.result(), step = optimizer.iterations);
        tf.summary.image('segmentation', visualize, step = optimizer.iterations);
      print('Step #%d loss: %.6f' % (optimizer.iterations, avg_loss.result()));
      if avg_loss.result() < 1e-5: break;
      avg_loss.reset_states();
    if tf.equal(optimizer.iterations % 1000, 0):
      checkpoint.save(os.path.join('checkpoints', 'ckpt'));
  silnet.save('silnet.h5');

if __name__ == "__main__":

  assert tf.executing_eagerly();
  import sys;
  if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " [(synthetic|real)]");
    exit(0);
  if sys.argv.strip() not in ['synthetic', 'real']:
    print('training mode must be synthetic or real');
    exit(1);
  main(True if sys.argv.strip() == "synthetic" else False);
