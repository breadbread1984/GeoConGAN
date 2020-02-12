#!/usr/bin/python3

import tensorflow as tf;
from create_dataset import ganerated_parse_function;
from models import RegNet;

batch_size = 16;
input_shape = (256,256,3);

def main():

  regnet = RegNet(input_shape);
  optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3, beta_1 = 0.5);
  trainset = iter(tf.data.TFRecordDataset(os.path.join('datasets','ganerated.tfrecord')).repeat(-1).map(ganerated_parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE));
  checkpoint = tf.train.Checkpoint(model = regnet, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  log = tf.summary.create_file_writer('checkpoints');
  avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
  while True:
    data, (pos3d, heatmap) = next(trainset);
    with tf.GradientTape() as tape:
      inter3d, final2d, final3d = regnet(data);
      inter3d_loss = tf.keras.losses.MeanSquaredError()(pos3d, inter3d);
      final2d_loss = tf.keras.losses.MeanSquaredError()(heatmap, final2d);
      final3d_loss = tf.keras.losses.MeanSquaredError()(pos3d, final3d);
      loss = inter3d_loss + final2d_loss + final3d_loss;
    grads = tape.gradient(loss, regnet.trainable_variables);
    avg_loss.update_state(loss);
    optimizer.apply_gradients(zip(grads, regnet.trainable_variables));
    if tf.equal(optimizer.iterations % 100, 0):
      image = tf.clip_by_value(data * 255., clip_value_min = 0., clip_value_max = 255.);
      image = tf.cast(image, dtype = tf.uint8);
      #TODO:

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();
  
