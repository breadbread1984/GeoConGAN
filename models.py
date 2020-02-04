#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_addons as tfa;

def SilNet(input_shape):

  inputs = tf.keras.Input(input_shape[-3:]);
  results = inputs;
  # downsampling (64-128-256)
  layers = list();
  for i in range(3):
    channels = 64 * 2 ** i;
    results = tf.keras.layers.Conv2D(filters = channels, kernel_size = (3,3), padding = 'same')(results);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tfa.layers.InstanceNormalization()(results);
    results = tf.keras.layers.Conv2D(filters = channels, kernel_size = (3,3), padding = 'same')(results);
    layers.append(results);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tfa.layers.InstanceNormalization()(results);
    results = tf.keras.layers.MaxPooling2D()(results);
  # upsampling (512-256-128)
  for i in range(3):
    channels = 64 * 2 ** (3-i);
    results = tf.keras.layers.Conv2D(filters = channels, kernel_size = (3,3), padding = 'same')(results);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tfa.layers.InstanceNormalization()(results);
    results = tf.keras.layers.Conv2D(filters = channels, kernel_size = (3,3), padding = 'same')(results);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2DTranspose(filters = channels // 2, kernel_size = (2,2), strides = (2,2), padding = 'same')(results);
    results = tf.keras.layers.Concatenate(axis = -1)([results, layers[2-i]]);
  # output
  results = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'same')(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tfa.layers.InstanceNormalization()(results);
  results = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'same')(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tf.keras.layers.Conv2D(filters = 1, kernel_size = (1,1), padding = 'same', activation = tf.math.tanh)(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  silnet = SilNet((256,256,3));
  tf.keras.utils.plot_model(model = silnet, to_file = 'silnet.png', show_shapes = True, dpi = 64);
