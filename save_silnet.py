#!/usr/bin/python3

import tensorflow as tf;
from models import SilNet;

input_shape = (256,256,3);

def main():

  silnet = SilNet(input_shape);
  optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3, beta_1 = 0.5);
  checkpoint = tf.train.Checkpoint(model = silnet, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  silnet.save('silnet.h5');

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();
