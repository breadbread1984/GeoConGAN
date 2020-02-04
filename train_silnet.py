#!/usr/bin/python3

import os;
import tensorflow as tf;
from create_dataset import parse_function;
from models import SilNet;

batch_size = 8;
input_shape = (256,256,3);

def main():

  silnet = SilNet(input_shape);
  @tf.function
  def loss(outputs, labels):
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)(labels, outputs);
  silnet.compile(optimizer = tf.keras.optimizers.Adam(0.001), loss = loss);
  # load dataset
  trainset = 
  # train
  silnet.fit(trainset, epochs = 20);
  if not os.path.exists('models'): os.mkdir('models');
  silnet.save(os.path.join('models', 'silnet.h5'));
  
if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();
