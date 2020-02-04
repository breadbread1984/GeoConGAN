#!/usr/bin/python3

import tensorflow as tf;
from create_dataset import parse_function;
from models import SilNet;

batch_size = 8;
input_shape = (480,640,3);

def main():

  silnet = SilNet(input_shape);
  optimizer = tf.keras.optimizers.Adam(0.0)
