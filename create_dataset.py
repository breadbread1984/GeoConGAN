#!/usr/bin/python3

import os;
from re import match;
import tensorflow as tf;

def create_synthesis_segment_dataset(rootdir, with_object = False):

  background = (14,255,14);
  dirs = {True: ['male_noobject', 'male_object', 'female_noobject', 'female_object'], \
          False: ['male_noobject', 'female_noobject']};
  for dir in dirs[with_object]:
    assert os.path.exists(os.path.join(rootdir, dir));
    for seq in os.listdir(os.path.join(rootdir, dir)):
      for cam in os.listdir(os.path.join(rootdir, dir, seq)):
        for num in os.listdir(os.path.join(rootdir, dir, seq, cam)):
          for file in os.listdir(os.path.join(rootdir, dir, seq, cam, num)):
            filepath = os.path.join(rootdir, dir, seq, cam, num, file)
            if match(r"^[0-9]+_color\.png", file) is None: continue;
            print(filepath);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  create_synthesis_segment_dataset('/mnt/SynthHands_Release');
