#!/usr/bin/python3

import os;
from re import search;
import numpy as np;
import cv2;
import tensorflow as tf;

def segment_parse_function(serialized_example):

  feature = tf.io.parse_single_example(
    serialized_example,
    features = {
      'data': tf.io.FixedLenFeature((), dtype = tf.string),
      'joints': tf.io.FixedLenFeature((21 * 3,), dtype = tf.float32),
      'mask': tf.io.FixedLenFeature((480 * 640), dtype = tf.int64)
    }
  );
  data = tf.io.decode_jpeg(feature['data']);
  data = tf.cast(tf.reshape(data, (480, 640, 3)), dtype = tf.float32);
  data = tf.image.resize(data, (256, 256)) / 127.5 - 1;
  #joints = tf.reshape(feature['joints'], (21, 3));
  mask = tf.reshape(feature['mask'], (480, 640, 1));
  mask = tf.image.resize(mask, (256, 256));
  mask = tf.cast(mask, dtype = tf.int32);
  return data, mask;

def create_dataset(rootdir, with_object = False, filename = "synthesis.tfrecord"):

  background = (14,255,14);
  dirs = {True: ['male_noobject', 'male_object', 'female_noobject', 'female_object'], \
          False: ['male_noobject', 'female_noobject']};
  if not os.path.exists('datasets'): os.mkdir('datasets');
  writer = tf.io.TFRecordWriter(os.path.join('datasets', filename));
  count = 0;
  for dir in dirs[with_object]:
    assert os.path.exists(os.path.join(rootdir, dir));
    for seq in os.listdir(os.path.join(rootdir, dir)):
      for cam in os.listdir(os.path.join(rootdir, dir, seq)):
        for num in os.listdir(os.path.join(rootdir, dir, seq, cam)):
          for file in os.listdir(os.path.join(rootdir, dir, seq, cam, num)):
            result = search(r"^([0-9]+)_color\.png", file);
            if result is None: continue;
            imgpath = os.path.join(rootdir, dir, seq, cam, num, file);
            labelpath = os.path.join(rootdir, dir, seq, cam, num, result[1] + "_joint_pos.txt");
            img = cv2.imread(imgpath);
            if img is None:
              print("failed to open " + imgpath);
              continue;
            label = open(labelpath,'r');
            if label is None:
              print("failed to open " + labelpath);
              continue;
            joints = np.array(label.readlines()[0].strip().split(',')).astype('float32');
            if joints.shape[0] != 21 * 3:
              print("invalid joint coordinate number");
              continue;
            mask = np.logical_not(np.logical_and(np.logical_and(img[...,0] == 14, img[...,1] == 255),img[...,2] == 14)).astype('int8');
            trainsample = tf.train.Example(features = tf.train.Features(
              feature = {
                'data': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.encode_jpeg(img).numpy()])),
                'joints': tf.train.Feature(float_list = tf.train.FloatList(value = joints)),
                'mask': tf.train.Feature(int64_list = tf.train.Int64List(value = mask.reshape(-1)))
              }
            ));
            writer.write(trainsample.SerializeToString());
            count += 1;
  writer.close();
  print('written ' + str(count) + " samples to " + filename);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  create_dataset('/mnt/SynthHands_Release');
