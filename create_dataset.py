#!/usr/bin/python3

import os;
from re import match;
import cv2;
import tensorflow as tf;

def parse_function(serialized_example):

  feature = tf.io.parse_single_example(
    serialized_example,
    features = {
      'data': tf.io.FixedLenFeature((), dtype = tf.string),
      'joints': tf.io.FixedLenFeature((21 * 3,), dtype = tf.float32),
      'mask': tf.io.FixedLenFeature((480 * 640), dtype = tf.int64)
    }
  );
  data = tf.io.decode_raw(feature['data']);
  data = tf.reshape(data, (480, 640, 3));
  joints = tf.reshape(feature['joints'], (21, 3));
  mask = tf.reshape(feature['mask'], (480, 640));
  mask = tf.cast(mask, dtype = tf.int32);
  return data, (joints, mask);

def create_synthesis_segment_dataset(rootdir, with_object = False, filename = "synthesis.tfrecord"):

  background = (14,255,14);
  dirs = {True: ['male_noobject', 'male_object', 'female_noobject', 'female_object'], \
          False: ['male_noobject', 'female_noobject']};
  if os.path.exists(os.path.exists('datasets')): os.path.mkdir('datasets');
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
            mask = np.logical_not(np.logical_and(np.logical_and(img[...,0] == 14, img[...,1] == 255),img[...,2] == 14)).astype('int64');
            trainsample = tf.train.Example(features = tf.train.Features(
              feature = {
                'data': tf.train.Feature(bytes_list = tf.train.BytesList(value = [img.tobytes()])),
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
  create_synthesis_segment_dataset('/mnt/SynthHands_Release');
