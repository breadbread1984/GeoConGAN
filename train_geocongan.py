#!/usr/bin/python3

import os;
import numpy as np;
import tensorflow as tf;
from models import GeoConGAN;
from create_dataset import real_parse_function, synthetic_parse_function;

batch_size = 1;
dataset_size = 93476;
img_shape = (256,256,3);

def main():

  # models
  geocongan = GeoConGAN();
  optimizerGA = tf.keras.optimizers.Adam(
    tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries = [dataset_size * 100 + i * dataset_size * 100 / 4 for i in range(5)],
      values = list(reversed([i * 2e-4 / 5 for i in range(6)]))),
    beta_1 = 0.5);
  optimizerGB = tf.keras.optimizers.Adam(
    tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries = [dataset_size * 100 + i * dataset_size * 100 / 4 for i in range(5)],
      values = list(reversed([i * 2e-4 / 5 for i in range(6)]))),
    beta_1 = 0.5);
  optimizerDA = tf.keras.optimizers.Adam(
    tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries = [dataset_size * 100 + i * dataset_size * 100 / 4 for i in range(5)],
      values = list(reversed([i * 2e-4 / 5 for i in range(6)]))),
    beta_1 = 0.5);
  optimizerDB = tf.keras.optimizers.Adam(
    tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries = [dataset_size * 100 + i * dataset_size * 100 / 4 for i in range(5)],
      values = list(reversed([i * 2e-4 / 5 for i in range(6)]))),
    beta_1 = 0.5);
  
  # load dataset
  '''
  A = tf.data.TFRecordDataset(os.path.join('dataset', 'A.tfrecord')).map(parse_function_generator(img_shape)).shuffle(batch_size).batch(batch_size).__iter__();
  B = tf.data.TFRecordDataset(os.path.join('dataset', 'B.tfrecord')).map(parse_function_generator(img_shape)).shuffle(batch_size).batch(batch_size).__iter__();
  '''
  A = iter(tf.data.TFRecordDataset(os.path.join('datasets','real.tfrecord')).repeat(-1).map(real_parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE));
  B = iter(tf.data.TFRecordDataset(os.path.join('datasets','synthetic.tfrecord')).repeat(-1).map(synthetic_parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE));
  # restore from existing checkpoint
  checkpoint = tf.train.Checkpoint(GA = geocongan.GA, GB = geocongan.GB, DA = geocongan.DA, DB = geocongan.DB, 
                                   optimizerGA = optimizerGA, optimizerGB = optimizerGB, optimizerDA = optimizerDA, optimizerDB = optimizerDB);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  # create log
  log = tf.summary.create_file_writer('checkpoints');
  # train model
  avg_ga_loss = tf.keras.metrics.Mean(name = 'real2synth loss', dtype = tf.float32);
  avg_gb_loss = tf.keras.metrics.Mean(name = 'synth2real loss', dtype = tf.float32);
  avg_da_loss = tf.keras.metrics.Mean(name = 'synth disc loss', dtype = tf.float32);
  avg_db_loss = tf.keras.metrics.Mean(name = 'real disc loss', dtype = tf.float32);
  while True:
    imageA, maskA = next(A);
    imageB, maskB = next(B);
    with tf.GradientTape(persistent = True) as tape:
      outputs = geocongan((imageA, imageB, maskA, maskB));
      GA_loss = geocongan.GA_loss(outputs);
      GB_loss = geocongan.GB_loss(outputs);
      DA_loss = geocongan.DA_loss(outputs);
      DB_loss = geocongan.DB_loss(outputs);
    # calculate discriminator gradients
    da_grads = tape.gradient(DA_loss, geocongan.DA.trainable_variables); avg_da_loss.update_state(DA_loss);
    db_grads = tape.gradient(DB_loss, geocongan.DB.trainable_variables); avg_db_loss.update_state(DB_loss);
    # calculate generator gradients
    ga_grads = tape.gradient(GA_loss, geocongan.GA.trainable_variables); avg_ga_loss.update_state(GA_loss);
    gb_grads = tape.gradient(GB_loss, geocongan.GB.trainable_variables); avg_gb_loss.update_state(GB_loss);
    # update discriminator weights
    optimizerDA.apply_gradients(zip(da_grads, geocongan.DA.trainable_variables));
    optimizerDB.apply_gradients(zip(db_grads, geocongan.DB.trainable_variables));
    # update generator weights
    optimizerGA.apply_gradients(zip(ga_grads, geocongan.GA.trainable_variables));
    optimizerGB.apply_gradients(zip(gb_grads, geocongan.GB.trainable_variables));
    if tf.equal(optimizerGA.iterations % 500, 0):
      real_A = tf.cast(tf.clip_by_value((imageA + 1) * 127.5, clip_value_min = 0., clip_value_max = 255.), dtype = tf.uint8);
      real_B = tf.cast(tf.clip_by_value((imageB + 1) * 127.5, clip_value_min = 0., clip_value_max = 255.), dtype = tf.uint8);
      fake_B = tf.cast(tf.clip_by_value((outputs[1] + 1) * 127.5, clip_value_min = 0., clip_value_max = 255.), dtype = tf.uint8);
      fake_A = tf.cast(tf.clip_by_value((outputs[9] + 1) * 127.5, clip_value_min = 0., clip_value_max = 255.), dtype = tf.uint8);
      with log.as_default():
        tf.summary.scalar('real2synth loss', avg_ga_loss.result(), step = optimizerGA.iterations);
        tf.summary.scalar('synth2real loss', avg_gb_loss.result(), step = optimizerGB.iterations);
        tf.summary.scalar('synth discriminator loss', avg_da_loss.result(), step = optimizerDA.iterations);
        tf.summary.scalar('real discriminator loss', avg_db_loss.result(), step = optimizerDB.iterations);
        tf.summary.image('real', real_A, step = optimizerGA.iterations);
        tf.summary.image('real generated synth', fake_B, step = optimizerGA.iterations);
        tf.summary.image('synth', real_B, step = optimizerGA.iterations);
        tf.summary.image('synth generated real', fake_A, step = optimizerGA.iterations);
      print('Step #%d real2synth Loss: %.6f synth2real Loss: %.6f synth disc Loss: %.6f real disc Loss: %.6f lr: %.6f' % \
            (optimizerGA.iterations, avg_ga_loss.result(), avg_gb_loss.result(), avg_da_loss.result(), avg_db_loss.result(), \
            optimizerGA._hyper['learning_rate'](optimizerGA.iterations)));
      avg_ga_loss.reset_states();
      avg_gb_loss.reset_states();
      avg_da_loss.reset_states();
      avg_db_loss.reset_states();
    if tf.equal(optimizerGA.iterations % 10000, 0):
      # save model
      checkpoint.save(os.path.join('checkpoints', 'ckpt'));
    if GA_loss < 0.01 and GB_loss < 0.01 and DA_loss < 0.01 and DB_loss < 0.01: break;
  # save the network structure with weights
  if False == os.path.exists('models'): os.mkdir('models');
  geocongan.GA.save(os.path.join('models', 'real2synth.h5'));
  geocongan.GB.save(os.path.join('models', 'synth2real.h5'));
  geocongan.DA.save(os.path.join('models', 'synthD.h5'));
  geocongan.DB.save(os.path.join('models', 'realD.h5'));

if __name__ == "__main__":
    
  assert True == tf.executing_eagerly();
  main();
