#!/usr/bin/python3

import cv2;
import tensorflow as tf;
from create_dataset import ganerated_parse_function;
from models import RegNet;

batch_size = 16;
input_shape = (256,256,3);
ptpairs = [(0,1),(1,2),(2,3),(3,4), \
             (0,5),(5,6),(6,7),(7,8), \
             (0,9),(9,10),(10,11),(11,12), \
             (0,13),(13,14),(14,15),(15,16), \
             (0,17),(17,18),(18,19),(19,20)];

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
      image = tf.clip_by_value(data[0,...] * 255., clip_value_min = 0., clip_value_max = 255.);
      image = tf.cast(image, dtype = tf.uint8); # (height, width, 3)
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR);
      heatmaps = final2d[0,...];
      heatmaps = tf.transpose(heatmaps, (2, 0, 1)).numpy(); # heatmaps.shape = (21, height, width)
      pts = list();
      for heatmap in heatmaps:
        pt = np.array(np.unravel_index(heatmap.argmax(), heatmap.shape)).astype('float32');
        pt = tuple(reversed(pt * 8)); # in sequence of x,y
        pts.append(pt);
      for pt in pts:
        cv2.circle(image, pt, 3, (0,255,0), -1);
      for pair in ptpairs:
        cv2.line(image, pts[pair[0]], pts[pair[1]], (0,0,255), 2);
      image = np.expand_dims(image, axis = 0);
      with log.as_default():
        tf.summary.scalar('loss', avg_loss.result(), step = optimizer.iterations);
        tf.summary.image('2d key points', image, step = optimizer.iterations);
      print('Step #%d loss: %.6f' % (optimizer.iterations, avg_loss.result()));
      if avg_loss.result() < 1e-5: break;
      avg_loss.reset_states();
    if tf.equal(optimizer.iterations % 1000, 0):
      checkpoint.save(os.path.join('checkpoints', 'ckpt'));
  regnet.save('regnet.h5');

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();
  
