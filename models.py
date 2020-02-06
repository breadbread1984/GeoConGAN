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
  results = tf.keras.layers.Conv2D(filters = 2, kernel_size = (1,1), padding = 'same', activation = tf.keras.layers.Softmax())(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def Generator(input_filters = 3, output_filters = 3, inner_filters = 64, blocks = 9):

  # input
  inputs = tf.keras.Input((None, None, input_filters));
  results = tf.keras.layers.Conv2D(filters = inner_filters, kernel_size = (7,7), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(inputs);
  results = tfa.layers.InstanceNormalization(axis = -1)(results);
  results = tf.keras.layers.ReLU()(results);
  # downsampling
  # 128-256
  for i in range(2):
    m = 2**(i + 1);
    results = tf.keras.layers.Conv2D(filters = inner_filters * m, kernel_size = (3,3), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(results);
    results = tfa.layers.InstanceNormalization(axis = -1)(results);
    results = tf.keras.layers.ReLU()(results);
  # resnet blocks
  # 256
  for i in range(blocks):
    short_circuit = results;
    results = tf.keras.layers.Conv2D(filters = inner_filters * 4, kernel_size = (3,3), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(results);
    results = tfa.layers.InstanceNormalization(axis = -1)(results);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(filters = inner_filters * 4, kernel_size = (3,3), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(results);
    results = tfa.layers.InstanceNormalization(axis = -1)(results);
    results = tf.keras.layers.Concatenate()([short_circuit, results]);
  # upsampling
  # 128-64
  for i in range(2):
    m = 2**(1 - i);
    results = tf.keras.layers.Conv2DTranspose(filters = inner_filters * m, kernel_size = (3,3), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(results);
    results = tfa.layers.InstanceNormalization(axis = -1)(results);
    results = tf.keras.layers.ReLU()(results);
  # output
  results = tf.keras.layers.Conv2D(filters = output_filters, kernel_size = (7,7), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(results);
  results = tfa.layers.InstanceNormalization(axis = -1)(results);
  results = tf.keras.layers.Lambda(lambda x: tf.math.tanh(x))(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def Discriminator(input_filters, inner_filters, layers = 3):

  inputs = tf.keras.Input((None, None, input_filters));
  results = tf.keras.layers.Conv2D(filters = inner_filters, kernel_size = (4,4), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(inputs);
  results = tf.keras.layers.LeakyReLU(0.2)(results);
  # 128-256-512
  for i in range(layers):
    m = min(2 ** (i + 1), 8);
    results = tf.keras.layers.Conv2D(filters = inner_filters * m, kernel_size = (4,4), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(results);
    results = tfa.layers.InstanceNormalization(axis = -1)(results);
    results = tf.keras.layers.LeakyReLU(0.2)(results);
  m = min(2 ** layers, 8); # 512
  results = tf.keras.layers.Conv2D(filters = inner_filters * m, kernel_size = (4,4), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(results);
  results = tfa.layers.InstanceNormalization(axis = -1)(results);
  results = tf.keras.layers.LeakyReLU(0.2)(results);
  results = tf.keras.layers.Conv2D(filters = 1, kernel_size = (4,4), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

class ImgPool(object):

  def __init__(self, size = 50):

    self.pool = list();
    self.size = size;

  def pick(self, image):

    if len(self.pool) < self.size:
      self.pool.append(image);
      return image;
    elif np.random.uniform() < 0.5:
      return image;
    else:
      index = np.random.randint(low = 0, high = self.size);
      retval = self.pool[index];
      self.pool[index] = image;
      return retval;

class GeoConGAN(tf.keras.Model):

  def __init__(self, input_filters = 3, output_filters = 3, inner_filters = 64, blocks = 9, layers = 3, ** kwargs):

    super(GeoConGAN, self).__init__(**kwargs);
    self.GA = Generator(input_filters = input_filters, output_filters = output_filters, inner_filters = inner_filters, blocks = blocks);
    self.GB = Generator(input_filters = output_filters, output_filters = input_filters, inner_filters = inner_filters, blocks = blocks);
    self.DA = Discriminator(input_filters = output_filters, inner_filters = inner_filters, layers = layers);
    self.DB = Discriminator(input_filters = input_filters,  inner_filters = inner_filters, layers = layers);
    self.pool_A = ImgPool(50);
    self.pool_B = ImgPool(50);
    self.l1 = tf.keras.losses.MeanAbsoluteError();
    self.l2 = tf.keras.losses.MeanSquaredError();
    self.real_seg = tf.keras.models.load_model('models/real_silnet.h5', compile = False, custom_objects = {'tf': tf, 'Softmax': tf.keras.layers.Softmax});
    self.synth_seg = tf.keras.models.load_model('models/synthetic_silnet.h5', compile = False, custom_objects = {'tf': tf, 'Softmax': tf.keras.layers.Softmax});
    self.real_seg.trainable = False;
    self.synth_seg.trainable = False;
    self.cce = tf.keras.losses.SparseCategoricalCrossentropy();

  def call(self, inputs):

    real_A = inputs[0]; # real
    real_B = inputs[1]; # synth
    mask_A = inputs[2]; # real mask
    mask_B = inputs[3]; # synth mask
    # real_A => GA => fake_B   fake_B => DA => pred_fake_B
    # real_B => GA => idt_B    real_B => DA => pred_real_B
    # fake_B => GB => rec_A    fake_B => synth_seg => pred_mask_B
    # real_B => GB => fake_A   fake_A => DB => pred_fake_A
    # real_A => GB => idt_A    real_A => DB => pred_real_A
    # fake_A => GA => rec_B    fake_A => real_seg => pred_mask_A
    fake_B = self.GA(real_A);
    idt_B = self.GA(real_B);
    pred_fake_B = self.DA(fake_B);
    pred_real_B = self.DA(real_B);
    rec_A = self.GB(fake_B);
    pred_mask_B = self.synth_seg(fake_B);
    fake_A = self.GB(real_B);
    idt_A = self.GB(real_A);
    pred_fake_A = self.DB(fake_A);
    pred_real_A = self.DB(real_A);
    rec_B = self.GA(fake_A);
    pred_mask_A = self.real_seg(fake_A);
    
    return (real_A, fake_B, idt_B, pred_fake_B, pred_real_B, rec_A, pred_mask_B, mask_A, real_B, fake_A, idt_A, pred_fake_A, pred_real_A, rec_B, pred_mask_A, mask_B);

  def GA_loss(self, inputs):
    
    (real_A, fake_B, idt_B, pred_fake_B, pred_real_B, rec_A, pred_mask_B, mask_A, real_B, fake_A, idt_A, pred_fake_A, pred_real_A, rec_B, pred_mask_A, mask_B) = inputs;
    # wgan gradient penalty
    loss_adv_A = self.l2(tf.ones_like(pred_fake_B), pred_fake_B);
    # generated image should not deviate too much from origin image
    loss_idt_A = self.l1(real_A, idt_A);
    # reconstruction loss
    loss_cycle_A = self.l1(real_A, rec_A);
    loss_cycle_B = self.l1(real_B, rec_B);
    # geometrical constraint
    loss_geo_A = self.cce(mask_A, pred_mask_B);
    
    return 5 * loss_idt_A + loss_adv_A + 10 * (loss_cycle_A + loss_cycle_B) + loss_geo_A;

  def GB_loss(self, inputs):
      
    (real_A, fake_B, idt_B, pred_fake_B, pred_real_B, rec_A, pred_mask_B, mask_A, real_B, fake_A, idt_A, pred_fake_A, pred_real_A, rec_B, pred_mask_A, mask_B) = inputs;
    # wgan gradient penalty
    loss_adv_B = self.l2(tf.ones_like(pred_fake_A), pred_fake_A);
    # generated image should not deviate too much from origin image
    loss_idt_B = self.l1(real_B, idt_B);
    # reconstruction loss
    loss_cycle_A = self.l1(real_A, rec_A);
    loss_cycle_B = self.l1(real_B, rec_B);
    # geometrical constraint
    loss_geo_B = self.cce(mask_B, pred_mask_A);
    
    return 5 * loss_idt_B + loss_adv_B + 10 * (loss_cycle_A + loss_cycle_B) + loss_geo_B;    
 
  def DA_loss(self, inputs):

    (real_A, fake_B, idt_B, pred_fake_B, pred_real_B, rec_A, pred_mask_B, mask_A, real_B, fake_A, idt_A, pred_fake_A, pred_real_A, rec_B, pred_mask_A, mask_B) = inputs;
    real_loss = self.l2(tf.ones_like(pred_real_B), pred_real_B);
    fake_loss = self.l2(tf.zeros_like(pred_fake_B), self.DA(self.pool_A.pick(fake_B)));
    return 0.5 * (real_loss + fake_loss);

  def DB_loss(self, inputs):

    (real_A, fake_B, idt_B, pred_fake_B, pred_real_B, rec_A, pred_mask_B, mask_A, real_B, fake_A, idt_A, pred_fake_A, pred_real_A, rec_B, pred_mask_A, mask_B) = inputs;
    real_loss = self.l2(tf.ones_like(pred_real_A), pred_real_A);
    fake_loss = self.l2(tf.zeros_like(pred_fake_A), self.DB(self.pool_B.pick(fake_A)));
    return 0.5 * (real_loss + fake_loss);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  silnet = SilNet((256,256,3));
  tf.keras.utils.plot_model(model = silnet, to_file = 'silnet.png', show_shapes = True, dpi = 64);
