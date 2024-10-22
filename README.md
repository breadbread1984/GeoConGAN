# GeoConGAN
This project implement the 3d hand joint detection algorithm introduced in [GANerated Hands for Real-Time 3D Hand Tracking from Monocular RGB](https://arxiv.org/abs/1712.01057).

## where to download dataset
real hands images can be downloaded [here](https://handtracker.mpi-inf.mpg.de/data/RealHands.zip)

synth hands images and corresponding 3d joints positions can be downloaded [here](https://handtracker.mpi-inf.mpg.de/data/SynthHands.zip)

uncompress with the following command

```bash
mkdir /mnt/RealHands
unzip -d /mnt/RealHands RealHands.zip
unzip -d /mnt SynthHands.zip
```

## how to generate dataset
generate tfrecord format dataset with the following command

```bash
python3 create_dataset.py
```

executing the command successfully, you can find real.tfrecord and synthetic.tfrecord generated under directory datasets.

## how to train SilNet
SilNet serves as a geometrical constraint supervisor during training GeoConGAN. two silnets need to be trained for real and synthetic hands respectively.

train silnet for real hands with

```bash
python3 train_silnet.py real
```

after convergence, save model with

```bash
mkdir models
python3 save_silnet.py
mv silnet.h5 models/real_silnet.h5
rm -rf checkpoints
```

train silnet for synthetic hands with

```bash
python3 train_silnet.py synthetic
```

after convergence, save model with

```bash
python3 save_silnet.py
mv silnet.h5 models/synthetic_silnet.h5
rm -rf checkpoints
```

the pretrained model has been enclosed in this project.

## how to train GeoConGAN
GeoConGAN is trained to generate real-like hands from synthetic samples. it is trained with

```bash
python3 train_geocongan.py
```

there are a lot of inaccurated annotated mask in the real hand image dataset, so training this model is bound to unsuccessful. if you want to improve the training result, you need to wash the data yourself.

## how to train RegNet

train RegNet with command

```python
python3 train_regnet.py
```

unfortunately, I failed training regnet as well. my model can't converge. hope you can contribute to this project to make it work.

