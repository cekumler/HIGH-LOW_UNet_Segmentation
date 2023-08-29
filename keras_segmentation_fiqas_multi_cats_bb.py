#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   keras_segmentation.py

   original script to train NN model

   author Jebb Q Stewart (jebb.q.stewart@noaa.gov)
   Copyright 2019. Colorado State University. All rights reserved.

   modified to read in the FIQAS and GfS High-Low labeled data
   Christina Kumler

"""

import os
import platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action="ignore",category=DeprecationWarning)

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

useHorovod = False 

import zarr
import tensorflow.keras as keras
import argparse
import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, TensorBoard

#ck added where she added the numpy() to figure out tensor issue
from tensorflow.python.ops.numpy_ops import np_config

if useHorovod:
   import horovod.keras as hvd

#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.pyplot as plt

from utils.utils import usage
from learning.models.unet import unet
from learning.losses.losses import dice_coeff, bce_dice_loss, dice_loss, tversky_coeff, tversky_loss, focal_loss
from learning.learning_utils import get_model_memory_usage

import logging
logging_format = '%(asctime)s - %(name)s - %(message)s'
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, 
    format=logging_format, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("KerasSegmentation")

headless = True

# gray = plt.cm.gray

# for gfs, default batch size
batch_size = 2

# default epochs
epochs = 100

# Horovod: initialize Horovod.
if useHorovod:
  hvd.init()
  logger.info("horovod size: %s  rank: %s  device rank: %s  host: %s", hvd.size(), hvd.rank(), hvd.local_rank(), platform.node())

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

if useHorovod and gpus:
  tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
 
print(tf.config.list_physical_devices('GPU'))
model_name = ""

def preprocessGFSData(train, test):
  # resize and normalize data
  # forces the data to a power of 2 for depth of unet
  # FIQAS is originally 1799x1059
  # 1056 can get me 5 layers deep
  # 1792 can get me at least 5 deep -- max 9 (almost a perfect power) 
  train = train[:,1:-2,3:-4,:]
  test = test[:,1:-2,3:-4,:]
#way2  train = train[:,3:-4,1:-2,:]
#way2  test = test[:,3:-4,1:-2,:]

  # normalize train
  # normalize on my range of values ~9000 to 11000 
#first multi cats  train = (train - 101000.0)/3000.0
  train = (train - 101000.0)/4000.0

# never uncomment this will make the code set any TC to 0 or 1 based on above category = 2
#og  #test[test<category] = 0
#og  #test[test>=category] = 1
#og  #test = tf.stack(test)

  # new way to handle cateogries
  test = test.astype(np.int32)
#  print('test: ', test)
#  test = test.numpy()
  test = to_categorical(test, num_classes=3)

  return train, test

class generator(Sequence):
    """ Data generator

    We can read all the data into memory so we need to split it up using a generator

    """
    def __init__(self, x_set, y_set, batch_size, name="", limit=None):
        self.x, self.y = x_set, y_set
        self.name = name
        self.batch_size = batch_size
        self.limit = limit

        # set the size of our dataset to the max,
        # Limit, or the length of the array
        if self.limit is None:
          self.length = self.x.shape[0]
        else:
          self.length = min(self.limit, self.x.shape[0])

        # create random order on how to read the data
        self.order = np.random.permutation(self.x.shape[0])[0:self.length]

    def __len__(self):
        # how many batches do we have?
        return int(np.ceil(self.length / float(self.batch_size)))

    def __getitem__(self, idx):
        cutStart = idx * self.batch_size
        cutEnd = (idx + 1) * self.batch_size

        indices = self.order[cutStart:cutEnd]

        train = np.array(self.x.oindex[indices])
        test = np.array(self.y.oindex[indices])

        # for gfs
        train, test = preprocessGFSData(train, test)

        return train, test

#class generator2():
#    """ Data generator
    
#    We can read all the data into memory so we need to split it up using a generator

#    """
#    def __init__(self, x_set, y_set, batch_size, name="", limit=None):
#        self.x, self.y = x_set, y_set
#        self.name = name
#        self.batch_size = batch_size
#        self.limit = limit

#        # set the size of our dataset to the max, 
#        # Limit, or the length of the array
#        if self.limit is None:
#          self.length = self.x.shape[0]
#        else:
#          self.length = min(self.limit, self.x.shape[0])
#
#        # create random order on how to read the data
#        self.order = np.random.permutation(self.x.shape[0])[0:self.length]
#
#    def __len__(self):
#        # how many batches do we have?
#        return int(np.ceil(self.length / float(self.batch_size)))
#
#    def __call__(self):
#        while True:
#
#            batches = int(np.ceil(self.length / float(self.batch_size)))
#
#            for idx in range(batches):
#                cutStart = idx * self.batch_size
#                cutEnd = (idx + 1) * self.batch_size
#
#                indices = self.order[cutStart:cutEnd]

#                #train = np.array(self.x.oindex[indices])
#                #test = np.array(self.y.oindex[indices])

#                train = self.x.oindex[indices]
#                test = self.y.oindex[indices]

#                # for gfs
#                #train, test = self.preprocessGFSData2(train, test)

#                # yield tf.convert_to_tensor(train, dtype=np.float32), tf.convert_to_tensor(test, dtype=np.uint8)
#                yield train, test


def readData(file):
 
  if not useHorovod or hvd.rank() == 0:
     logger.info("reading in training %s", file)
  loaded_data = zarr.open(file, mode='r')
  x_train = loaded_data['train']
  y_train = loaded_data['test']

  return x_train,y_train

def main(lossfunction="tversky", lossrate=1e-4, depth=4, optimizer="rms", n_filters=32, fixed=False, resnet=False, batchnorm=True, dropout=False, dropout_rate=0.10, noise=False, noise_rate=0.1, ramp=False, earlystop=False):

  verbose = 0
  if not useHorovod:
    verbose = 1
  elif hvd.rank() == 0:
    verbose = 2

  verbose = 2
  train_file = '../data_ddrf_bigger_box/fiqas-block_train_2019.zip'
  val_file = '../data_ddrf_bigger_box/fiqas-block_val_2019.zip'
  test_file = '../data_ddrf_bigger_box/fiqas-block_test_2019.zip'

  if verbose > 0:
     logger.info('reading in train data')

  x_train, y_train = readData(train_file)
  x_val, y_val = readData(val_file)
  x_test, y_test = readData(test_file)

  #sample = np.min(100, x_train.shape[0])
  sample = 100
  if verbose > 0:
     logger.info("Sample data: ")
     logger.info("  Training input : max[0]: %s", np.max(x_train[sample,:,:,0]))
     logger.info("                   min[0]: %s", np.min(x_train[sample,:,:,0]))
     logger.info("           shape : %s", x_train.shape)
     logger.info("  Train labels: max[0]: %s min[1]: %s  dtype: %s", np.max(y_train[sample,:,:,:]), np.min(y_train[sample,:,:,:]), y_train.dtype)
     logger.info("  using loss function %s", lossfunction)

  loss = tversky_loss(alpha=0.3, beta=0.7)
  if lossfunction == "dice":
     loss = dice_loss
  elif lossfunction == "bce_dice":
     loss = bce_dice_loss
  elif lossfunction == "focal":
     loss = focal_loss(gamma=2,alpha=0.6)
     #loss = focal_loss(gamma=2,alpha=0)
  elif lossfunction == "bce":
     loss = 'binary_crossentropy'

  if useHorovod:
     # for gfs 
     model = unet(img_rows=1056, img_cols=1792, channels=x_train.shape[3], activation='relu', final_activation='sigmoid', fixed=fixed, 
               batchnorm=batchnorm, output_channels=3, resnet=resnet, n_filters=n_filters, depth=depth, 
               dropout=dropout, dropout_rate=dropout_rate, noise=noise, noise_rate=noise_rate, verbose=verbose)

     opt = hvd.DistributedOptimizer(RMSprop(lr=lossrate*hvd.size()))
     if optimizer == "adam":
        opt = hvd.DistributedOptimizer(Adam(lr=lossrate*hvd.size()))
     #testmodel.compile(optimizer=opt, loss=loss, metrics=[tversky_coeff(alpha=0.3, beta=0.7), dice_coeff, 'accuracy'])
     model.compile(run_eagerly=True, optimizer=opt, loss=loss, metrics=[tversky_coeff(alpha=0.3, beta=0.7), dice_coeff, 'accuracy'])

  else:
     # Create a MirroredStrategy.
     strategy = tf.distribute.MirroredStrategy()
#     strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
     print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

     # Open a strategy scope.
     with strategy.scope():    # for gfs
         model = unet(img_rows=1056, img_cols=1792, channels=x_train.shape[3], activation='relu', final_activation='sigmoid', fixed=fixed,
               batchnorm=batchnorm, output_channels=3, resnet=resnet, n_filters=n_filters, depth=depth,
               dropout=dropout, dropout_rate=dropout_rate, noise=noise, noise_rate=noise_rate, verbose=verbose)

         opt = RMSprop(lr=lossrate)
         if optimizer == "adam":
           opt = Adam(lr=lossrate)

         #testmodel.compile(optimizer=opt, loss=loss, metrics=[tversky_coeff(alpha=0.3, beta=0.7), dice_coeff, 'accuracy'])
         model.compile(optimizer=opt, loss=loss, metrics=[tversky_coeff(alpha=0.3, beta=0.7), dice_coeff, 'accuracy'])
#         model.compile(run_eagerly=True, optimizer=opt, loss=loss, metrics=[tversky_coeff(alpha=0.3, beta=0.7), dice_coeff, 'accuracy'])

  if verbose > 0:
 
     logger.info("Model Summary:\n%s", model.summary())
     logger.info("Estimated Model GPU usage: %s GB", get_model_memory_usage(batch_size, model))
     logger.info("Current host memory usage: %s", usage());

     # serialize model to JSON
     model_json = model.to_json()
     if not os.path.isdir("models"): os.makedirs("models") 
     model_file = "models/" + model_name + ".json"
     with open(model_file, "w") as json_file:
        json_file.write(model_json)
     logger.info("saved model to %s", model_file)

  callbacks = []

  if useHorovod:
      callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
      callbacks.append(hvd.callbacks.MetricAverageCallback())
      callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=verbose))

  if earlystop:
              callbacks.append(EarlyStopping(monitor='val_loss',
                         patience=30,
                         verbose=verbose,
                         min_delta=1e-4,
                         restore_best_weights=True))

  if ramp:
              # Horovod: after the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
              if useHorovod:
                callbacks.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=15, end_epoch=40, multiplier=1.))
                callbacks.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=40, end_epoch=70, multiplier=1e-1))
                callbacks.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=70, end_epoch=100, multiplier=1e-2))
                callbacks.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=100, multiplier=1e-3))
 
              # Reduce the learning rate if training plateaues.
              #keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1)]
              #ReduceLROnPlateau(monitor='val_loss',
              #               factor=0.1,
              #               patience=4,
              #               verbose=1,
              #               min_delta=1e-4),


  training_bg = generator(x_train, y_train, batch_size, name="train", limit=None)
  val_bg = generator(x_val, y_val, batch_size, name="val", limit=None)
  test_bg = generator(x_test, y_test, batch_size, name="test", limit=None)

  if useHorovod:
    training_bg.order = hvd.broadcast(training_bg.order, 0, name='training_bg_order').numpy()
    val_bg.order = hvd.broadcast(val_bg.order, 0, name='val_bg_order').numpy()
    test_bg.order = hvd.broadcast(test_bg.order, 0, name='test_bg_order').numpy()

  if verbose > 0:
     logger.info("Training size: %s : steps : %s", training_bg.length, (training_bg.length//batch_size))
     logger.info("Validation size: %s : steps : %s", val_bg.length, (val_bg.length//batch_size))

  # Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
  if not useHorovod or hvd.rank() == 0:
      if not os.path.isdir("checkpoints"):
        os.makedirs("checkpoints")
      #callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoints/' + model_name + '_checkpoint-{epoch:02d}-{val_loss:.3f}.hdf5', monitor='val_loss', save_best_only=True, save_weights_only=False))
      #callbacks.append(keras.callbacks.TensorBoard(log_dir='tflogs'))

  size = 1
  if useHorovod:
    size = hvd.size()

#  useTensorboard = True
#  if useTensorboard:
#     # Import TensorBoard
#     from tensorflow.keras.callbacks import TensorBoard
#
#     # Define Tensorboard as a Keras callback
#     tensorboard = TensorBoard(
#       log_dir='./tblogs',
#       histogram_freq=1,
#       profile_batch=(2,10),
#       write_images=True
#     )
#     callbacks.append(tensorboard)

  history = model.fit_generator(generator=training_bg,
         steps_per_epoch=(training_bg.length//batch_size) // size,
         epochs=epochs,
         verbose=verbose,
         callbacks=callbacks,
         validation_data=val_bg,
         validation_steps=(val_bg.length // batch_size) // size,
         #use_multiprocessing=True,
         workers=2,
         max_queue_size=8)

  if not useHorovod or hvd.rank() == 0:
     # serialize weights to HDF5
     logger.info("saving weights")
     if not os.path.isdir("weights"):
        os.makedirs("weights")

     weights_file = "weights/" + model_name + ".h5"
     model.save_weights(weights_file)
     logger.info("Saved weights to disk %s", weights_file)

     # clear session
     #logger.debug("clearing session")
     #K.clear_session()
   
     logger.info("evaluating results")

  scores = model.evaluate(test_dataset, steps=(test_bg.length//batch_size) // size, workers=2,
         max_queue_size=8, use_multiprocessing=True, verbose=0)

  if not useHorovod or hvd.rank() == 0:

    logger.info('Test scores: %s', scores)
    if not os.path.isdir("images"):
        os.makedirs("images")
  
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig("images/" + model_name +"_acc.png")
    # plt.clf()
  
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig("images/" + model_name +"_loss.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run Keras Segmentation")

    parser.add_argument('-n', '--name',
       dest="name", required=True,
       help="name of output prefix for saving model, weights, images, etc...")

    parser.add_argument('-l', '--loss',
       dest="loss", required=False, default="dice",
       help="loss function to use")

    parser.add_argument('-lr', '--lossrate',
       dest="lossrate", required=False, default=1e-5, type=float,
       help="loss rate")

    parser.add_argument('-b', '--batch',
       dest="batch_size", required=False, default=4, type=int,
       help="batch size")
  
    parser.add_argument('-bn', '--batchnorm',
       dest="batchnorm", required=False, action='store_true',
       help="use batchnorm")

    parser.add_argument('-o', '--optimizer',
       dest="optimizer", required=False, default="adam",
       help="optimizer to use")

    parser.add_argument('-nf', '--n_filters',
       dest='n_filters', required=False, default=32, type=int,
       help="number of filters to use")

    parser.add_argument('-d', '--depth',
       dest='depth', required=False, default=6, type=int,
       help="depth of NN")

    parser.add_argument('-rn', '--resnet',
       dest="resnet", required=False, action='store_true',
       help="use resnet")

    parser.add_argument('-f', '--fixed',
       dest="fixed", required=False, action='store_true',
       help="use fixed NN")

    parser.add_argument('-e', '--epochs',
       dest='epochs', required=False, default=100, type=int,
       help="number of epochs")

    parser.add_argument('--dropout', dest='dropout', required=False, action='store_true', help='use dropout')

    parser.add_argument('--dropout_rate',
       dest="dropout_rate", required=False, default=0.1, type=float,
       help="dropout rate")

    parser.add_argument('--noise', dest='noise', required=False, action='store_true', help='use noise')

    parser.add_argument('--noise_rate',
       dest="noise_rate", required=False, default=0.1, type=float,
       help="noise rate")

    parser.add_argument('--ramp', dest='ramp', required=False, action='store_true', help='use ramp')
    parser.add_argument('--earlystop', dest='earlystop', required=False, action='store_true', help='use earlystop')


    args = parser.parse_args()

    model_name = args.name
    loss = args.loss.lower()
    lossrate = args.lossrate
    optimizer = args.optimizer.lower()
    n_filters = args.n_filters
    depth = args.depth
    resnet = args.resnet
    fixed = args.fixed
    batchnorm = args.batchnorm
    epochs = args.epochs
    batch_size = args.batch_size
    dropout = args.dropout
    dropout_rate = args.dropout_rate
    noise = args.noise
    noise_rate = args.noise_rate
    earlystop = args.earlystop
    ramp = args.ramp

    if not useHorovod or hvd.rank() == 0:
       logger.info("running with loss function: %s and loss rate of %s using optimizer %s", loss, lossrate, optimizer)
       logger.info("  saving info with prefix of %s", model_name)
       logger.info("  batch size: %s", batch_size)

    main(lossfunction=loss, lossrate=lossrate, optimizer=optimizer, n_filters=n_filters, batchnorm=batchnorm, resnet=resnet,
         fixed=fixed, depth=depth, dropout=dropout, dropout_rate=dropout_rate, noise=noise, noise_rate=noise_rate, earlystop=earlystop, ramp=ramp)

