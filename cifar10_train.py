from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import cifar10_input

FLAGS = tf.app.flags.FLAGS

#basic model parameters
tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in batch""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data', """Path to CIFAR10 data directory""")
tf.app.flags.DEFINE_boolean('useFp16',False,"""Train the model using fp16.""")

#Global constants describing the CIFAR-10 dataset

IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

#Constants describing the training process

MOVING_AVERAGE_DECAY = 0.9999 
NUM_EPOCH_PER_DECAY = 350
LEARNING_RATE_DECAY_FACOTR = 0.1
INITIAL_LEARNING_RATE = 0.1

TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def inputs(evalData):
    if not FLAGS.dataDir:
        raise ValueError("Please supply a data directory")
    dataDir = os.path.join(FLAGS.dataDir, "cifar-10-batches-bin")
    images, labels = cifar10_inputs.createInput(evalData=evalData, dataDir=dataDir,batchSize = FLAGS.batchSize)
    
    if FLAGS.useFp16:
        images = tf.cast(images,tf.float16)
        labels = tf.cast(labels,tf.float16)
    return images, labels

def train(totalLoss, globalStep):
    numBatchPerExpoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/ FlAGS.batchSize
    decaySteps = int(numBatchesPerEpoch * NUM_EPOCHS_PER_DAY)
    
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step,decaySteps, LEARNING_RATE_DECAY_FACOTR, staircase=True)
    tf.summary.scalar('learning rate', lr)
    
    lossAverageOp = addLossSummaries(totalLoss)
    
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.computeGradients(total_loss)
        
        
    applyGradientOp = opt.apply_gradients(grads, global_step=globalStep) 
    
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name,var)
    
    
    
    





