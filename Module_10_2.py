# -*- coding: utf-8 -*-
'''
Retraining (Finetuning) Example with vgg.tflearn. Using weights from VGG model to retrain
network for a new task (your own dataset).All weights are restored except
last layer (softmax) that will be retrained to match the new task (finetuning).
'''
from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.datasets import cifar10
from tflearn.data_utils import shuffle, to_categorical

num_classes = 10 # num of your dataset



(X, Y), (X_test, Y_test) = cifar10.load_data('cifar-10-batches-py')
X, Y = shuffle(X, Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)


# Redefinition of convnet_cifar10 network
network = input_data(shape=[None, 32, 32, 3])
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = dropout(network, 0.75) 
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = dropout(network, 0.5)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
# Finetuning Softmax layer (Setting restore=False to not restore its weights)
softmax = fully_connected(network, num_classes, activation='softmax', restore=False)
regression = regression(softmax, optimizer='adam',
                        loss='categorical_crossentropy',
                        learning_rate=0.001)  

model = tflearn.DNN(regression, checkpoint_path='cifar_apply',
                    max_checkpoints=3, tensorboard_verbose=0)
# Load pre-existing model, restoring all weights, except softmax layer ones
model.load('./models/cifar_10_50_96')

# Start finetuning
model.fit(X, Y, n_epoch=1, validation_set=(X_test, Y_test), shuffle=True, 
          show_metric=True, batch_size=64, snapshot_step=200,
          snapshot_epoch=False, run_id='cifar_apply')

model.save('cifar_apply')