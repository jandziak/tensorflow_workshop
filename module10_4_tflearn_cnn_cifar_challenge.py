# Module 9: TF Learn
# Challenge : CIFAR-10 dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# Parameters
learning_rate = 0.001
training_epochs = 1
batch_size = 100
logdir = '/tmp/cifar/12'

import tensorflow as tf
from tflearn.datasets import cifar10
from tflearn.data_utils import shuffle, to_categorical

# Step 1: Preprocessing Data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, y_train = shuffle(X_train, y_train)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Step 2: Built the Network
network = input_data(shape=[None, 32, 32, 3])
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam',loss='categorical_crossentropy',learning_rate=learning_rate)

# Step 3: Training
model = tflearn.DNN(network)
#model = tflearn.DNN(network,tensorboard_dir=logdir,tensorboard_verbose=3)
model.fit(X_train, y_train, n_epoch=training_epochs, shuffle=True, validation_set=(X_test, y_test),
show_metric=True, batch_size=batch_size)