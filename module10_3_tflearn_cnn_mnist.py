# Module 10: TFLearn
# CNN model for MNIST dataset and save model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
n_classes = 10
learning_rate = 0.5
training_epochs = 2
batch_size = 100
logdir = '/tmp/mnist/12'

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Step 1: Preprocess the  Data
import tflearn.datasets.mnist as mnist
X_train, y_train, X_test, y_test = mnist.load_data(one_hot=True)

X_train = X_train.reshape([-1, 28, 28, 1])
X_test = X_test.reshape([-1, 28, 28, 1])


# Step 2: Build the Network
network = input_data(shape=[None, 28, 28, 1])
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = dropout(network, 0.25)
network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, n_classes, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy')

# Step 3: Training
#model = tflearn.DNN(network)
model = tflearn.DNN(network,tensorboard_dir=logdir,tensorboard_verbose=3)
model.fit(X_train, y_train, n_epoch=training_epochs, show_metric=True, batch_size=batch_size)


# model.save('mnist_cnn.model')

# model.load('mnist_cnn.model')
# import numpy as np
# print( np.round(model.predict([X_test[1]])[0]) )
# print(test_y[1])