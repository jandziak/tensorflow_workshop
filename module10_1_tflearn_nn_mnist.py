# Module 10: TFLearn
# NN model for MNIST dataset and save model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
n_features = 784
n_classes = 10
learning_rate = 0.5
training_epochs = 2
batch_size = 100
logdir = '/tmp/mnist/12'

# Step 1: Preprocess the  Data
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

import tflearn.datasets.mnist as mnist
X_train, y_train, X_test, y_test = mnist.load_data(one_hot=True)

# Step 2: Build the  Network
L1 = 200
L2 = 100
L3 = 60
L4 = 30
network = input_data(shape=[None, n_features])
network = fully_connected(network, L1, activation='relu')
network = fully_connected(network, L2, activation='relu')
network = fully_connected(network, L3, activation='relu')
network = fully_connected(network, L4, activation='relu')
network = fully_connected(network, n_classes, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy')

# Step 3: Training
model = tflearn.DNN(network)
#model = tflearn.DNN(network,tensorboard_dir=logdir,tensorboard_verbose=3)
#model.fit(X_train, y_train, n_epoch=training_epochs, show_metric=True, batch_size=batch_size)

#model.save('mnist2.model')

# Step 4: Evaluation
model.load('mnist2.model')

import numpy as np
print( np.round(model.predict([X_test[1]])[0]) )
print(y_test[1])