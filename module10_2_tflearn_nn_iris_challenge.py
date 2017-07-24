# Module 10: TF Learn
# Challenge: NN model for iris dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

# Parameters
n_features = 4
n_classes = 3
learning_rate = 0.05
training_epochs = 20
logdir = '/tmp/iris/1'

# Step 1: Preprocess the  Data
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
target = iris.target

# Convert the label into one-hot vector
num_labels = len(np.unique(target))
Y = np.eye(num_labels)[target]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Step 2: Build the  Network
L1 = 100
L2 = 40
L3 = 20
network = input_data(shape=[None, n_features])
network = fully_connected(network, L1, activation='relu')
network = fully_connected(network, L2, activation='relu')
network = fully_connected(network, L3, activation='relu')
network = fully_connected(network, n_classes, activation='softmax')
network = regression(network, optimizer='sgd', learning_rate=learning_rate,loss='categorical_crossentropy')

# Step 3: Training
model = tflearn.DNN(network,tensorboard_dir=logdir,tensorboard_verbose=3)
model.fit(X_train, y_train, n_epoch=training_epochs, validation_set=(X_test,y_test),show_metric=True)

# model.save('iris.model')

# model.load('iris.model')
# import numpy as np
# print( np.round(model.predict([X_test[1]])[0]) )
# print(test_y[1])