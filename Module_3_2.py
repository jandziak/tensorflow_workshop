# Tensorflow workshop with Jan Idziak
#-------------------------------------
#
#script based on the:
# Implementation of a simple MLP network with 
# one hidden layer.
#
# Linear Regression
#----------------------------------
#
# This function shows how to use TensorFlow to
# solve linear regression.
# y = Ax + b
# y = Wx
#
# We will use the iris data, specifically:
#  y = Sepal Length
#  x = Pedal Length, Petal Width, Sepal Width
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1):
    """
    Forward-propagation.
    """
    yhat = tf.matmul(X, w_1)
    return yhat

def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris   = datasets.load_iris()
    data   = np.array([[x[1], x[2], x[3]] for x in iris.data])
    target = np.array([y[0] for y in iris.data])

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data
    all_Y = target  # One liner trick!
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def train_model():
    train_X, test_X, train_y, test_y = get_iris_data()


    # Symbols
    X = tf.placeholder("float", shape=[None, 4])
    y = tf.placeholder("float", shape=[None, ])

    # Weight initializations
    w_1 = init_weights((4, 1))

    # Forward propagation
    yhat    = forwardprop(X, w_1)
    predict = yhat

    # Backward propagation
    cost    = tf.reduce_mean(tf.sqrt(tf.pow(predict-y, 2)))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(15):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

            train_accuracy = sess.run(cost, feed_dict={X: train_X, y: train_y})
            test_accuracy  = sess.run(cost, feed_dict={X: test_X, y: test_y})

        print("Epoch = %d, train MSE = %.2f, test MSE = %.2f"
              % (epoch + 1, train_accuracy,  test_accuracy))
        # print sess.run((y, predict), feed_dict={X: train_X, y: train_y})
    sess.close()
train_model()