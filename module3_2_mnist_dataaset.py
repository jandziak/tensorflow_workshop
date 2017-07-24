# Module 3: Datasets
# MNIST Handwriting Dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

def show_digit(index):
    label = y_train[index].argmax(axis=0)
    image = X_train[index].reshape([28,28])
    plt.title('Digit : {}'.format(label))
    plt.imshow(image, cmap='gray_r')
    plt.show()


show_digit(1)


# batch_X, batch_Y = mnist.train.next_batch(100)
# print(batch_X.shape)