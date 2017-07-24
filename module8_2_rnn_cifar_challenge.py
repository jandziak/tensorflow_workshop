# Module 8: Recurrent Neural Network
# Challenge: RNN on CIFAR-10 dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow.contrib import rnn
from tflearn.datasets import cifar10
from tflearn.data_utils import shuffle, to_categorical

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, y_train = shuffle(X_train, y_train)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

training_epochs = 8
n_classes = 10
batch_size = 100
chunk_size = 32 * 3
n_chunks = 32
rnn_size = 128

# Step 1: Initial Setup
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_normal([rnn_size, n_classes]))
B = tf.Variable(tf.random_normal([n_classes]))

# Step 2: Setup Model
inp = tf.reshape(X, [-1, n_chunks, chunk_size])
inp = tf.unstack(inp, axis=1)

# LSTM Cell
LSTM = rnn.BasicLSTMCell(rnn_size)
H, states = rnn.static_rnn(LSTM, inp, dtype=tf.float32)

# GRU Cell
# GRU = rnn.GRUCell(rnn_size)
# H, states = rnn.static_rnn(GRU, inp, dtype=tf.float32)

Ylogits = tf.matmul(H[-1], W) + B
yhat = tf.nn.softmax(Ylogits)

# Step 3: Loss Functions
loss = tf.reduce_mean(
   tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=Ylogits))

# Step 4: Optimizer
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

# accuracy of the trained model, between 0 (worst) and 1 (best)
is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Step 5: Training Loop
for epoch in range(training_epochs):
    for i in range(int(X_train.shape[0] / batch_size)):
        batch_X = X_train[(i*batch_size):((i+1)*batch_size)]
        batch_y = y_train[(i*batch_size):((i+1)*batch_size)]
        train_data = {X: batch_X, y: batch_y}
        sess.run(train, feed_dict=train_data)
        print("Training Accuracy = ", sess.run(accuracy, feed_dict=train_data))

# Step 6: Evaluation
acc = []
for i in range(int(X_test.shape[0] / batch_size)):
    batch_X = X_test[(i*batch_size):((i+1)*batch_size)]
    batch_y = y_test[(i*batch_size):((i+1)*batch_size)]
    test_data = {X: batch_X, y: batch_y}
    sess.run(train, feed_dict = test_data)
    acc.append(sess.run(accuracy, feed_dict = test_data))

print("Testing Accuracy/Loss = ", sess.run(tf.reduce_mean(acc)))