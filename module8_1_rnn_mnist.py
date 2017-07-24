# Module 8: Recurrent Neural Network
# RNN model for MNIST dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
learning_rate = 0.5
training_epochs = 2
batch_size = 100
rnn_size = 128

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist", one_hot=True)

# Step 1: Initial Setup
X = tf.placeholder(tf.float32, [None, 28, 28])
y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([rnn_size, 10]))
B = tf.Variable(tf.random_normal([10]))

# Step 2: Setup Model
inp = tf.unstack(X, axis=1)

cell = rnn.BasicRNNCell(rnn_size) # Simple RNN  Cell
# cell = rnn.BasicLSTMCell(rnn_size) # LSTM Cell
# cell = rnn.GRUCell(rnn_size) # GRU Cell

H, states = rnn.static_rnn(cell, inp, dtype=tf.float32)

Ylogits = tf.matmul(H[-1], W) + B
yhat = tf.nn.softmax(Ylogits)

# Step 3: Loss Functions
loss = tf.reduce_mean(
   tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=Ylogits))

# Step 4: Optimizer
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)

# accuracy of the trained model, between 0 (worst) and 1 (best)
is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Step 5: Training Loop
for epoch in range(training_epochs):
    for i in range(int(mnist.train.num_examples/batch_size)):
        batch_X, batch_y = mnist.train.next_batch(batch_size)
        batch_X = batch_X.reshape((batch_size, 28, 28))
        train_data = {X: batch_X, y: batch_y}
        sess.run(train, feed_dict=train_data)
        print("Training Accuracy = ", sess.run(accuracy, feed_dict=train_data))

# Step 6: Evaluation
test_X = mnist.test.images
test_y = mnist.test.labels
test_X = test_X.reshape((-1, 28, 28))
print("Testing Accuracy = ", sess.run(accuracy, feed_dict = {X:test_X,y:test_y}))

