# Module 6: Tensorboard
# Author: Dr. Alfred Ang
# Challange: Restore model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
learning_rate = 0.5
training_epochs = 2
batch_size = 100
logdir = '/tmp/mnist/8'

import tensorflow as tf
tf.set_random_seed(25)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

# Step 1: Initial Setup
with tf.name_scope('Inputs') as scope:
    X = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

L1 = 200
L2 = 100
L3 = 60
L4 = 30

with tf.name_scope('Variables'):
    W1 = tf.Variable(tf.truncated_normal([784, L1], stddev=0.1))
    B1 = tf.Variable(tf.zeros([L1]))
    W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))
    B2 = tf.Variable(tf.zeros([L2]))
    W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
    B3 = tf.Variable(tf.zeros([L3]))
    W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
    B4 = tf.Variable(tf.zeros([L4]))
    W5 = tf.Variable(tf.truncated_normal([L4, 10], stddev=0.1))
    B5 = tf.Variable(tf.zeros([10]))

# Step 2: Setup Model
with tf.name_scope('Model'):
    Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
    Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
    Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
    Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
    Ylogits = tf.matmul(Y4, W5) + B5
    yhat = tf.nn.softmax(Ylogits)

# Step 3: Loss Functions
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=Ylogits))

# Step 4: Optimizer
with tf.name_scope('Train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # optimizer = tf.train.AdamOptimizer(0.1)
    train = optimizer.minimize(loss)

# accuracy of the trained model, between 0 (worst) and 1 (best)
with tf.name_scope('Accuracy'):
    is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Collecting data to Tensorboard
writer = tf.summary.FileWriter(logdir, sess.graph)

tf.summary.scalar("Loss", loss)
tf.summary.scalar("Accuracy", accuracy)

tf.summary.histogram('W1', W1)
tf.summary.histogram('B1', B1)
tf.summary.histogram('W2', W2)
tf.summary.histogram('B2', B2)
tf.summary.histogram('W3', W3)
tf.summary.histogram('B3', B3)
tf.summary.histogram('W4', W4)
tf.summary.histogram('B4', B4)
tf.summary.histogram('W5', W5)
tf.summary.histogram('B5', B5)

summary_op = tf.summary.merge_all()

# Step 5: Training Loop
for epoch in range(training_epochs):
    num_batches = int(mnist.train.num_examples/batch_size)
    for i in range(num_batches):
        batch_X, batch_y = mnist.train.next_batch(batch_size)
        train_data = {X: batch_X, y: batch_y}

        summary = sess.run(summary_op, feed_dict =train_data)
        writer.add_summary(summary, global_step=epoch*num_batches+i+1)

        sess.run(train, feed_dict=train_data)
        print(epoch*num_batches+i+1, "Training accuracy =", sess.run(accuracy, feed_dict=train_data),
              "Loss =", sess.run(loss, feed_dict=train_data))


# Step 6: Evaluation
test_data = {X:mnist.test.images,y:mnist.test.labels}
print("Testing Accuracy = ", sess.run(accuracy, feed_dict = test_data))
