# Module 4: Simple TF Models
# Challenge: Iris flower dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
learning_rate = 0.05
training_epochs = 20

import tensorflow as tf
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

# Step 1: Initial Setup
X = tf.placeholder(tf.float32, [None, 4])
W = tf.Variable(tf.truncated_normal([4, 3],stddev=0.1))
b = tf.Variable(tf.truncated_normal([3],stddev=0.1))

# Step 2: Define Model
yhat = tf.matmul(X, W) + b
y = tf.placeholder(tf.float32, [None, 3]) # Placeholder for correct answer

# Step 3: Loss Function
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))

# Step 4: Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(yhat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Step 5: Training Loop
for epoch in range(training_epochs):
    for i in range(len(X_train)):
        train_data = {X: X_train[i: i + 1], y: y_train[i: i + 1]}
        sess.run(train, feed_dict = train_data)
        print(epoch*len(X_train)+i, "Training accuracy =", sess.run(accuracy, feed_dict=train_data),
          "Loss =", sess.run(loss, feed_dict=train_data))

# Step 6: Evaluation
test_data = {X: X_test, y: y_test}
print("Training Accuracy = ", sess.run(accuracy, feed_dict = test_data))


