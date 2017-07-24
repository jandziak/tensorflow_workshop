# Module 4: Simple TF Model
# Simple TF Model - Linear Regression

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

# Step 1: Initial Setup
X = tf.placeholder(tf.float32)
W = tf.Variable([0.1],tf.float32)
b = tf.Variable([0.1],tf.float32)

# Step 2: Model
yhat = tf.multiply(W,X) + b
y = tf.placeholder(tf.float32) # Placeholder for correct answer

# # Step 3: Loss Function
loss = tf.reduce_sum(tf.square(yhat - y)) # sum of the squares error

# # Step 4: Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# # training data
X_train = [1,2,3,4,5]
y_train = [0,-1.5,-1.6,-3.1,-4]

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# # Step 5: Training Loop
for i in range(1000):
  sess.run(train, {X:X_train, y:y_train})

# Step 6: Evaluation
import matplotlib.pyplot as plt
plt.plot(X_train,y_train,'o')
plt.plot(X_train,sess.run(tf.multiply(W,X_train)+b),'r')
plt.show()

