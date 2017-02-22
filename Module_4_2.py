# Tensorflow workshop with Jan Idziak
#-------------------------------------
#
#script harvested from:
#https://github.com/nfmcclure
#
# Elastic Net Regression
#----------------------------------
#
# This function shows how to use TensorFlow to
# solve elastic net regression.
# y = Ax + b
#
# We will use the iris data, specifically:
#  y = Sepal Length
#  x = Pedal Length, Petal Width, Sepal Width

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

# Create graph
sess = tf.Session()

# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([[x[1], x[2], x[3]] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])
print(x_vals)
print(y_vals)
# Declare batch size
batch_size = 50

# Initialize placeholders
x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[3,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Declare model operations
model_output = tf.add(tf.matmul(x_data, A), b)

# Declare the elastic net loss function
elastic_param1 = tf.constant(1.)
elastic_param2 = tf.constant(1.)
l1_a_loss = tf.reduce_mean(tf.abs(A))
l2_a_loss = tf.reduce_mean(tf.square(A))
e1_term = tf.multiply(elastic_param1, l1_a_loss)
e2_term = tf.multiply(elastic_param2, l2_a_loss)
#Expand_dims function:
#if you have a single image of 
#shape [height, width, channels],
#you can make it a batch of 1 image with 
#expand_dims(image, 0), which will make 
#the shape [1, height, width, channels].
loss = tf.expand_dims(
	tf.add(
		tf.add(
			tf.reduce_mean(
				tf.square(y_target - model_output)), 
			e1_term), 
		e2_term), 
	0)

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
for i in range(1000):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss[0])
    if (i+1)%250==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))

# Get the optimal coefficients
[[sw_coef], [pl_coef], [pw_ceof]] = sess.run(A)
[y_intercept] = sess.run(b)

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

### Exercise modelue_4_2

#Create new variables with:
#   a) use mean absolute error instead of MSE to find solution
#   b) Fit 3 models and see for which the loss value is smallest:
#   b') ElasticNet model with lambda_1 = 1., lambda_2 = 1. 
#   b'') Lasso model lambda_1 = 1., lambda_2 = 0 
#   b''') Ridge model lambda_1 = 0, lambda_2 = 1. 
#   c) see what happenes after you change num of cluster to 3, 4, or 5
