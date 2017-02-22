# Tensorflow workshop with Jan Idziak
#-------------------------------------
#
# Introduction to tensors with tensorflow
# constants, Variables, placeholder, feed_dict
import tensorflow as tf
import numpy as np


#Simple daffinition of the tensor, constant, and variable
tensor = tf.zeros([1, 10])
print(tensor)

constant = tf.constant([2, 3])
print(constant)

variable = tf.Variable(tf.zeros([1, 10]))
print(variable)

#tf.Session, variables initialization - What is inside?
initialize_ = tf.global_variables_initializer()


with tf.Session() as sess:
	sess.run(initialize_)
	print("Variable returns: ", variable)
	print("sess.run(variable) returns:", sess.run(variable))
	print("tensor returns: ", tensor)
	print("sess.run(tensor) returns:", sess.run(tensor))
	print("Variable returns: ", constant)
	print("sess.run(variable) returns:", sess.run(constant))

#Session initializaiton
sess = tf.Session()

#More sophisticated variables
n_row = 3
n_col = 2

#Ones vector
ones = tf.Variable(tf.ones([n_row, n_col]))

#Constant value tensor
const = tf.Variable(tf.constant(-1, shape=[n_row, n_col]))

#Linear Space tensor
linear = tf.Variable(tf.linspace(start=0.0, stop=13, num=7))

#Sequence tensor
seq = tf.Variable(tf.range(start=6, limit=23, delta=3))

#Random number 
rand = tf.Variable(tf.random_normal([n_row, n_col], mean=0.0, stddev=1.5))

#Printing ther results
#Remember always to initialize your variables
sess.run(rand.initializer)
print(sess.run(rand))
sess.close()

#Placeholders what is it and how does it work?
sess = tf.Session()

x = tf.placeholder(tf.float32, shape=(4, 4))
y = tf.identity(x)

rand_array = np.random.rand(4, 4)


print(sess.run(y, feed_dict={x: rand_array}))



### Exercise modelue_1_1

#Create new variables with:
#	a) values 1, 2, 3 
#	b) values 1, 2, 3, 4, 5, 6 having 2 columns and 3 rows
#	c) values 1, 2, 3,... 100 having 25 columns and 4 rows
#	d) constant values 5 having 10 25 columns and 4 rows

#Print results using single initializer 


