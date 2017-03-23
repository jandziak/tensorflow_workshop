# Tensorflow workshop with Jan Idziak
#-------------------------------------
#
# Introduction to tensors with tensorflow
# constants, Variables, placeholder, feed_dict
import tensorflow as tf
import numpy as np


#Simple daffinition of the tensor, constant, and variable
tensor = tf.zeros([2, 2, 3])
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


#Placeholders what is it and how does it work?
sess = tf.Session()

x = tf.placeholder(tf.float32, shape=(4, 4))
y = tf.identity(x)

rand_array = np.random.rand(4, 4)


print(sess.run(y, feed_dict={x: rand_array}))



# ### Exercise modelue_2_1

# #    create placeholder x of a shape (2, 2)
# #    y = x**2 + x
# #    create random matrix of a shape (2, 2)
# #    feed x
# #    look at y

# #Print results using single initializer 


x = tf.placeholder(tf.float32, shape=(2, 2))
#x**2 + x
y = tf.add(tf.multiply(x, x), x)
rand_array_2 = np.random.rand(2, 2)
print(sess.run(y, feed_dict ={x: rand_array_2}))