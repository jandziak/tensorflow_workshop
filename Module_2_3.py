# Tensorflow workshop with Jan Idziak
#-------------------------------------
#
# Basic operations and functions
# Sin, cos, mod, square

import tensorflow as tf
import numpy as np

sess = tf.Session()

#Division functions
print(sess.run(tf.div(3, 4)))
print(sess.run(tf.truediv(3, 4)))
print(sess.run(tf.floordiv(3.0, 4.0)))

#Modulo
print(sess.run(tf.mod(22, 5)))

#Cross product
print(sess.run(tf.cross([1, 0, 0], [0, 1, 0])))

#Square, squareroot
print(sess.run(tf.square(2)))
print(sess.run(tf.sqrt(4.0)))

#Trigonometric
print(sess.run(tf.sin(3.1416)))
print(sess.run(tf.tan(3.1416)))
print(sess.run(tf.cos(3.1416)))
print(sess.run(tf.exp(1.0)))

#Equal 
print(sess.run(tf.equal(tf.cos(3.1416),-1.0)))

##Helper for exercises
#For loop 
for i in range(15):
	print(i)

#Definition of function
def square_plus_two(x):
	return x**2 + 2

print(square_plus_two(2))

### Exercise modelue_1_3

# Using tensorflow syntax
# Write function that return x*sin(x) - x^2*cos(x)
# For x values: 3.1416, 3.1416/2, 3.1416/3, ..., 3.1416/10 
# find the smalles value using for loop



