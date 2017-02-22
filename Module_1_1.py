# Tensorflow workshop with Jan Idziak
#-------------------------------------
#
# Based on the tf presentation	 
# First tensor flow 
#
import tensorflow as tf
import numpy as np
sess = tf.Session()
x = tf.placeholder('float', [1,3])
w = tf.Variable(tf.random_normal([3, 3]), name='w')
y = tf.matmul(x, w)
relu_out = tf.nn.relu(y)
softmax = tf.nn.softmax(relu_out)
sess.run(tf.global_variables_initializer())
answer = np.array([[0.0, 1.0, 0.0]])
result = answer - sess.run(softmax, feed_dict={x:np.array([[1.0, 2.0, 3.0]])})
print result

# optimization

labels = tf.placeholder("float", [1, 3])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(relu_out, labels, name='xentropy')
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_op = optimizer.minimize(cross_entropy)
for step in range(10):
	sess.run(train_op, feed_dict={x:np.array([[1.0, 2.0, 3.0]]), labels:answer})
	print(sess.run(softmax, feed_dict={x:np.array([[1.0, 2.0, 3.0]]), labels:answer}))
