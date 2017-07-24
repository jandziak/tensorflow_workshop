# Module 6: Tensorboard
# Author: Dr. Alfred Ang

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
logdir = '/tmp/demo/4'

import tensorflow as tf

a = tf.constant(12,name='a')
b = tf.constant(4,name='b')

# c = tf.multiply(a,b,name='c')
# d = tf.div(a, b, name='d')

with tf.name_scope('multiply'):
    c = tf.multiply(a, b, name='c')

with tf.name_scope('divide'):
    d = tf.div(a, b, name='d')


sess = tf.Session()
tf.summary.scalar('c',c)
tf.summary.scalar('d',d)
merged_summary = tf.summary.merge_all()
s = sess.run(merged_summary)

writer = tf.summary.FileWriter(logdir)
writer.add_summary(s)
writer.add_graph(sess.graph)
print(sess.run(c))
print(sess.run(d))