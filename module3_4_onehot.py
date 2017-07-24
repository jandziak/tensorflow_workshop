# Module 3: Datasets and Split Datasets

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf

sess = tf.Session()

# One Hot Encoding
a = [0,1,2,1]
num_labels = len(np.unique(a))
b = np.eye(num_labels)[a]
print(b)

# One Hot Decoding
c = tf.argmax(b,axis=1)
print(sess.run(c))




