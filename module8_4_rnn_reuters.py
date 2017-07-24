# Module 8: Recurrent Neural Network
# Challenge: RNN on Reuter dataset

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from keras.datasets import reuters
from keras.preprocessing import sequence
from tflearn.data_utils import shuffle

hm_epochs = 5
batch_size = 100
embedding_size = 128
rnn_size = 128
max_features = 20000
max_len = 200

(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=max_features)
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

n_classes = len(np.unique(y_train))
y_train = np.eye(n_classes)[y_train]
y_test = np.eye(n_classes)[y_test]

graph = tf.Graph()

with graph.as_default():
    x = tf.placeholder('int32', [None, max_len])
    y = tf.placeholder('int32')
    keep_prob = tf.placeholder('float32')

    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    embeddings = tf.Variable(tf.random_uniform([max_features, embedding_size], -1.0, 1.0))
    x_embedded = tf.nn.embedding_lookup(embeddings, x)
    x_embedded = tf.unstack(x_embedded, axis=1)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x_embedded, dtype=tf.float32)

    drop = tf.nn.dropout(outputs[-1], keep_prob)
    output = tf.matmul(drop, layer['weights']) + layer['biases']

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(hm_epochs):
        epoch_loss = 0
        (X_train, y_train) = shuffle(X_train, y_train)
        for step in range(int(X_train.shape[0] / batch_size)):
            _, c = sess.run([optimizer, cost], feed_dict={x: X_train[step*batch_size:(step+1)*batch_size],
                                                          y: y_train[step*batch_size:(step+1)*batch_size],
                                                          keep_prob: 0.7})
            epoch_loss += c
        print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

    print('Accuracy:', accuracy.eval({x: X_test, y: y_test, keep_prob: 1}))