import tensorflow as tf
# tf.reset_default_graph()
session = tf.InteractiveSession()
import utils

max_length = 50
X, y, index_to_word, sentences = utils.load_sentiment_data(max_length)
X_train, y_train, X_test, y_test = utils.split_data(X, y)
vocab_size = len(index_to_word)
n_classes = y.shape[1]

s_i = 50
print("Sentence:", sentences[s_i])
print("Label:", utils.label_to_desc(y[s_i]))

data_placeholder = tf.placeholder(tf.float32, shape=(None, max_length, vocab_size), name='data_placeholder')
labels_placeholder = tf.placeholder(tf.float32, shape=(None, n_classes), name='labels_placeholder')
keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')

# Helper function for fully connected layers

def linear(input_, output_size, layer_scope, stddev=0.02, bias_start=0.0):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(layer_scope):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        return tf.matmul(input_, matrix) + bias

# Define Computation Graph
n_rnn_layers = 3
n_fc_layers = 2
n_rnn_nodes = 256
n_fc_nodes = 128

print "step 1"

with tf.name_scope("recurrent_layers") as scope:
    # Create LSTM Cell
    cell = tf.nn.rnn_cell.LSTMCell(n_rnn_nodes, state_is_tuple=False)
    cell = tf.nn.rnn_cell.DropoutWrapper(
        cell, output_keep_prob=keep_prob_placeholder)
    stacked_cells = tf.nn.rnn_cell.MultiRNNCell([cell] * n_rnn_layers, state_is_tuple=False)
    output, encoding = tf.nn.dynamic_rnn(stacked_cells, data_placeholder, dtype=tf.float32)

print "step 2"
with tf.name_scope("fc_layers") as scope:
    # Connect RNN Embedding output into fully connected layers
    prev_layer = encoding
    for fc_index in range(0, n_fc_layers-1):
        fci = tf.nn.relu(linear(prev_layer, n_fc_nodes, 'fc{}'.format(fc_index)))
        fc_prev = fci

    fc_final = linear(fc_prev, n_classes, 'fc{}'.format(n_fc_layers-1))

print "step 3"
logits = tf.nn.softmax(fc_final)

# Define Loss Function + Optimizer
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, labels_placeholder))

optimizer = tf.train.GradientDescentOptimizer(0.0002).minimize(loss)
prediction = tf.nn.softmax(logits)
prediction_is_correct = tf.equal(
    tf.argmax(logits, 1), tf.argmax(labels_placeholder, 1))
accuracy = tf.reduce_mean(tf.cast(prediction_is_correct, tf.float32))

# Train loop

num_steps = 20
batch_size = 32
keep_prob_rate = 0.75

tf.initialize_all_variables().run()
print "step 4"
for step in xrange(num_steps):
    offset = (step * batch_size) % (X_train.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = X_train[offset:(offset + batch_size), :, :]
    batch_labels = y_train[offset:(offset + batch_size), :]
    # We built our networking using placeholders. It's like we've made reservations for a party of 6.
    # So use feed_dict to fill what we reserved. And we can't show up with 9 people. 

    feed_dict_train = {data_placeholder: batch_data, labels_placeholder : batch_labels, keep_prob_placeholder: keep_prob_rate}
    # Run the optimizer, get the loss, get the predictions.
    # We can run multiple things at once and get their outputs
    _, loss_value_train, predictions_value_train, accuracy_value_train = session.run(
      [optimizer, loss, prediction, accuracy], feed_dict=feed_dict_train)
    if (step % 2 == 0):
        print "Minibatch train loss at step", step, ":", loss_value_train
        print "Minibatch train accuracy: %.3f%%" % accuracy_value_train
        # feed_dict_test = {data_placeholder: X_test, labels_placeholder: y_test, keep_prob_placeholder: 1.0}
        # loss_value_test, predictions_value_test, accuracy_value_test = session.run(
        #     [loss, prediction, accuracy], feed_dict=feed_dict_test)
        # print "Test loss: %.3f" % loss_value_test
        # print "Test accuracy: %.3f%%" % accuracy_value_test