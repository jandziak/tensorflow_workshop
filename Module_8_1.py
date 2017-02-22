# Tensorflow workshop with Jan Idziak
#-------------------------------------
#
#script harvested from:
#https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/learn/python/learn
#
# skflow intro clasifiers 
#---------------------------------------
#
import tensorflow.contrib.learn.python.learn as learn
from sklearn import datasets, metrics, preprocessing
import tensorflow as tf
import tensorflow.contrib.layers.python.layers as layers
import tensorflow.contrib.learn.python.learn as learn

#Linear Classifier
iris = datasets.load_iris()
feature_columns = learn.infer_real_valued_columns_from_input(iris.data)
classifier = learn.LinearClassifier(n_classes=3, feature_columns=feature_columns)
classifier.fit(iris.data, iris.target, steps=200, batch_size=32)
iris_predictions = list(classifier.predict(iris.data, as_iterable=True))
score = metrics.accuracy_score(iris.target, iris_predictions)
print("Accuracy: %f" % score)

#Linear Regression
boston = datasets.load_boston()
x = preprocessing.StandardScaler().fit_transform(boston.data)
feature_columns = learn.infer_real_valued_columns_from_input(x)
regressor = learn.LinearRegressor(feature_columns=feature_columns)
regressor.fit(x, boston.target, steps=200, batch_size=32)
boston_predictions = list(regressor.predict(x, as_iterable=True))
score = metrics.mean_squared_error(boston_predictions, boston.target)
print ("MSE: %f" % score)

# Deep Neural Network
iris = datasets.load_iris()
feature_columns = learn.infer_real_valued_columns_from_input(iris.data)
classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3, feature_columns=feature_columns)
classifier.fit(iris.data, iris.target, steps=200, batch_size=32)
iris_predictions = list(classifier.predict(iris.data, as_iterable=True))
score = metrics.accuracy_score(iris.target, iris_predictions)
print("Accuracy: %f" % score)

# #Custom model
iris = datasets.load_iris()

def my_model(features, labels):
  """DNN with three hidden layers."""
  # Convert the labels to a one-hot tensor of shape (length of features, 3) and
  # with a on-value of 1 for each one-hot vector of length 3.
  labels = tf.one_hot(labels, 3, 1, 0)
  # Create three fully connected layers respectively of size 10, 20, and 10.
  features = layers.stack(features, layers.fully_connected, [10, 20, 10])
  # Create two tensors respectively for prediction and loss.
  prediction, loss = (
      tf.contrib.learn.models.logistic_regression(features, labels)
  )
  # Create a tensor for training op.
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
      learning_rate=0.1)
  return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op
classifier = learn.Estimator(model_fn=my_model)
classifier.fit(iris.data, iris.target, steps=1000)
y_predicted = [
  p['class'] for p in classifier.predict(iris.data, as_iterable=True)]
score = metrics.accuracy_score(iris.target, y_predicted)
print('Accuracy: {0:f}'.format(score))

# Deep Neural Network plus save
iris = datasets.load_iris()
feature_columns = learn.infer_real_valued_columns_from_input(iris.data)
classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3, 
    feature_columns=feature_columns, model_dir="tmp/my_model")
classifier.fit(iris.data, iris.target, steps=200, batch_size=32)
iris_predictions = list(classifier.predict(iris.data, as_iterable=True))
score = metrics.accuracy_score(iris.target, iris_predictions)
print("Accuracy: %f" % score)

# By pasting tensorboard --logdir=/tmp/my_model
# into your terminal the session of tensorboard 
# would be created
# Look at your computational graph at:
# http://127.0.1.1:6006