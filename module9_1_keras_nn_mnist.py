# Module 9 Keras
# NN Model on MNIST dataaset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
n_features = 784
n_classes = 10
learning_rate = 0.5
training_epochs = 2
batch_size = 100

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Step 1: Pre-process the  Data
# from tflearn.datasets import mnist
# X_train, y_train, X_test, y_test = mnist.load_data(one_hot=True)

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

# Step 2: Build the  Network
L1 = 200
L2 = 100
L3 = 60
L4 = 30
model = Sequential()
model.add(Dense(L1, input_dim=n_features, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(L2, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(L3, activation='relu'))
model.add(Dense(L4, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Step 3: Training
model.fit(X_train, y_train, epochs=training_epochs,batch_size=batch_size)

# Step 4: Evaluation
score = model.evaluate(X_test, y_test, batch_size=batch_size)
print("\nTraining Accuracy = ",score[1],"Loss",score[0])

