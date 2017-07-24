# Module 9 Keras
# RNN Model on MNIST dataaset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM,GRU

batch_size = 32
n_classes = 10
epochs = 20
hidden_units = 10

# Step 1 Preprocess data
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1, 1)
X_test = X_test.reshape(X_test.shape[0], -1, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

# Step 2 Create the Model
model = Sequential()

# Simple RNN Cell
model.add(SimpleRNN(hidden_units,
                    activation='relu',
                    input_shape=X_train.shape[1:]))

# LSTM Cell
# model.add(LSTM(hidden_units,
#                     activation='relu',
#                     input_shape=X_train.shape[1:]))

# GRU Cell
# model.add(GRU(hidden_units,
#                     activation='relu',
#                     input_shape=X_train.shape[1:]))

model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

# Step 3: Training
model.fit(X_train, y_train,batch_size=batch_size, epochs=epochs,)

# Step 4: Evaluation
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])