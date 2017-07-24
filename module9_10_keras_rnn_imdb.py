# Module 9 Keras
# RNN Model on IMDB dataaset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, SpatialDropout1D
from keras.layers import LSTM

# Parameters
max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

# Step 1: Pre-process the data
from keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
# print(len(X_train), 'train sequences')
# print(len(X_test), 'test sequences')

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
# print('X_train shape:', X_train.shape)
# print('X_test shape:', X_test.shape)

#Step 2: Build the Network
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(SpatialDropout1D(rate=0.2))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Step 3: Training
model.fit(X_train, y_train, batch_size=batch_size, epochs=2)

#Step 4: Evaluation
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)