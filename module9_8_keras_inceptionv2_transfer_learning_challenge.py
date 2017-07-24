# Module 9 Keras
# Challenge InceptionV3 Transfer Learning

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input

import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split
X, Y = oxflower17.load_data(one_hot=True, resize_pics=(224, 224))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Step 1: Create the base pre-trained model
input_tensor = Input(shape=(224, 224, 3))
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

# Step 2: Create a new model with dense and softamx layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(17, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Step 3: Freeze all pre-trained layers and train the top layers with new dataaset
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=100, epochs=2)

# Step 4: Unfreeze some pre-trained layers and train with new dataset
for layer in model.layers[:5]:
    layer.trainable = False
for layer in model.layers[5:]:
    layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
model.fit(X_train, Y_train, batch_size=100, epochs=2)