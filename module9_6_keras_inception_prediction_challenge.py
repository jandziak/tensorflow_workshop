# Module 9 Keras
# Challenge: Inception Demo

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np

model = InceptionV3(weights='imagenet', include_top=True, input_tensor=None, input_shape=None)
print(model.summary())
# img_path = 'images/merlion-299.jpg'
# img = image.load_img(img_path, target_size=(299, 299))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# preds = model.predict(x)
# top3 = decode_predictions(preds,top=3)[0]
#
# predictions = [{'label': label, 'description': description, 'probability': probability * 100.0}
#                     for label,description, probability in top3]
#
# print(predictions)