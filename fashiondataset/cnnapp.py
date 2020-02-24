#!/bin/python

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#标签个数
num_classes = 10
conv1d_kernal = 3


model = keras.models.Sequential()
model.add(keras.layers.Conv1D(32, conv1d_kernal, padding='same', input_shape=(28, 28)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv1D(32, conv1d_kernal))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv1D(64, conv1d_kernal, padding='same'))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(keras.layers.Conv1D(64, conv1d_kernal))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512))
model.add(keras.layers.Activation('relu'))
#model.add(keras.layers.Dense(64, activation='relu'))
#model.add(keras.layers.Conv1D(32, conv1d_kernal))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes, activation = 'softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


print('\nTest accuracy:', test_acc)

predictions = model.predict(test_images)

print(predictions[0])

print("sss ",np.argmax(predictions[0]))

