import keras
import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt

# plt.imshow(x_train[0])
# plt.show()

# preprocessing
x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255
x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255

# model definition
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding="valid", activation="relu", input_shape=(28, 28, 1)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=128, activation="relu"))
model.add(keras.layers.Dense(units=128, activation="relu"))
model.add(keras.layers.Dense(units=128, activation="relu"))
model.add(keras.layers.Dense(units=10, activation="softmax"))

import tensorflow as tf

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

hist = model.fit(x_train, y_train, epochs=10, batch_size=256, validation_data=(x_test, y_test))

plt.plot(hist.history["accuracy"], color="blue")
plt.plot(hist.history["val_accuracy"], color="red")
plt.show()