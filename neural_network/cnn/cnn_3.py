# Dataset
import keras
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt

# plt.imshow(x_train[10])

# Preprocessing
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255

# Model definition
model = keras.Sequential()
# Conv2D layer + BN + Activation
model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
# Max pool
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(10, activation=keras.activations.softmax))

model.compile(optimizer="adam", loss=keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])

hist = model.fit(x_train, y_train, epochs=10, batch_size=256, validation_data=(x_test, y_test))

plt.plot(hist.history["accuracy"], color="blue", label="accuracy")
plt.plot(hist.history["val_accuracy"], color="red", label="val_accuracy")
plt.show()