import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# preprocessing
x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255
x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255

# model definition
import tensorflow as tf

model = keras.Sequential()
model.add(keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="valid", activation="relu", input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="valid", activation="relu"))
model.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding="valid", activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=256, activation="relu"))
model.add(keras.layers.Dense(units=128, activation="relu"))
model.add(keras.layers.Dense(units=10, activation="softmax"))

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])

# training
hist = model.fit(x_train, y_train, epochs=10, batch_size=256, validation_data=(x_test, y_test))

# plotting
plt.plot(hist.history["accuracy"], color="blue", label="accuracy")
plt.plot(hist.history["val_accuracy"], color="red", label="val_accuracy")
plt.show()