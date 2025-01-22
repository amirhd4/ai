# dataset
from pprint import pprint

import numpy as np
import pandas as pd

pd.options.display.max_columns = None
pd.options.display.max_rows = None

df = pd.read_csv("files/heart.csv")

# pre-processing
x = df.drop("HeartDisease", axis="columns")
y = df["HeartDisease"]

from sklearn.preprocessing import LabelEncoder

x = x.apply(LabelEncoder().fit_transform)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# morel definition
import tensorflow as tf
import keras

model = keras.Sequential()

model.add(keras.layers.Input(shape=(11,)))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(units=128, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(units=128, activation="relu"))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(units=128, activation="relu"))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(units=128, activation="relu"))
model.add(keras.layers.Dense(units=1, activation="sigmoid"))

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.binary_crossentropy, metrics=["accuracy"])
model.summary()

hist = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_test, y_test))


# plot
import matplotlib.pyplot as plt

plt.plot(hist.history["accuracy"], color="blue", label="accuracy")
plt.plot(hist.history["val_accuracy"], color="red", label="validation_accuracy")
plt.show()

plt.plot(hist.history["loss"], color="blue", label="loss")
plt.plot(hist.history["val_loss"], color="red", label="validation_loss")

