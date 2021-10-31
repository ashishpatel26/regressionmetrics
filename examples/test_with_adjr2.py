from regressionmetrics.keras import *

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(path="boston_housing.npz", test_split=0.2, seed=113)

print(x_train.dtype, y_train.dtype, x_test.dtype, y_test.dtype)
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='rmsprop', loss='mae', metrics=[r2, adj_r2])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))