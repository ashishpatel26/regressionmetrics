# Regression Metrics

* Mean Absolute Error - `sklearn, keras`
* Mean Square Error - `sklearn, keras`
* Root Mean Square Error - `sklearn, keras`
* Root Mean Square Logarithmic Error - `sklearn, keras`
* Root Mean Square Logarithmic Error with negative value handle - `sklearn`
* R2 Score - `sklearn, keras`
* Adjusted R2 Score - `sklearn, keras`
* Mean Absolute Percentage Error - `sklearn, keras`
* Mean squared logarithmic Error - `sklearn, keras`
* Symmetric mean absolute percentage error - `sklearn, keras`
* Normalized Root Mean Squared Error - `sklearn, keras`

## Installation

To install the package from the PyPi repository you can execute the following
command:
```sh
pip install regmetrics
```

## Usage

> Usage with scikit learn :

```python
from regmetrics.metrics import *

y_true = [3, 0.5, 2, 7]
y_pred = [2.5, 0.0, 2, -8]


print("R2Score: ",r2(y_true, y_pred))
print("Adjusted_R2_Score:",adj_r2(y_true, y_pred))
print("RMSE:", rmse(y_true, y_pred))
print("MAE:",mae(y_true, y_pred))
print("RMSLE with Neg Value:", rmsle_with_negval(y_true, y_pred))
print("MSE:", mse(y_true, y_pred))
print("MAPE: ", mape(y_true, y_pred))
```
> Usage with Tensorflow keras:

```python
from regmetrics.kerasmetrics import *
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(path="boston_housing.npz", test_split=0.2, seed=113)

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='rmsprop', loss='mse', metrics=[r2, mae, mse, rmse, mape, rmsle, nrmse])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
```
