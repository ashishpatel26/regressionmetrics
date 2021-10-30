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

```bash
Epoch 1/10
 1/13 [=>............................] - ETA: 7s - loss: 1574.7567 - r2: 0.6597 - mae: 37.1803 - mse: 1574.7567 - rmse: 37.1802 - mape: 159.261313/13 [==============================] - 1s 15ms/step - loss: 270.0653 - r2: 0.9472 - mae: 11.5427 - mse: 270.0653 - rmse: 11.5427 - mape: 57.3519 - rmsle: 0.6445 - nrmse: 0.5735 - val_loss: 88.6351 - val_r2: 0.9727 - val_mae: 6.6028 - val_mse: 88.6351 - val_rmse: 6.6028 - val_mape: 29.6502 - val_rmsle: 0.3161 - val_nrmse: 0.2965
Epoch 2/10
 1/13 [=>............................] - ETA: 0s - loss: 74.6623 - r2: 0.9913 - mae: 5.5958 - mse: 74.6623 - rmse: 5.5958 - mape: 25.3655 - rmsl13/13 [==============================] - 0s 3ms/step - loss: 87.1876 - r2: 0.9856 - mae: 6.9466 - mse: 87.1876 - rmse: 6.9466 - mape: 33.4256 - rmsle: 0.3057 - nrmse: 0.3343 - val_loss: 81.7884 - val_r2: 0.9712 - val_mae: 6.6424 - val_mse: 81.7884 - val_rmse: 6.6424 - val_mape: 28.8687 - val_rmsle: 0.3334 - val_nrmse: 0.2887
Epoch 3/10
 1/13 [=>............................] - ETA: 0s - loss: 41.2790 - r2: 0.9722 - mae: 5.3798 - mse: 41.2790 - rmse: 5.3798 - mape: 28.7497 - rmsl13/13 [==============================] - 0s 3ms/step - loss: 103.6462 - r2: 0.9825 - mae: 7.1041 - mse: 103.6462 - rmse: 7.1041 - mape: 34.6278 - rmsle: 0.3231 - nrmse: 0.3463 - val_loss: 71.7539 - val_r2: 0.9769 - val_mae: 6.1455 - val_mse: 71.7539 - val_rmse: 6.1455 - val_mape: 27.5078 - val_rmsle: 0.2893 - val_nrmse: 0.2751
Epoch 4/10
 1/13 [=>............................] - ETA: 0s - loss: 113.6758 - r2: 0.9917 - mae: 6.6575 - mse: 113.6758 - rmse: 6.6575 - mape: 20.8683 - rm13/13 [==============================] - 0s 3ms/step - loss: 88.1601 - r2: 0.9823 - mae: 6.8479 - mse: 88.1601 - rmse: 6.8479 - mape: 32.5867 - rmsle: 0.3080 - nrmse: 0.3259 - val_loss: 63.3707 - val_r2: 0.9829 - val_mae: 6.0845 - val_mse: 63.3707 - val_rmse: 6.0845 - val_mape: 33.1628 - val_rmsle: 0.2747 - val_nrmse: 0.3316
Epoch 5/10
 1/13 [=>............................] - ETA: 0s - loss: 85.8188 - r2: 0.9893 - mae: 7.0097 - mse: 85.8188 - rmse: 7.0097 - mape: 34.8362 - rmsl13/13 [==============================] - 0s 3ms/step - loss: 82.3233 - r2: 0.9860 - mae: 6.5795 - mse: 82.3233 - rmse: 6.5795 - mape: 32.5198 - rmsle: 0.3105 - nrmse: 0.3252 - val_loss: 74.4783 - val_r2: 0.9813 - val_mae: 6.8936 - val_mse: 74.4783 - val_rmse: 6.8936 - val_mape: 41.9492 - val_rmsle: 0.3067 - val_nrmse: 0.4195
Epoch 7/10
 1/13 [=>............................] - ETA: 0s - loss: 105.6430 - r2: 0.9658 - mae: 9.4737 - mse: 105.6430 - rmse: 9.4737 - mape: 53.0854 - rm13/13 [==============================] - 0s 3ms/step - loss: 76.0740 - r2: 0.9856 - mae: 6.4234 - mse: 76.0740 - rmse: 6.4234 - mape: 31.8728 - rmsle: 0.2828 - nrmse: 0.3187 - val_loss: 104.1779 - val_r2: 0.9679 - val_mae: 7.5539 - val_mse: 104.1779 - val_rmse: 7.5539 - val_mape: 30.9401 - val_rmsle: 0.3692 - val_nrmse: 0.3094
Epoch 8/10
 1/13 [=>............................] - ETA: 0s - loss: 100.0114 - r2: 0.9833 - mae: 6.8492 - mse: 100.0114 - rmse: 6.8492 - mape: 27.9621 - rm13/13 [==============================] - 0s 4ms/step - loss: 68.4268 - r2: 0.9892 - mae: 5.9540 - mse: 68.4268 - rmse: 5.9540 - mape: 29.7586 - rmsle: 0.2623 - nrmse: 0.2976 - val_loss: 171.7968 - val_r2: 0.9412 - val_mae: 10.5855 - val_mse: 171.7968 - val_rmse: 10.5855 - val_mape: 47.9010 - val_rmsle: 0.7561 - val_nrmse: 0.4790
Epoch 9/10
 1/13 [=>............................] - ETA: 0s - loss: 291.8670 - r2: 0.9725 - mae: 13.9899 - mse: 291.8670 - rmse: 13.9899 - mape: 61.3658 - 13/13 [==============================] - 0s 3ms/step - loss: 92.3889 - r2: 0.9796 - mae: 6.8932 - mse: 92.3889 - rmse: 6.8932 - mape: 33.2856 - rmsle: 0.3333 - nrmse: 0.3329 - val_loss: 67.2208 - val_r2: 0.9808 - val_mae: 5.8498 - val_mse: 67.2208 - val_rmse: 5.8498 - val_mape: 26.4504 - val_rmsle: 0.2680 - val_nrmse: 0.2645
Epoch 10/10
 1/13 [=>............................] - ETA: 0s - loss: 97.0853 - r2: 0.9923 - mae: 5.9866 - mse: 97.0853 - rmse: 5.9866 - mape: 24.9878 - rmsl13/13 [==============================] - 0s 3ms/step - loss: 78.3823 - r2: 0.9856 - mae: 6.5958 - mse: 78.3823 - rmse: 6.5958 - mape: 32.8136 - rmsle: 0.3025 - nrmse: 0.3281 - val_loss: 69.5314 - val_r2: 0.9787 - val_mae: 6.8302 - val_mse: 69.5314 - val_rmse: 6.8302 - val_mape: 37.3933 - val_rmsle: 0.2974 - val_nrmse: 0.3739
```

:smiley: Thanks for reading and forking.
---
