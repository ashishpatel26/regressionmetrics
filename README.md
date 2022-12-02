# Regression Metrics

## Installation

To install the package from the PyPi repository you can execute the following
command:
```bash
pip install regressionmetrics
```
If you prefer, you can clone it and run the setup.py file. Use the following commands to get a copy from GitHub and install all dependencies:
```bash
git clone https://github.com/ashishpatel26/regressionmetrics.git
cd regressionmetrics
pip install .
```

| Metrics                     | Full Form                                      | Interpretation                      | Sklearn | Keras |
| --------------------------- | ---------------------------------------------- | ----------------------------------- | ------- | ----- |
| **MeanAbsoErr**                     | Mean Absolute Error                            | Smaller is better (Best value is 0) | ☑️       | ☑️     |
| **MeanSqrtErr**                     | Mean Squared Error                             | Smaller is better(Best value is 0)  | ☑️       | ☑️     |
| **RootMeanSqrtErr**                    | Root Mean Square Error                         | Smaller is better(Best value is 0)  | ☑️       | ☑️     |
| **RootMeanSqrtLogErr**                   | Root Mean Square Log Error                     | Smaller is better(Best value is 0)  | ☑️       | ☑️     |
| **RootMeanSqrtLogErrNeg**       | Root Mean Square Log Error with neg. value     | Smaller is better(Best value is 0)  | ☑️       |       |
| **R2CoefScore**                | coefficient of determination                   | Best possible score is 1            | ☑️       | ☑️     |
| **AdjR2CoefScore**       | Adjusted R2 score                              | Best possible score is 1            | ☑️       | ☑️     |
| **MeanAbsPercErr**                    | Mean Absolute Percentage Error                 | Smaller is better(Best value is 0)  | ☑️       | ☑️     |
| **MeanSqrtLogErr**                    | Mean Squared Logarithm Error                   | Smaller is better(Best value is 0)  | ☑️       | ☑️     |
| **SymMeanAbsPercErr**                   | Symmetric mean absolute percentage error       | Smaller is better(Best value is 0)  | ☑️       |       |
| **NormRootMeanSqrtErr**                   | Normalized Root Mean Square Error.             |                                     | ☑️       | ☑️     |
| **NormRootMeanSqrtLogErr**                  | Normalized Root Mean Squared Logarithmic Error |                                     | ☑️       |       |
| **MedianAbsErr**                | Median Absolute Error                          | Smaller is better(Best value is 0)  | ☑️       |       |
| **MediaRelErr**                     | Mean Relative Error                            | Smaller is better(Best value is 0)  | ☑️       |       |
| **MeanArcAbsPercErr**                   | Mean Arctangent Absolute Percentage Error      | Smaller is better(Best value is 0)  | ☑️       |       |
| **NashSutCoeff**                     | Nash-Sutcliffe Efficiency Coefficient          | Larger is better (Best = 1)         | ☑️       |       |
| **WillMottIndexAgreeMent** | Willmott Index                                 | Larger is better (Best = 1)         | ☑️       |       |

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

Colaboratory File :  [![Open In Colab](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/ashishpatel26/regressionmetrics/blob/main/RegressionMetricsDemo.ipynb)

## Usage

> **Usage with scikit learn :**

```python
from regressionmetrics.metrics import *
y_true = [3, 0.5, 2, 7]
y_pred = [2.5, 0.0, 2, -8]

print("R2Score: ",R2CoefScore(y_true, y_pred))
print("Adjusted_R2_Score:",AdjR2CoefScore(y_true, y_pred))
print("RMSE:", RootMeanSqrtErr(y_true, y_pred))
print("MAE:",MeanAbsoErr(y_true, y_pred))
print("RMSLE with Neg Value:", RootMeanSqrtLogErrNeg(y_true, y_pred))
print("MSE:", MeanSqrtErr(y_true, y_pred))
print("MAPE: ", MeanAbsPercErr(y_true, y_pred))
```
**Output:**

```bash
R2Score:  -8.725067385444744
Adjusted_R2_Score: 20.450134770889484
RMSE: 7.508328708840604
MAE: 4.0
RMSLE with Neg Value: 0.21344354447336292
MSE: 56.375
MAPE:  0.8273809523809523
```

> **Usage with TensorFlow keras:**

```python
try:
  from regressionmetrics.keras import *
except:
  import os
  os.system("pip install regressionmetrics")

from regressionmetrics.keras import *
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
model.compile(optimizer='rmsprop', loss='mse', metrics=[R2CoefScore, 
                                                        MeanAbsoErr, 
                                                        MeanSqrtErr, 
                                                        RootMeanSqrtErr, 
                                                        MeanAbsPercErr, 
                                                        RootMeanSqrtLogErr, 
                                                        NormRootMeanSqrtErr])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
```
**Output**

```bash
Epoch 1/10
13/13 [==============================] - 2s 29ms/step - loss: 461.6622 - R2CoefScore: 0.9004 - MeanAbsoErr: 14.6653 - MeanSqrtErr: 461.6622 - RootMeanSqrtErr: 14.6653 - MeanAbsPercErr: 75.2677 - RootMeanSqrtLogErr: 0.7278 - NormRootMeanSqrtErr: 0.7527 - val_loss: 300.4463 - val_R2CoefScore: 0.8947 - val_MeanAbsoErr: 15.4050 - val_MeanSqrtErr: 300.4463 - val_RootMeanSqrtErr: 15.4050 - val_MeanAbsPercErr: 69.2703 - val_RootMeanSqrtLogErr: 1.2662 - val_NormRootMeanSqrtErr: 0.6927
Epoch 2/10
13/13 [==============================] - 0s 4ms/step - loss: 184.7860 - R2CoefScore: 0.9527 - MeanAbsoErr: 10.9894 - MeanSqrtErr: 184.7860 - RootMeanSqrtErr: 10.9894 - MeanAbsPercErr: 56.5819 - RootMeanSqrtLogErr: 0.5995 - NormRootMeanSqrtErr: 0.5658 - val_loss: 305.9124 - val_R2CoefScore: 0.8910 - val_MeanAbsoErr: 15.4291 - val_MeanSqrtErr: 305.9124 - val_RootMeanSqrtErr: 15.4291 - val_MeanAbsPercErr: 71.9620 - val_RootMeanSqrtLogErr: 1.3943 - val_NormRootMeanSqrtErr: 0.7196
Epoch 3/10
13/13 [==============================] - 0s 5ms/step - loss: 198.5649 - R2CoefScore: 0.9507 - MeanAbsoErr: 12.0198 - MeanSqrtErr: 198.5649 - RootMeanSqrtErr: 12.0198 - MeanAbsPercErr: 62.6733 - RootMeanSqrtLogErr: 0.6901 - NormRootMeanSqrtErr: 0.6267 - val_loss: 80.2263 - val_R2CoefScore: 0.9807 - val_MeanAbsoErr: 7.0446 - val_MeanSqrtErr: 80.2263 - val_RootMeanSqrtErr: 7.0446 - val_MeanAbsPercErr: 43.2890 - val_RootMeanSqrtLogErr: 0.3114 - val_NormRootMeanSqrtErr: 0.4329
Epoch 4/10
13/13 [==============================] - 0s 6ms/step - loss: 197.9205 - R2CoefScore: 0.9613 - MeanAbsoErr: 10.8593 - MeanSqrtErr: 197.9205 - RootMeanSqrtErr: 10.8593 - MeanAbsPercErr: 56.8981 - RootMeanSqrtLogErr: 0.5830 - NormRootMeanSqrtErr: 0.5690 - val_loss: 139.6424 - val_R2CoefScore: 0.9512 - val_MeanAbsoErr: 9.2244 - val_MeanSqrtErr: 139.6424 - val_RootMeanSqrtErr: 9.2244 - val_MeanAbsPercErr: 38.9547 - val_RootMeanSqrtLogErr: 0.5582 - val_NormRootMeanSqrtErr: 0.3895
Epoch 5/10
13/13 [==============================] - 0s 4ms/step - loss: 164.3372 - R2CoefScore: 0.9641 - MeanAbsoErr: 10.6009 - MeanSqrtErr: 164.3372 - RootMeanSqrtErr: 10.6009 - MeanAbsPercErr: 55.5600 - RootMeanSqrtLogErr: 0.5740 - NormRootMeanSqrtErr: 0.5556 - val_loss: 142.1380 - val_R2CoefScore: 0.9564 - val_MeanAbsoErr: 10.7172 - val_MeanSqrtErr: 142.1380 - val_RootMeanSqrtErr: 10.7172 - val_MeanAbsPercErr: 63.0724 - val_RootMeanSqrtLogErr: 0.4243 - val_NormRootMeanSqrtErr: 0.6307
Epoch 6/10
13/13 [==============================] - 0s 5ms/step - loss: 176.5649 - R2CoefScore: 0.9584 - MeanAbsoErr: 11.0135 - MeanSqrtErr: 176.5649 - RootMeanSqrtErr: 11.0135 - MeanAbsPercErr: 56.6267 - RootMeanSqrtLogErr: 0.5719 - NormRootMeanSqrtErr: 0.5663 - val_loss: 217.2575 - val_R2CoefScore: 0.9235 - val_MeanAbsoErr: 12.4566 - val_MeanSqrtErr: 217.2575 - val_RootMeanSqrtErr: 12.4566 - val_MeanAbsPercErr: 55.4557 - val_RootMeanSqrtLogErr: 0.9559 - val_NormRootMeanSqrtErr: 0.5546
Epoch 7/10
13/13 [==============================] - 0s 4ms/step - loss: 157.5359 - R2CoefScore: 0.9567 - MeanAbsoErr: 9.5872 - MeanSqrtErr: 157.5359 - RootMeanSqrtErr: 9.5872 - MeanAbsPercErr: 50.5483 - RootMeanSqrtLogErr: 0.5250 - NormRootMeanSqrtErr: 0.5055 - val_loss: 411.2795 - val_R2CoefScore: 0.8542 - val_MeanAbsoErr: 18.6303 - val_MeanSqrtErr: 411.2795 - val_RootMeanSqrtErr: 18.6303 - val_MeanAbsPercErr: 85.9467 - val_RootMeanSqrtLogErr: 1.6382 - val_NormRootMeanSqrtErr: 0.8595
Epoch 8/10
13/13 [==============================] - 0s 4ms/step - loss: 115.8139 - R2CoefScore: 0.9795 - MeanAbsoErr: 7.9076 - MeanSqrtErr: 115.8139 - RootMeanSqrtErr: 7.9076 - MeanAbsPercErr: 39.5189 - RootMeanSqrtLogErr: 0.3936 - NormRootMeanSqrtErr: 0.3952 - val_loss: 72.1911 - val_R2CoefScore: 0.9813 - val_MeanAbsoErr: 6.7830 - val_MeanSqrtErr: 72.1911 - val_RootMeanSqrtErr: 6.7830 - val_MeanAbsPercErr: 40.6487 - val_RootMeanSqrtLogErr: 0.2993 - val_NormRootMeanSqrtErr: 0.4065
Epoch 9/10
13/13 [==============================] - 0s 5ms/step - loss: 214.5103 - R2CoefScore: 0.9397 - MeanAbsoErr: 10.9144 - MeanSqrtErr: 214.5103 - RootMeanSqrtErr: 10.9144 - MeanAbsPercErr: 56.2217 - RootMeanSqrtLogErr: 0.5520 - NormRootMeanSqrtErr: 0.5622 - val_loss: 87.2555 - val_R2CoefScore: 0.9733 - val_MeanAbsoErr: 6.8626 - val_MeanSqrtErr: 87.2555 - val_RootMeanSqrtErr: 6.8626 - val_MeanAbsPercErr: 28.4989 - val_RootMeanSqrtLogErr: 0.3236 - val_NormRootMeanSqrtErr: 0.2850
Epoch 10/10
13/13 [==============================] - 0s 6ms/step - loss: 159.1116 - R2CoefScore: 0.9662 - MeanAbsoErr: 9.1501 - MeanSqrtErr: 159.1116 - RootMeanSqrtErr: 9.1501 - MeanAbsPercErr: 46.7719 - RootMeanSqrtLogErr: 0.5018 - NormRootMeanSqrtErr: 0.4677 - val_loss: 69.8977 - val_R2CoefScore: 0.9841 - val_MeanAbsoErr: 6.0780 - val_MeanSqrtErr: 69.8977 - val_RootMeanSqrtErr: 6.0780 - val_MeanAbsPercErr: 32.4612 - val_RootMeanSqrtLogErr: 0.2741 - val_NormRootMeanSqrtErr: 0.3246
<keras.callbacks.History at 0x7f78e997f550>
```

:smiley: Thanks for reading and forking.
---
