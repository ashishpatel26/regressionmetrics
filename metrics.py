from sklearn.metrics import *
import numpy as np


def mae(y_true, y_pred):
    """
    Mean Absolute Error
    """
    return mean_absolute_error(y_true, y_pred)


def mse(y_true, y_pred):
    """
    Mean Square Error
    """
    return mean_squared_error(y_true, y_pred)
    
def rmse(y_true, y_pred):
    """
    Root Mean Square Error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def rmsle(y_true, y_pred):
    """
    Root Mean Square Logarithmic Error
    """
    for i in range(len(y_true)):
        if y_true[i] < 0 or y_pred[i] < 0:
            continue
    
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def rmsle_with_negval(y_true, y_pred):
    """
    Root Mean Square Logarithmic Error with negative value handle
    """
    sum=0.0
    for x in range(len(y_pred)):
        if y_pred[x]<0 or y_true[x]<0: #check for negative values
            continue
        p = np.log(y_pred[x]+1)
        r = np.log(y_true[x]+1)
        sum = sum + (p - r)**2
    
    RMSLE= (sum/len(y_pred))**0.5
    
    return RMSLE

def r2(y_true, y_pred):
    """
    R Squared
    """
    return r2_score(y_true, y_pred)

def adj_r2(y_true, y_pred):
    """
    Adjusted R Squared formula generate
    """
    return r2_score(y_true, y_pred) - ((1 - r2_score(y_true, y_pred)) * (len(y_true) - 1) / (len(y_true) - len(y_pred) - 1))

def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error
    """
    return mean_absolute_percentage_error(y_true, y_pred)
    


