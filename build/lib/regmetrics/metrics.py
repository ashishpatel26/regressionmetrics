from sklearn.metrics import *
import numpy as np

EPSILON = 1e-10

def mae(y_true, y_pred):
    """
    Mean absolute error regression loss.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: mean absolute error
    """
    return mean_absolute_error(y_true, y_pred)


def mse(y_true, y_pred):
    """
    Mean squared error regression loss.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: mean squared error
    """
    return mean_squared_error(y_true, y_pred)
    
def rmse(y_true, y_pred):
    """
    Root Mean Square Error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def rmsle(y_true, y_pred):
    """
    Root Mean Squared Logarithm Error
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: root mean squared logarithm error
    """
    for i in range(len(y_true)):
        if y_true[i] < 0 or y_pred[i] < 0:
            continue
    
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def rmsle_with_negval(y_true, y_pred):
    """
    Root Mean Squared Logarithmic Error with negative values.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: root mean squared logarithm error
    """
    sum=0.0
    for x in range(len(y_pred)):
        if y_pred[x]<0 or y_true[x]<0: #check for negative values
            continue
        p = np.log(y_pred[x]+1)
        r = np.log(y_true[x]+1)
        sum = sum + (p - r)**2
    
    RMSLE = (sum/len(y_pred))**0.5
    
    return RMSLE

def r2(y_true, y_pred):
    """
    :math:`R^2` (coefficient of determination) regression score function.

    Best possible score is 1.0, lower values are worse.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: R2 
    """
    return r2_score(y_true, y_pred)

def adj_r2(y_true, y_pred):
    """
    Adjusted R2 regression score function.

    Best possible score is 1.0, lower values are worse.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: adjusted R2
    """
    return r2_score(y_true, y_pred) - ((1 - r2_score(y_true, y_pred)) * (len(y_true) - 1) / (len(y_true) - len(y_pred) - 1))

def mape(y_true, y_pred):
    """
    Mean absolute percentage error regression loss.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: mean absolute percentage error
    """
    return mean_absolute_percentage_error(y_true, y_pred)

def msle(y_true, y_pred):
    """
    Mean squared logarithmic error regression loss.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: mean squared logarithmic error
    """
    return mean_squared_log_error(y_true, y_pred)

def smape(y_true: np.ndarray, y_pred: np.ndarray):
    """
    :math:`SMAPE` Symmetric mean absolute percentage error regression loss.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: SMAPE
    """
    return np.mean(2.0 * np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) + EPSILON))

def nrmse(y_true, y_pred):
    """
    Normalized Root Mean Square Error.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: normalized root mean square error
    """
    return rmse(y_true, y_pred) / (y_true.max() - y_true.min())

def nrmsle(y_true, y_pred):
    """
    Normalized Root Mean Squared Logarithmic Error.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: normalized root mean squared logarithm error
    """
    return rmsle(y_true, y_pred) / (y_true.max() - y_true.min())


METRICS = {
    'mse': mse,
    'rmse': rmse,
    'nrmse': nrmse,
    'me': me,
    'mae': mae,
    'mad': mad,
    'gmae': gmae,
    'mdae': mdae,
    'mpe': mpe,
    'mape': mape,
    'mdape': mdape,
    'smape': smape,
    'smdape': smdape,
    'maape': maape,
    'mase': mase,
    'std_ae': std_ae,
    'std_ape': std_ape,
    'rmspe': rmspe,
    'rmdspe': rmdspe,
    'rmsse': rmsse,
    'inrse': inrse,
    'rrse': rrse,
    'mre': mre,
    'rae': rae,
    'mrae': mrae,
    'mdrae': mdrae,
    'gmrae': gmrae,
    'mbrae': mbrae,
    'umbrae': umbrae,
    'mda': mda,
}



