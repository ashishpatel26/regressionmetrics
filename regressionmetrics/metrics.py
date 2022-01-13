from sklearn.metrics import *
import numpy as np

EPSILON = 1e-10

def MeanAbsoErr(y_true, y_pred):
    """
    Mean absolute error regression loss. smaller is better.
    
    interpretation: smaller is better.(Best value is 0.0)
    - MAE value is between 0 to inf.
    - MAE treats larger and small errors equally. Not sensitive to outliers in the data.
    - MAE is Robust to outliers compared to RMSE.
    - 

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: mean absolute error
    """
    return mean_absolute_error(y_true, y_pred)


def MeanSqrtErr(y_true, y_pred):
    """
    Mean squared error regression loss. 
    
    interpretation: smaller is better.(Best value is 0.0)
    - MSE = 0.0, indicates that model is perfectly fitting the data.
    - MSE = inf, indicates that model is overfitting the data.
    - MSE = NaN, indicates that model is not fitting the data.
    - MSE value is between 0 to inf.
    - MSE treats Sensitive to outliers, punishes larger error more.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: mean squared error
    """
    return mean_squared_error(y_true, y_pred)
    
def RootMeanSqrtErr(y_true, y_pred):
    """
    Root Mean Square Error regression loss. 
    
    interpretation: smaller is better.(Best value is 0.0)
    - RMSE = 0.0, indicates that model is perfectly fitting the data.
    - RMSE = inf, indicates that model is overfitting the data.
    - RMSE = NaN, indicates that model is not fitting the data.
    - RMSE value is between 0 to inf.
    - RMSE treats Sensitive to outliers, punishes larger error more.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: mean squared error    
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def RootMeanSqrtLogErr(y_true, y_pred):
    """
    Root Mean Squared Logarithm Error regression loss. 
    
    interpretation: smaller is better.(Best value is 0.0)
    - RMSLE = 0.0, indicates that model is perfectly fitting the data.
    - RMSLE = inf, indicates that model is overfitting the data.
    - RMSLE = NaN, indicates that model is not fitting the data.
    - RMSLE is usually used when you don't want to penalize the large differences in the predicted and the actual values when the predicted 
    and the actual values are big numbers.
    - RMSLE is used when y has long tail distribution, or we are interested in the ratio of true value and predicted value.
    
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


def RootMeanSqrtLogErrNeg(y_true, y_pred):
    """
    Root Mean Squared Logarithmic Error with negative values regression loss.  
    
    interpretation: smaller is better.(Best value is 0.0)

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

def R2CoefScore(y_true, y_pred):
    """
    :math:`R^2` (coefficient of determination) regression score function.

    interpretation: Best possible score is 1.0(perfect prediction), lower values are worse.
    - 0.0 <= r2<= 1.0 is standard value but worst cases are -1.0 and -2.0 or any negative values.
    - r2 > 0.9 is good.
    - 0.7 < r2 < 0.9 indicates high level of correlation.
    - 0.4 < r2 < 0.7 indicates medium level of correlation.
    - r2 < 0.4 indicates low level of correlation.
    - 
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: R2 
    """
    return r2_score(y_true, y_pred)

def AdjR2CoefScore(y_true, y_pred):
    """
    Adjusted R2 regression score function.(Best value is 1.0)

    interpretation: Best possible score is 1.0, lower values are worse.
    - adj_r2 < 1.0 indicates that at least some variability in the data is explained by the model.
    - adj_r2 =0.5 indicates that the variability in the outcome data cannot be explained by the model.
    - adj_r2 < 0.0 indicates that model fitted wrong.
    - Adjusted R-squared to compare the goodness-of-fit for regression models that contain differing numbers of independent variables.
    - Adjusted R2 is the better model when you compare models that have a different amount of variables. 
    - 

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: adjusted R2
    """
    return r2_score(y_true, y_pred) - ((1 - r2_score(y_true, y_pred)) * (len(y_true) - 1) / (len(y_true) - len(y_pred) - 1))

def MeanAbsPercErr(y_true, y_pred):
    """
    Mean absolute percentage error regression loss. 
    
    interpretation: smaller is better.(Best value is 0.0)
    - MAPE < 10 % Very Good
    - 10 % < MAPE < 20% Good
    - 20 % < MAPE < 50% Average
    - MAPE > 50% Not Good
    - MAPE widely used measure for checking forecast accuracy.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: mean absolute percentage error
    """
    return mean_absolute_percentage_error(y_true, y_pred)

def MeanSqrtLogErr(y_true, y_pred):
    """
    Mean squared logarithmic error regression loss. 
    
    interpretation: smaller is better.(Best value is 0.0)
    - MSLE = 0.0, indicates that model is perfectly fitting the data.
    - MSLE = inf, indicates that model is overfitting the data.
    - MSLE used when target, conditioned on the input, is normally distributed, and 
    you don’t want large errors to be significantly more penalized than small ones, 
    in those cases where the range of the target value is large.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: mean squared logarithmic error
    """
    return mean_squared_log_error(y_true, y_pred)

def SymMeanAbsPercErr(y_true: np.ndarray, y_pred: np.ndarray):
    """
    :math:`SMAPE` Symmetric mean absolute percentage error regression loss.

    interpretation: smaller is better.(Best value is 0.0)
    
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: SMAPE
    """
    return np.mean(2.0 * np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) + EPSILON))

def NormRootMeanSqrtErr(y_true, y_pred, type = "minmax"):
    """
    Normalized Root Mean Square Error.
    
    interpretation: smaller is better.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples
        type (str): type of normalization. default is "minmax"
        - sd (standard deviation) : it divides the value by the standard deviation of the data.
        - mean (mean) : it divides the value by the mean of the data.
        - minmax (min-max) : it divides the value by the range of the data.
        - iqr (interquartile range) : it divides the value by the interquartile range of the data.

    Returns:
        [float]: normalized root mean square error
    """
    if type=="sd":
        return np.sqrt(mean_squared_error(y_true, y_pred))/np.std(y_true)
    elif type=="mean":
        return np.sqrt(mean_squared_error(y_true, y_pred))/np.mean(y_true)
    elif type=="minmax":
        return np.sqrt(mean_squared_error(y_true, y_pred))/(np.max(y_true) - np.min(y_true))
    elif type=="iqr":
        return np.sqrt(mean_squared_error(y_true, y_pred))/(np.quantile(y_true, 0.75) - np.quantile(y_true, 0.25))
    elif type!="":
        raise ValueError("type must be either 'sd', 'mean', 'minmax', or 'iqr'")

def NormRootMeanSqrtLogErr(y_true, y_pred):
    """
    Normalized Root Mean Squared Logarithmic Error.
    
    interrpretation: smaller is better.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples
        type (str): type of normalization. default is "minmax"
        - sd (standard deviation) : it divides the value by the standard deviation of the data.
        - mean (mean) : it divides the value by the mean of the data.
        - minmax (min-max) : it divides the value by the range of the data.
        - iqr (interquartile range) : it divides the value by the interquartile range of the data.

    Returns:
        [float]: normalized root mean squared logarithm error
    """
    if type=="sd":
            return np.sqrt(mean_squared_log_error(y_true, y_pred))/np.std(y_true)
    elif type=="mean":
        return np.sqrt(mean_squared_log_error(y_true, y_pred))/np.mean(y_true)
    elif type=="minmax":
        return np.sqrt(mean_squared_log_error(y_true, y_pred))/(np.max(y_true) - np.min(y_true))
    elif type=="iqr":
        return np.sqrt(mean_squared_log_error(y_true, y_pred))/(np.quantile(y_true, 0.75) - np.quantile(y_true, 0.25))
    elif type!="":
        raise ValueError("type must be either 'sd', 'mean', 'minmax', or 'iqr'")

def MedianAbsErr(y_true, y_pred):
    """
    Median absolute error regression loss. (Best value is 0.0)
    
    interpretation: smaller is better.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: median absolute error
    """
    return median_absolute_error(y_true, y_pred)


def MediaRelErr(y_true, y_pred):
    """
    Mean relative error regression loss. 
    
    interpretation: smaller is better.(Best value is 0.0)

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: mean relative error
    """
    return np.mean(np.abs(y_true - y_pred) / (y_true + EPSILON))

def MeanArcAbsPercErr(y_true, y_pred):
    """
    Mean Arctangent Absolute Percentage Error regression loss. 
    
    interpretation: smaller is better.(Best value is 0.0)

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: mean arctangent absolute percentage error
    """
    return np.mean(np.abs(np.arctan(y_true) - np.arctan(y_pred)))


def MeanAbsScaErr(y_true, y_pred, y_train):
    """
    Mean absolute scale error regression loss. 
    
    Reference : R.J. Hyndman, A.B. Koehler, Another look at measures of forecast accuracy, International Journal of Forecasting, 22 (2006), pp. 679-688
    
    interpretation: smaller is better.(Best value is 0.0)
    - MASE = 0.5 means model has doubled the prediction accurracy.
    - MASE > 1.0 means model has overpredicted.
    - MASE < 1.0 means model has underpredicted.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: mean absolute scale error
    """
    e_t = np.mean(np.abs(y_true - y_train))
    scale = mean_absolute_error(y_true[1:] - y_pred[:-1])
    return np.mean(np.abs(e_t/ scale))

def NashSutCoeff(y_true, y_pred):
    """
    The Nash-Sutcliffe efficiency (NSE) is a normalized statistic that determines the relative magnitude of the residual variance compared to the measured data variance (Nash and Sutcliffe, 1970). 
    
    Reference: McCuen, R.H., Knight, Z. and Cutter, A.G., 2006. Evaluation of the Nash–Sutcliffe efficiency index. Journal of hydrologic engineering, 11(6), pp.597-602.
    
    interpretation: Larger is better.(Best possible score is 1.0, lower values are worse.)
    - NSE = 1, corresponds to a perfect match of the model to the observed data. 
    - NSE = 0, indicates that the model predictions are as accurate as the mean of the observed data.
    - Inf < NSE < 0, indicates that the observed mean is a better predictor than the model.
    
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: Nash-Sutcliffe efficiency coefficient
    """
    return 1 - ((np.sum((y_true - y_pred)**2)) / (np.sum((y_true - np.mean(y_true))**2)))


def WillMottIndexAgreeMent(y_true, y_pred):
    """
    Willmott (1981) proposed an index of agreement (d) as a standardized measure of the degree of model prediction error which varies between 0 and 1. 
    
    References : Willmott, C.J., 1981. On the validation of models. Physical geography, 2(2), pp.184-194.
    
    interpretation: Larger is better (Best value is 1.0)
    - d = 1, indicates that the model prediction is perfect match.
    - d = 0, indicates that no agreement at all
    
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: Willmott index of agreement
    """
    residuals = y_true - y_pred
    abs_diff_pred = np.abs(y_pred - np.nanmean(y_true))
    abs_diff_obs  = np.abs(y_true  - np.nanmean(y_true))
    return 1 - np.nansum(residuals**2) / np.nansum((abs_diff_pred * abs_diff_obs)**2)
