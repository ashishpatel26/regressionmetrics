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
