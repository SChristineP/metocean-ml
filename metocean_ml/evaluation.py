from .spectra_tools import integrated_parameters
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,root_mean_squared_error
import pandas as pd
import numpy as np

def error_metrics_1D(target,pred):
    '''
    Calculate common error metrics for 1D data, such as Mean Squared Error (MSE), 
    Mean Absolute Error (MAE), R-squared (R²), and Root Mean Squared Error (RMSE).
    
    Parameters
    ----------
    target : np.ndarray
        The ground truth (target) values.
    pred : np.ndarray
        The predicted values.

    Returns
    -------
    dict
        A dictionary with the calculated metrics
    '''
    return {
        "mse": mean_squared_error(target, pred),
        "mae": mean_absolute_error(target, pred),
        "r2": r2_score(target, pred),
        "rmse":root_mean_squared_error(target,pred)}

def spec_prediction_performance(spec_pred, spec_target, freq, dir,upsample=1000):
    '''
    Evaluate the prediction performance for spectral data by computing error metrics 
    for each integrated parameter, including direction errors and other spectral errors.
    
    Parameters
    ----------
    spec_pred : np.ndarray
        The predicted spectral data.
    spec_target : np.ndarray
        The target (ground truth) spectral data.
    freq : np.ndarray
        The frequency values corresponding to the spectral data.
    dir : float or np.ndarray
        The direction values corresponding to the spectral data.
    upsample : int, optional
        The upsampling factor for the spectral data, by default 1000.
    Returns
    -------
    params_pred : dict
        A dictionary of the integrated predicted parameters.
    params_target : dict
        A dictionary of the integrated target parameters.
    pd.DataFrame
        A DataFrame containing the calculated error metrics for each parameter.
    
    Notes
    -----
    The function uses the `integrated_parameters` function to compute the integrated
    spectral parameters for both predicted and target data. It also calculates
    the error metrics for directional data separately (by handling the periodic nature of directions).

    '''
    
    def dir_error(target,pred):
        diff = np.abs(target-pred)%360
        diff = np.minimum(diff,360-diff)
        return error_metrics_1D(np.zeros_like(diff),diff)

    params_pred = integrated_parameters(spec_pred,freq,dir,upsample=upsample)
    params_target = integrated_parameters(spec_target,freq,dir,upsample=upsample)
    perf = {}
    for k,pred in params_pred.items():
        target = params_target[k]
        if "dir" in k:
            perf[k] = dir_error(target,pred)
        else:
            perf[k] = error_metrics_1D(target,pred)
    return params_pred, params_target, pd.DataFrame(perf)

