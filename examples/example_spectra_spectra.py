import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,root_mean_squared_error
from sklearn.decomposition import PCA

from metocean_ml.spectra_tools import integrated_parameters

def spectra_prediction_model(
    spec_origin:xr.DataArray, 
    spec_target:xr.DataArray, 
    target_freq_var:str="frequency", 
    target_dir_var:str="direction",
    model_type:str="linear", 
    validation_data_fraction:float=0.5,
    PCA_reduction:bool=True,
    PCA_components:int=50):

    '''
    This function trains a model to predict target spectra from origin spectra.
    Shared timestamps between spec_origin and spec_target are used to train and validate the model.
    The full timeseries of spec_origin is then used to predict spec_target for the full timeseries.
    
    Parameters
    ----------
    spec_origin : xarray.DataArray 
        Spectra timeseries from the origin
    target_freq_var : str
        the origin spectra frequency variable name
    target_dir_var : str
        the origin spectra direction variable name
    spec_target : xarray.DataArray
        Spectra timeseries from the target location
    model_type : str
        one of ("linear", "ridge", "lasso", "elasticnet", 
        "random_forest", "gradient_boosting", "svr", "mlp")
    validation_data_fraction: float
        Fraction of data to use for validation (performance metric)
    
    Returns
    --------
    predicted_spectra : DataArray of the predicted spectra for all timestamps of spec_origin, 
        with time coordinates from spec_origin, and frequency and direction coordinates from spec_target.
    performance_metrics : the performance of the model when trained and tested on the shared timestamps.
    '''

    # Coordinates
    directions = spec_target[target_dir_var].data
    frequencies = spec_target[target_freq_var].data

    # An intersection of timestamp labels is used to match spectra between datasets.
    timestamps = list(set(spec_origin["time"].data) & set(spec_target["time"].data))
    input_data = spec_origin.loc[timestamps]
    target_data = spec_target.loc[timestamps]
    print("Found",len(timestamps),"timestamps for training from",timestamps[0],"to",timestamps[-1])

    input_data = input_data.data.reshape(input_data.shape[0],-1)
    target_data = target_data.data.reshape(target_data.shape[0],-1)

    # Data split into training and validation
    ind = int((1-validation_data_fraction)*len(timestamps))
    train_input = input_data[:ind]
    train_target = target_data[:ind]
    val_input = input_data[ind:]
    val_target = target_data[ind:]

    # Using PCA dimensionality reduction of input spectra (recommended)
    if PCA_reduction:
        pca = PCA(n_components=PCA_components)
        train_input = pca.fit_transform(train_input)
        val_input = pca.transform(val_input)
    print("Train input:",train_input.shape,"val_input:",val_input.shape,
          "train_target:",train_target.shape,"val_target:",val_target.shape)

    print("Training...")
    # Train model
    model = train_model(
        train_input=train_input,train_target=train_target,
        model_type=model_type)

    print("Testing...")
    # Test model
    val_pred = model.predict(val_input)
    val_pred = np.clip(val_pred, a_min=0, a_max=None)

    # Calculate performance metrics using significant wave height
    predicted_Hs = integrated_parameters(val_pred.reshape(val_pred.shape[0], len(frequencies), len(directions)), frequencies, directions)["Hs"]
    actual_Hs = integrated_parameters(val_target.reshape(val_target.shape[0], len(frequencies), len(directions)), frequencies, directions)["Hs"]

    mse = mean_squared_error(actual_Hs, predicted_Hs)
    mae = mean_absolute_error(actual_Hs, predicted_Hs)
    r2 = r2_score(actual_Hs, predicted_Hs)
    rmse = root_mean_squared_error(actual_Hs,predicted_Hs)
    
    performance_metrics = {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "rmse": rmse
    }

    print(performance_metrics)

    print("Predicting full timeseries...")
    
    # Run model on complete time period
    origin_spectra = spec_origin.data.reshape(spec_origin.shape[0],-1)
    if PCA_reduction:
        origin_spectra = pca.transform(origin_spectra)
    predicted_spectra = model.predict(origin_spectra)
    print("Origin spectra",origin_spectra.shape,"predicted spectra",predicted_spectra.shape,"target",spec_target.shape)
    predicted_spectra = predicted_spectra.reshape((predicted_spectra.shape[0],)+spec_target.shape[-2:])
    predicted_spectra = np.clip(predicted_spectra, a_min=0, a_max=None)
    
    new_coordinates = [spec_origin.coords["time"],spec_target.coords[target_freq_var],spec_target.coords[target_dir_var]]
    predicted_spectra = xr.DataArray(data=predicted_spectra,coords=new_coordinates)
    
    return predicted_spectra, performance_metrics

def train_model(train_input: np.ndarray, train_target: np.ndarray,
                model_type: str = "linear", random_state:int=42):
    '''

    Parameters
    -----------
    train_input : np.ndarray
        Input data
    train_target: np.ndarray
        Target data
    model_type: str
        One of ("linear", "ridge", "lasso", "elasticnet", 
                "random_forest", "gradient_boosting", "svr", "mlp")
    '''
    
    # Reshape inputs for the linear models
    train_input  = train_input.reshape(train_input.shape[0], -1)
    train_target = train_target.reshape(train_target.shape[0], -1)
    
    # Initialize the model
    if model_type == "linear":
        model = LinearRegression()
    
    elif model_type == "ridge":
        model = Ridge()

    elif model_type == "lasso":
        model = Lasso()

    elif model_type == "elasticnet":
        model = ElasticNet()

    elif model_type == "random_forest":
        model = RandomForestRegressor(random_state=random_state)

    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,random_state=random_state)

    elif model_type == "svr":
        model = SVR(kernel="linear", C=100, gamma="auto")

    elif model_type == "mlp":
        model = MLPRegressor(random_state=random_state)

    else:
        raise ValueError(f"Model type '{model_type}' is not recognized.")
    
    # Train the model
    model.fit(train_input, train_target)

    return model
