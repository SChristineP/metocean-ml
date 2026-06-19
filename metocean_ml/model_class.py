import itertools

import pandas as pd
import numpy as np
import xarray as xr
import random
from tqdm import tqdm
from itertools import product
import time
from scipy.integrate import simpson

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,root_mean_squared_error
from sklearn.preprocessing import FunctionTransformer,QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn

from metocean_ml import dataset, trainer, models, preprocessing



class ml_model_class():

    '''
    This class prepares data and runs machine learning models for wave spectrum prediction and integrated parameters prediction.

    It performs the following steps during initialization:
    - Adds a time shift between input and target data if enabled.
    - Transforms the data to follow approximately normal distributions.
    - Splits the data into training, validation, and testing subsets based on time.

    Attributes
    - target_datasets_spec : list of xarray.Dataset, default=None
        Target dataset(s) with 2D directional-frequency wave spectra.

    - neural_network : str, default='MLP'
        Type of neural network:
        - 'MLP' : Multilayer Perceptron
        - 'CNN' : Convolutional Neural Network

    - input_ml_model : str, default='spec'
        Model input configuration:
        - 'spec' : Use spectra as input to predict target spectra.
        - 'specwind' : Use spectra and wind data as input to predict target spectra.
        - 'wind' : Use wind data as input to predict spectra.
        - 'int_params' : Use integrated parameters derived from input and target spectra.

    - input_datasets_spec : list of xarray.Dataset, default=None
        Input dataset(s) containing 2D directional-frequency wave spectra.

    - input_datasets_wind : list of xarray.Dataset, default=None
        Input dataset(s) containing wind speed and direction.

    - hyperparameters : dict, default=None
        Dictionary of model hyperparameters.
        If hyperparameter_tuning is used, values can be provided as lists for grid search.
        If None, default values are used.

    - period : list of str or datetime-like, default=None
        Time selection:
        - [start_time, end_time] : filter between start and end time.
        - [start_time] : select a single timestamp.
        - None : use full available time range.

    - time_shift : bool, default=False
        Controls how temporal alignment between input and target data is handled.

        - False:
            Both input and target data are shifted forward by time_shift_hrs.
            Example: input(00:00) → 01:00, target(00:00) → 01:00
        - True:
            Only the target data is shifted forward by time_shift_hrs.
            Example: input(00:00) → 00:00, target(00:00) → 01:00

        This design allows direct comparison between runs with and without time shifting,
        since the resulting target timestamps are aligned.
        When comparing results between time_shift=True and time_shift=False,
        the same value of time_shift_hrs must be used.

    - time_shift_hrs : int, default=0
        Number of hours used for temporal shifting.

        - When time_shift=False:
            Both input and target data are shifted by this amount.
        - When time_shift=True:
            Only the target data is shifted by this amount.

        If no comparison between shifted and non-shifted runs is required,
        set time_shift=False and time_shift_hrs=0 (no shifting applied).

    - RANDOM_SEED : int, default=42
        Random seed for reproducibility.

    Methods
    - run_model()
        Runs the selected model using the given configuration.
        Returns predictions, training time, test timestamps, trained model, and target test values.

    - hyperparameter_tuning()
        Performs grid search over selected hyperparameter ranges.
        Returns a CSV file summarizing model performance across configurations
        (e.g., based on Taylor diagram metrics).

    Refer to individual methods for more detailed explanations.       
    '''

    def __init__(self, 
                 target_datasets_spec, 
                 neural_network = 'MLP',
                 input_ml_model = 'spec',
                 input_datasets_spec=None, 
                 input_datasets_wind=None, 
                 input_datasets_maps=None,
                 hyperparameters=None, 
                 period=None,
                 time_shift = False,  
                 time_shift_hrs=0,
                 RANDOM_SEED=42):
        super().__init__()

        if target_datasets_spec is not None and not isinstance(target_datasets_spec,list):
            print(f'target_datasets_spec is not a list; wrapping it in a list.')
            target_datasets_spec = [target_datasets_spec]


        if input_datasets_wind is not None and not isinstance(input_datasets_wind,list):
            print(f'input_datasets_wind is not a list; wrapping it in a list.')
            input_datasets_wind = [input_datasets_wind]

        requirements = {
            "spec": ["input_datasets_spec"],
            "int_params": ["input_datasets_spec"],
            "specwind": ["input_datasets_spec", "input_datasets_wind"],
            "wind": ["input_datasets_wind"],
            "specmap": ["input_datasets_spec", "input_datasets_maps"],
        }

        provided = {
            "input_datasets_spec": input_datasets_spec,
            "input_datasets_wind": input_datasets_wind,
            "input_datasets_maps": input_datasets_maps,
        }

        req = requirements.get(input_ml_model)
        if req is None:
            raise ValueError(f"Unknown input_ml_model: {input_ml_model}")

        for key in req:
            if provided[key] is None:
                raise ValueError(f"Need to provide {key} for '{input_ml_model}'.")

        for key in provided:
            setattr(self, key, provided[key] if key in req else None)
        
        
        self.target_datasets_spec = target_datasets_spec
        self.neural_network = neural_network
        self.input_ml_model = input_ml_model
        self.hyperparameters = hyperparameters
        self.period = period
        self.time_shift_hrs = time_shift_hrs
        self.RANDOM_SEED = RANDOM_SEED
        # self.input_datasets_maps = input_datasets_maps

        if input_datasets_spec is None and input_datasets_wind is None:
            raise ValueError('Need to provide input_datasets_spec or input_datasets_wind')
        if target_datasets_spec is None:
            raise ValueError('Need to provide target_datasets_spec')



        activation_funcs = ['relu', 'lrelu', 'sigmoid', 'tanh']   

        # Convert 'activ' to a list/tuple/set and verify all values are valid activation functions.
        activ = self.hyperparameters["activ"]
        activ = activ if isinstance(activ, (list, tuple, set)) else [activ]

        invalid = [a for a in activ if a not in activation_funcs]
        if invalid:
            raise KeyError(f"Invalid activation function {invalid}. Choose from: {activation_funcs}")

        hyperparameters_default_CNN = {
            'no_epochs' : 2,
            'batch_size' : 32,
            'hidden_channels' : 10,
            'kernel_size_conv' : [[3,3], [3,3], [3,3]],
            'kernel_size_pool' : [[3,2], [2], [3,2]],
            'linear_layers' : [[128,128], [128], [128]],
            'activ' : 'relu',
            'dropout' : 0.1,
            'pool' : 'maxpool',
        }

        hyperparameters_default_MLP = {
                'no_epochs' : 10,
                'batch_size' : 32,
                'activ' : 'relu',
                'layers' : [250, 250],
                'dropout' : 0.1
        }

        if neural_network == 'CNN':
            self.hyperparameters_default = hyperparameters_default_CNN
        elif neural_network == 'MLP':
            self.hyperparameters_default = hyperparameters_default_MLP

        # Fill in any missing hyperparameters with their default values, and remove keys that are not in the default. 
        missing_keys = set(self.hyperparameters_default.keys()) - set(self.hyperparameters.keys())

        if self.hyperparameters is None or missing_keys:
                self.hyperparameters = {
                                        **self.hyperparameters_default,
                                        **{k: v for k, v in self.hyperparameters.items() if k in self.hyperparameters_default}
                                    }

        # Standardize the spec datasets to have the same VAR names
        standardized_target_spec = [preprocessing.standardize_wave_dataset(ds) for ds in target_datasets_spec]
        
        if self.input_datasets_spec:
            standardized_input_spec = preprocessing.standardize_wave_dataset(self.input_datasets_spec)


        ################## time handling ##################
        # Use full dataset if period is None          
        if period is None:
            if len(target_datasets_spec) > 1:
                print('Datasets are filtered to match the time range of the first target dataset in the list.')
            # Finds data time range
            data_start = pd.to_datetime(standardized_target_spec[0].time.min().values)
            data_end = pd.to_datetime(standardized_target_spec[0].time.max().values)

            # Uses the full dataset if period is None
            if period == None:
                start_time, end_time = data_start, data_end
            elif period[0] is None:
                start_time = data_start
            elif period[1] is None:
                end_time = data_end
            period = [start_time, end_time]    

        # Perform time shift if enabled
        period[0], period[1] = pd.to_datetime(period[0], format="%Y-%m-%dT%H"), pd.to_datetime(period[1], format="%Y-%m-%dT%H")
        period_time_shift = period.copy()
        period_no_time_shift = period.copy()

        period_no_time_shift[0] = period_no_time_shift[0] + pd.Timedelta(hours=time_shift_hrs)  # Shift spec start time forward by wind_shift hours
        period_time_shift[1] = period_time_shift[1] - pd.Timedelta(hours=time_shift_hrs)        # Shift wind end time backward by the same number of hours to make sure they have the same shape

        # Filters the datasets to the specified time period with time shift
        standardized_target_datasets_time_filtered = [preprocessing.filter_period(ds, period_no_time_shift)[0] for ds in standardized_target_spec]
        print('target time: ', standardized_target_datasets_time_filtered[0].time.values)
        if self.input_datasets_spec:

            if time_shift:
                input_data,_ = preprocessing.filter_period(standardized_input_spec, period_time_shift)
                print('input data time period: ', input_data.time.values)
                print('\n') if not input_datasets_wind else None
            else:
                input_data,_ = preprocessing.filter_period(standardized_input_spec, period_no_time_shift)
                print('input data time period: ', input_data.time.values)
                print('\n') if not input_datasets_wind else None

            input_spec = input_data.SPEC.values
            print('input_spec shape: ', input_spec.shape)

        if self.input_datasets_wind:
            if time_shift:
                datasets_wind_filtered = [preprocessing.filter_period(ds, period_time_shift)[0] for ds in self.input_datasets_wind] 
                print('wind data time: ', datasets_wind_filtered[0].time.values, '\n')
            else:
                datasets_wind_filtered = [preprocessing.filter_period(ds, period_no_time_shift)[0] for ds in self.input_datasets_wind] 
                print('wind data time: ', datasets_wind_filtered[0].time.values, '\n')

        ################## Split into training, validation and test data based on time ##################
        target_data = np.stack([ds.SPEC.values for ds in standardized_target_datasets_time_filtered])
        target_data_reshaped = target_data.swapaxes(1,0)      

        if self.input_datasets_wind:
            wind_features_stack = self.__stack_wind_feaures__(datasets_wind_filtered)
            wind_features_stack_reshaped = wind_features_stack.swapaxes(2,0).swapaxes(1,2)    # Swap axes to (time, loc, wind features)
        
        # For comparison with the other configurations which does not have a shift in time
        time_period,_ = preprocessing.filter_period(standardized_target_spec[0] if isinstance(target_datasets_spec, list) else standardized_target_spec, period)

        if time_shift_hrs>0:
            time_period = time_period.SPEC.values[:-time_shift_hrs].shape[0]
        else:
            time_period = time_period.SPEC.values.shape[0]

        indices = np.arange(time_period)
        print('Time indices: ', indices)

        train_idx = indices[:int(0.8 * len(indices))]
        val_idx = indices[int(0.8 * len(indices)): int(0.9 * len(indices))]
        test_idx = indices[int(0.9 * len(indices)):]

        print('Train indices: ', train_idx)
        print('Val indices', val_idx)
        print('Test indices: ', test_idx, '\n')

        if self.input_datasets_wind and time_shift_hrs and time_shift_hrs>1:       # NEED TO CHECK THIS
            train_idx = train_idx[:-(time_shift_hrs-1)]

        if self.input_datasets_spec:
            X_train_spec = input_spec[train_idx]
            X_val_spec = input_spec[val_idx]
            X_test_spec = input_spec[test_idx]

        if self.input_datasets_wind:
            X_train_wind = wind_features_stack_reshaped[train_idx]
            X_val_wind = wind_features_stack_reshaped[val_idx]
            X_test_wind = wind_features_stack_reshaped[test_idx]

        print('target_data_reshaped shape', target_data_reshaped.shape)
        y_train = target_data_reshaped[train_idx]
        y_val = target_data_reshaped[val_idx]
        self.y_test = target_data_reshaped[test_idx]
        # print('y_test shape', self.y_test.shape)
        
        if self.input_ml_model == 'int_params':

            input_spec = input_data.SPEC.values
            if self.input_ml_model == 'int_params':
                input_int_params = preprocessing.integrated_parameters_dict(input_spec,self.input_datasets_spec.freq.values,self.input_datasets_spec.direction.values, params=['Hs', 'peak_freq', 'peak_dir'])
                input_int_params_stack = np.stack(list(input_int_params[k] for k in ['Hs', 'peak_freq', 'peak_dir_rad'])).swapaxes(1,0)        # Shape (time, int params), make sure that input params and target params are in the same order


            target_int_params_list = []

            for i, spec in enumerate(target_data):
                target_int_params = preprocessing.integrated_parameters_dict(spec,standardized_target_spec[i].freq.values,standardized_target_spec[i].direction.values, params=['Hs', 'peak_freq', 'peak_dir']) # Chooses the freq and dir values of the corresponding target dataset
                target_int_params_list.append(np.stack(list(target_int_params[k] for k in ['Hs', 'peak_freq', 'peak_dir_rad'])))

            target_int_params_stack = np.stack(target_int_params_list).swapaxes(2,0)   # shape (time, int_params, loc)
            target_int_params_stack = target_int_params_stack.swapaxes(2,1)
            target_int_params_stack_rehsaped = target_int_params_stack.reshape(len(target_int_params_list[0][0]), -1) # shape (time, int_params x loc)

            y_train_int = target_int_params_stack_rehsaped[train_idx]
            y_val_int = target_int_params_stack_rehsaped[val_idx]
            y_test_int = target_int_params_stack_rehsaped[test_idx]

            X_train_int = input_int_params_stack[train_idx]
            X_val_int = input_int_params_stack[val_idx]
            X_test_int = input_int_params_stack[test_idx]

        times = standardized_target_datasets_time_filtered[0].time.values if isinstance(standardized_target_datasets_time_filtered, list) else standardized_target_datasets_time_filtered
        time_train = times[train_idx]
        self.time_val = times[val_idx]
        self.time_test = times[test_idx]


        if self.input_datasets_maps:
            maps_stacked = np.stack([ds.elevation.values for ds in self.input_datasets_maps])     

        ################## Transform the data ##################                                                                                   
        tqdm.write('Processing pipelines:')
        no_pbar_updates = 4 if self.input_datasets_spec and self.input_datasets_wind else 2 if self.input_datasets_spec else 3
        pbar = tqdm(total=no_pbar_updates)
        pbar.set_description("Running spectra_pipeline on target_data")

        y_train_freq_dir_combined = y_train.reshape(y_train.shape[0],y_train.shape[1],-1)    # Combines freq and dir
        y_val_freq_dir_combined = y_val.reshape(y_val.shape[0],y_val.shape[1],-1)
        self.y_test_freq_dir_combined = self.y_test.reshape(self.y_test.shape[0],self.y_test.shape[1],-1)

        if self.input_ml_model == 'int_params':
            pbar.set_description("Computing target integration parameters and running integration parameter pipeline.")

            self.pipe_target = self.__quantile_pipeline__()
            y_train_s = self.pipe_target.fit_transform(y_train_int)
            y_val_s = self.pipe_target.transform(y_val_int)
            self.y_test_s = self.pipe_target.transform(y_test_int)
            
            print('scaled training target data shape: ', y_train_s.shape)
            pbar.update(1)

        else:
            self.pipe_target = self.__spectra_pipeline__()
            y_train_reshaped = y_train_freq_dir_combined.reshape(-1, y_train_freq_dir_combined.shape[-1])  
            y_val_reshaped = y_val_freq_dir_combined.reshape(-1, y_val_freq_dir_combined.shape[-1])
            y_test_reshaped = self.y_test_freq_dir_combined.reshape(-1, self.y_test_freq_dir_combined.shape[-1])               # Reshapes the target_data to (time x loc, freq x dir) for data transformation then 
            y_train_s  = self.pipe_target.fit_transform(y_train_reshaped).reshape(y_train_freq_dir_combined.shape)             # reshapes back to (time, loc, freq x dir)
            y_val_s  = self.pipe_target.transform(y_val_reshaped).reshape(y_val_freq_dir_combined.shape)
            self.y_test_s  = self.pipe_target.fit_transform(y_test_reshaped).reshape(self.y_test_freq_dir_combined.shape)

            print('scaled training target data shape', y_train_s.shape)
            pbar.update(1)

        if self.input_datasets_spec:
            if self.input_ml_model == 'int_params':
                pbar.set_description("Computing input integration parameters and running integration parameter pipeline.")
                input_pipeline = self.__quantile_pipeline__()
                X_train_s = input_pipeline.fit_transform(X_train_int)
                X_val_s = input_pipeline.transform(X_val_int)
                X_test_s = input_pipeline.transform(X_test_int)

                if self.neural_network == 'CNN':

                    X_train_s = X_train_s.reshape(X_train_s.shape[0], 1, -1)
                    X_val_s = X_val_s.reshape(X_val_s.shape[0], 1, -1)
                    X_test_s = X_test_s.reshape(X_test_s.shape[0], 1, -1)

            else:
                pbar.set_description("Running spectra_pipeline on input_spec")
                input_pipeline = self.__quantile_pipeline__()
                X_train_s = input_pipeline.fit_transform(X_train_spec.reshape(X_train_spec.shape[0],-1))
                X_val_s = input_pipeline.transform(X_val_spec.reshape(X_val_spec.shape[0],-1))
                X_test_s = input_pipeline.transform(X_test_spec.reshape(X_test_spec.shape[0],-1))

                if self.neural_network == 'CNN':

                    X_train_s = X_train_s.reshape(X_train_spec.shape)
                    X_val_s = X_val_s.reshape(X_val_spec.shape)
                    X_test_s = X_test_s.reshape(X_test_spec.shape)

            print('scaled training input shape: ', X_train_s.shape)


            pbar.update(1)

        if self.input_datasets_wind:
            pbar.set_description("Running wind_speed_pipeline on input_wind_speed")
            pipe_wind_speed = self.__wind_speed_pipeline__()
            X_train_wind_speed_s = pipe_wind_speed.fit_transform(X_train_wind[:,:,0])
            X_val_wind_speed_s = pipe_wind_speed.transform(X_val_wind[:,:,0])
            X_test_wind_speed_s = pipe_wind_speed.transform(X_test_wind[:,:,0])

            pbar.update(1)

            pbar.set_description("Running wind_dir_pipeline on input_wind_dir")
            pipe_wind_dir = self.__quantile_pipeline__()
            X_train_wind_dir_s = pipe_wind_dir.fit_transform(X_train_wind[:,:,[1,2]].reshape(X_train_wind.shape[0], -1))
            X_val_wind_dir_s = pipe_wind_dir.transform(X_val_wind[:,:,[1,2]].reshape(X_val_wind.shape[0], -1))
            X_test_wind_dir_s = pipe_wind_dir.transform(X_test_wind[:,:,[1,2]].reshape(X_test_wind.shape[0], -1))

            pbar.update(1)
            pbar.close()

            X_train_wind_s = np.concatenate([X_train_wind_speed_s, X_train_wind_dir_s], axis=1) # Shape (time, wind_speed + wind_dir)
            X_val_wind_s = np.concatenate([X_val_wind_speed_s, X_val_wind_dir_s], axis=1)
            X_test_wind_s = np.concatenate([X_test_wind_speed_s, X_test_wind_dir_s], axis=1)

            if self.neural_network == 'CNN':
                X_train_wind_s = X_train_wind_s.reshape(X_train_wind.shape)  # Reshape to (time, no. of wind inputs, no. of wind features)
                X_val_wind_s = X_val_wind_s.reshape(X_val_wind.shape)
                X_test_wind_s = X_test_wind_s.reshape(X_test_wind.shape)

            print('X_train_wind_s shape: ', X_train_wind_s.shape)

        tqdm.write('Done processing pipelines.\n')


        ################## Datasets for training, validation and testing ##################

        if self.neural_network == 'CNN':
            print('X_train_s shape: ', X_train_s.shape) if self.input_datasets_spec else None
            print('X_train_wind_s shape: ', X_train_wind_s.shape) if self.input_datasets_wind else None
            print('y_train_s shape: ', y_train_s.shape)

            self.train_dataset = dataset.CNN_Dataset(target=y_train_s, 
                                                        X_spec=X_train_s if self.input_datasets_spec else None, 
                                                        X_wind=X_train_wind_s if self.input_datasets_wind else None, 
                                                        X_map=maps_stacked if self.input_datasets_maps else None, 
                                                        loc=wind_features_stack_reshaped.shape[1] if self.input_datasets_wind else y_train.shape[1])  ###### CHANGE MAP FEATURES HERE
            self.val_dataset = dataset.CNN_Dataset(target=y_val_s, 
                                                    X_spec=X_val_s if self.input_datasets_spec else None, 
                                                    X_wind=X_val_wind_s if self.input_datasets_wind else None, 
                                                    X_map=maps_stacked if self.input_datasets_maps else None, 
                                                    loc=wind_features_stack_reshaped.shape[1] if self.input_datasets_wind else y_train.shape[1])
            self.test_dataset = dataset.CNN_Dataset(target=self.y_test_s, 
                                                    X_spec=X_test_s if self.input_datasets_spec else None, 
                                                    X_wind=X_test_wind_s if self.input_datasets_wind else None, 
                                                    X_map=maps_stacked if self.input_datasets_maps else None, 
                                                    loc=wind_features_stack_reshaped.shape[1] if self.input_datasets_wind else y_train.shape[1])
        
        elif self.neural_network == 'MLP' or self.input_ml_model == 'int_params':
            
            if self.input_datasets_spec and self.input_datasets_wind and self.neural_network == 'MLP':
                X_train = np.concatenate([X_train_s, X_train_wind_s], axis=1)
                X_val = np.concatenate([X_val_s, X_val_wind_s], axis=1)
                X_test = np.concatenate([X_test_s, X_test_wind_s], axis=1)

            elif self.input_datasets_spec:
                X_train = X_train_s
                X_val = X_val_s
                X_test = X_test_s
            elif self.input_datasets_wind:
                X_train = X_train_wind_s
                X_val = X_val_wind_s
                X_test = X_test_wind_s

            print('X_train shape: ', X_train.shape) 
            print('y_train shape: ', y_train_s.shape)

            self.train_dataset = dataset.MLP_Dataset(X_train, y_train_s)
            self.val_dataset = dataset.MLP_Dataset(X_val, y_val_s)
            self.test_dataset = dataset.MLP_Dataset(X_test, self.y_test_s)

        
        print(f'Train dataset length: {len(self.train_dataset)}, validation dataset length: {len(self.val_dataset)}, and test dataset length: {len(self.test_dataset)} \n')



    def run_model(self, hyperparam_tuning=False, target_data=False, **overrides):

        '''
        Runs the ML model and generates spectra or integrated parameter predictions.

        Parameters
        - hyperparameter_tuning : bool, default=False
            Enables hyperparameter tuning. If True, the validation dataset is used
            instead of the test dataset to evaluate model performance.

        - target_data : bool, default=False
            If True, returns the validation/test target dataset values together with
            the predictions.

        - **overrides
            Optional keyword arguments that override hyperparameter values.
            Used to run the model with specific settings during hyperparameter tuning.

        Returns
        - if hyperparameter_tuning=True
            pred_values : numpy.ndarray
            training_time : float

        - if target_data=True
            pred_values : numpy.ndarray
            training_time : float
            self.time_test : numpy.ndarray
            model : metocean_ml.models.FNN or metocean_ml.models.CNN
            target_test_values : numpy.ndarray

        - otherwise
            pred_values : numpy.ndarray
            training_time : float
            self.time_test : numpy.ndarray
            model

        '''
        
        if not hyperparam_tuning:
            print('Hyperparameters: ', self.hyperparameters)
        
        # Set hyperparameters to tuned values if provided; otherwise use defaults.
        for key, default_value in self.hyperparameters_default.items():
            if hyperparam_tuning:
                value = self.__get_param__(key, **overrides)
                self.hyperparameters[key] = value
            else:
                value = self.hyperparameters[key]

            # Ensure hyperparameters have the correct type and format.
            if not isinstance(value, type(default_value)):
                print(f'{key} in hyperparameters dict must be of type {type(default_value).__name__}, got {type(value).__name__}. Changed the type to {type(default_value).__name__}.')
                
                if not hyperparam_tuning:
                    if isinstance(value, list) and len(value) == 1:
                        value = value[0]
                
                target_type = type(default_value)
                self.hyperparameters[key] = target_type(value)
            

        BATCH_SIZE = self.hyperparameters['batch_size']
        train_dataloader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(self.val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_dataloader = DataLoader(self.test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
        print(f"Length of test dataloader: {len(val_dataloader)} batches of {BATCH_SIZE}")
        print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE} \n")

        ################## Model setup ##################
        if self.neural_network == 'CNN':
            print('Running CNN model')
            X_spec, X_wind, X_map, label= next(iter(train_dataloader))

            if self.input_datasets_spec:
                print(f'X_spec shape: {X_spec.shape}')
            if self.input_datasets_wind:
                print(f'X_wind shape: {X_wind.shape}')
            # if self.input_datasets_maps:
            #     print(f'X_map shape: {X_map.shape}')

            print(f"Target shape: {label.shape}")

            # If spec, wind or map datasets are not provided they will not be included in the ml-model
            spec = False if X_spec.shape[1]==0 else True
            map = False if X_map.shape[1]==0 else True
            wind = False if X_wind.shape[1]==0 else True

            model = models.CNN(input_channel=1,
                            output_shape=int(np.prod(self.train_dataset.y_flat_shape)),
                            target_data_shape = self.y_test_freq_dir_combined.shape[-1],
                            input_spec_shape= X_spec.shape,
                            input_wind_shape = X_wind.shape,
                            input_map_shape = X_map[0:1,:,:].shape if map else X_map.shape,
                            hidden_channels= self.hyperparameters['hidden_channels'],
                            kernel_size_conv = self.hyperparameters['kernel_size_conv'],
                            kernel_size_pool = self.hyperparameters['kernel_size_pool'],
                            linear_layers = self.hyperparameters['linear_layers'],
                            activ = self.hyperparameters['activ'],
                            dropout = self.hyperparameters['dropout'],
                            no_loc_target = len(self.target_datasets_spec), 
                            input_spec=spec,
                            input_map = map,
                            input_wind=wind) 
            print(model)
        
        elif self.neural_network == 'MLP':
            print('Running model with MLP neural network.')
            X_batch, label = next(iter(train_dataloader)) 
            print('X_batch shape: ', X_batch.shape)
            print('Target shape: ', label.shape)
            # print(int(np.prod(self.train_dataset.y_flat_shape)))
            model = models.FNN(
                input_size=X_batch.shape[-1],
                output_size=int(np.prod(self.train_dataset.y_flat_shape)),
                layers=self.hyperparameters['layers'],
                dropout_rate=self.hyperparameters['dropout'],
                activ=self.hyperparameters['activ']
            )
            print(model)


        # have loss_fn and optimizer as hyperparameter?  Add this to the function before uploading code
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Selects evaluation dataset: uses validation set during hyperparameter tuning, and test set for final model evaluation.
        if hyperparam_tuning:
            eval_dataloader = val_dataloader
            print('Using validation dataloader (hyperparameter tuning mode)')
        else:
            eval_dataloader = test_dataloader
            print('Using test dataloader (evaluation mode)')

        ################## Training the ml-model ##################
        training_start_time = time.time()
        pbar = tqdm(range(self.hyperparameters['no_epochs']), desc='Running training loop')

        for epoch in pbar:
            if self.neural_network == 'CNN':
                train_loss = trainer.training_step_multiple_input_shapes(model=model, loss_fn=loss_fn, optimizer=optimizer, train_dataloader=train_dataloader)
                _,test_loss = trainer.test_step_multiple_input_shapes(model=model, loss_fn=loss_fn, test_dataloader=eval_dataloader, return_mode='test_loss')
            elif self.neural_network == 'MLP':
                train_loss = trainer.training_step(model=model, loss_fn=loss_fn, optimizer=optimizer, train_dataloader=train_dataloader)
                _,test_loss = trainer.test_step(model=model, loss_fn=loss_fn, test_dataloader=eval_dataloader, return_mode='test_loss')
            
            pbar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "test_loss": f"{test_loss:.4f}"
            })
    
        training_end_time = time.time()
        training_time = training_end_time - training_start_time

        ################## Make predictions ##################
        if self.neural_network == 'CNN':
            y_preds,_ = trainer.test_step_multiple_input_shapes(model=model, loss_fn=loss_fn, test_dataloader=eval_dataloader, return_mode='y_preds')
            y_preds = y_preds.numpy()

        elif self.neural_network == 'MLP':
            y_preds,_ = trainer.test_step(model=model, loss_fn=loss_fn, test_dataloader=eval_dataloader, return_mode='y_preds')
            y_preds = y_preds.numpy()

        y_preds_reshaped = y_preds.reshape(self.y_test_s.reshape(-1, self.y_test_s.shape[-1]).shape) #Reshape to (time x loc, freq x dir)
  
        # Inverse transform predictions to the original scale.
        pred_values = self.pipe_target.inverse_transform(y_preds_reshaped)
        
        if self.input_ml_model == 'int_params':
            pred_values = pred_values.reshape(pred_values.shape[0], len(self.target_datasets_spec), -1).swapaxes(0,1) # (loc, time, int_params)

        else:
            pred_values = pred_values.reshape(self.y_test.shape[1], self.y_test.shape[0], self.y_test.shape[2], self.y_test.shape[3])
            # pred_values = pred_values.swapaxes(0,1)

        # target_test_values = self.pipe_target.inverse_transform(self.y_test_s.reshape(-1, self.y_test_s.shape[-1])).reshape(self.y_test.shape)
        # target_test_values = target_test_values.reshape(self.y_test.shape[1], self.y_test.shape[0], self.y_test.shape[2], self.y_test.shape[3])

        print('\nTest dataset time: ', self.time_test)
        

        if hyperparam_tuning: 
            return pred_values, training_time
        
        elif target_data:
            target_test_values = self.y_test.swapaxes(0,1)
            print('Target test values shape: ', target_test_values.shape)
            return pred_values, training_time, self.time_test, model, target_test_values
        
        else:
            return pred_values, training_time, self.time_test, model


    def hyperparameter_tuning(self, sorting_metric=['Hs_corr_loc_0'], ascending=False, int_params=['Hs', 'peak_freq', 'peak_dir'], output_file='ML_model_hyperparameter_tuning.csv'):
        '''
        Perform hyperparameter tuning of the ML model using grid search.

        For each hyperparameter combination, wave spectra are predicted and
        compared against target spectra. From these spectra, integrated wave
        parameters (e.g., significant wave height, peak frequency, peak direction,
        mean frequency, and mean direction) can be derived depending on the
        `int_params` argument. If the model uses integrated parameters directly
        as inputs and targets, no spectral derivation is performed.

        Performance metrics, including correlation, normalized standard deviation,
        and centered root mean square error, are computed between predictions
        and targets for each parameter combination. Results are iteratively saved
        to a CSV file to ensure progress is preserved in case of interruption.

        After all combinations have been evaluated, the CSV file is sorted based
        on the specified metric(s).

        Parameters
        - sorting_metric : list of str, default=['Hs_corr_loc_0']
            Metric(s) used to sort the final results in the CSV file.

        - ascending : bool, default=False
            If True, results are sorted in ascending order; otherwise in descending order.

        - int_params : list of str, default=['Hs', 'peak_freq', 'peak_dir']
            Integrated parameters to derive from predicted and target spectra
            for evaluation.

        - output_file : str, default='ML_model_hyperparameter_tuning.csv'
            Name of the CSV file where results are stored.S

        Returns
        - None
            Results are written to the specified CSV file, including integrated
            parameters and their corresponding evaluation metrics.
        '''

      
        # Ensure that the hyperprameters are in the right format.
        for key, default_value in self.hyperparameters_default.items():
            value = self.hyperparameters.get(key)

            if not isinstance(value, list):
                self.hyperparameters[key] = [value]                                                       
            
            if key=='kernel_size_conv' or key=='kernel_size_pool' or key=='linear_layers':
                if isinstance(value, list) and (len(value) == 0 or not all(isinstance(x, list) for x in value[0])):     
                    self.hyperparameters[key] = [value]

        print("Performing grid search with the following hyperparameters:")
        print(self.hyperparameters)


        # Creates a dataframe to store the results of each run
        metric_types = ['corr', 'std', 'rmse']#, 'acc']
        metric_columns = [
            f'{param}_{metric}_loc_{i}'
            for metric in metric_types
            for param in int_params
            for i in range(len(self.target_datasets_spec))]
        results_df = pd.DataFrame(columns=list(self.hyperparameters.keys()) + metric_columns + ['Training time [min]'])


        # Perform a grid search over all combinations of the given hyperparameters.
        for hyperparam_values in product(*self.hyperparameters.values()):
            hyperparam_tuning_config_dict = dict(zip(self.hyperparameters.keys(), hyperparam_values))
            print(f"Running config with {hyperparam_tuning_config_dict}")

            y_preds, training_time = self.run_model(**hyperparam_tuning_config_dict, hyperparam_tuning=True, target_data=False)

            # Dict with keys corresponding to the columns in the results df.
            config_dict = {k: [] for k in metric_columns}

            # Looping through all target locations and adding the metric values for each location to the config dict.
            for i in tqdm(range(len(self.target_datasets_spec)), desc='Evaluating configuration : '):
                if self.input_ml_model == 'int_params':
                    params_pred = {}
                    params_target = {}

                    for param in range(y_preds.shape[-1]):
                        if int_params[param] == 'peak_dir':
                            params_pred[int_params[param]] = np.rad2deg(y_preds[i,:,param])
                            # print('target_val',self.target_val.shape)                   # shape (time, int_params, loc)
                            params_target[int_params[param]] = np.rad2deg(self.target_val[:,i,param]) 
                        else:
                            params_pred[int_params[param]] = y_preds[i,:,param]
                            # print('target_val',self.target_val.shape)                   # shape (time, int_params, loc)
                            params_target[int_params[param]] = self.target_val[:,i,param] 

                else:
                    params_pred,params_target,errormatrix = preprocessing.spec_prediction_performance(spec_pred=y_preds[i,:,:], 
                                                                                        spec_target=self.target_val[i,:,:], 
                                                                                        freq=self.target_datasets_spec[0].frequency.values, 
                                                                                        dir=self.target_datasets_spec[0].direction.values,
                                                                                        int_params=int_params)


                df_target = preprocessing.Dataset_to_DF(params_target, 'target ', self.time_val)
                df_predicted = preprocessing.Dataset_to_DF(params_pred, 'predicted ', self.time_val)
                df_combined = (df_target.merge(df_predicted, left_index=True, right_index=True))

                for param in int_params:
                    std, ccf, rmse_centered = preprocessing.taylor_diagram(df_combined,var_ref=[f'target {param}'],var_comp=[f'predicted {param}'],norm_std=True)
                    config_dict[f'{param}_corr_loc_{i}'].append(float(ccf[-1]))
                    config_dict[f'{param}_std_loc_{i}'].append(float(std[-1]))
                    config_dict[f'{param}_rmse_loc_{i}'].append(float(rmse_centered[-1]))

            # Add the metric values from the run to the result dataframe
            new_row = {**hyperparam_tuning_config_dict, **config_dict, 'Training time [min]': round(training_time / 60, 2)}
            if results_df.empty:
                results_df = pd.DataFrame([new_row])
            else:
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

            # Save the results DataFrame to a CSV after each run to ensure all completed configurations are safely stored, even if the script is interrupted.
            results_df.to_csv(output_file, na_rep="NaN") 
            print(f'Results saved in {output_file}\n')

        results = results_df.sort_values(by=sorting_metric, ascending=ascending) 
        results.to_csv(output_file, na_rep="NaN") 


    def __get_param__(self, key, **overrides):
        if key in overrides:
            return overrides[key]
        return self.hyperparameters.get(key, self.hyperparam_tuning_config_dict.get(key))


    def __spectra_pipeline__(self):
        return Pipeline([
            ("log-transform", FunctionTransformer(self.__log_transform__, self.__log_inverse_transform__, check_inverse=False)),
            ("Quantile-transform", QuantileTransformer(
                n_quantiles=1000,
                output_distribution="normal",
                subsample=10000
            )),
        ])

    def __wind_speed_pipeline__(self):
        return Pipeline([
                ("log-transform",FunctionTransformer(self.__log_transform__,self.__log_inverse_transform__,check_inverse=False)),
                ('Quantile-transform', QuantileTransformer(n_quantiles=1000,output_distribution="normal",subsample=10000)),
            ])

    def __quantile_pipeline__(self):
        return Pipeline([
            ('Quantile-transform', QuantileTransformer(n_quantiles=1000,output_distribution="normal",subsample=10000)),
        ])


    def __log_transform__(self, x):
        if np.any(x<0):raise ValueError()
        return np.log(x+1e-30)
    def __log_inverse_transform__(self, x):
        return np.exp(x)-1e-30


    def __stack_wind_feaures__(self, datasets_wind):
        wind_features = []

        for dataset in datasets_wind:
            dataset = dataset.sel(height=10)
            u_wind = -dataset.wind_speed * np.sin(np.deg2rad(dataset.wind_direction))
            v_wind = -dataset.wind_speed * np.cos(np.deg2rad(dataset.wind_direction))

            wind_features.append(
                np.stack([
                    dataset.wind_speed.values,
                    u_wind.values,
                    v_wind.values
                ])
            )
        wind_features = np.stack(wind_features)
        return wind_features

