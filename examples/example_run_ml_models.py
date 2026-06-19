from datetime import datetime, timedelta
from metocean_ml import model_class
import xarray as xr
import pandas as pd
import numpy as np
import json

'''
Example script demonstrating how to use the ml_model_class defined in model_class.py.

The script preprocesses input data (e.g., normalization toward a normal distribution) and generates predictions using
a specified machine learning model and input configuration.

Predictions can be performed for a single location or multiple locations.

Parameters
----------
location : str, default='Sulafjorden'
    Location(s) for prediction.
    - 'Vartdalsfjorden'              : predict for a single location
    - 'Sulafjorden_Vartdalsfjorden'  : train and predict for multiple locations

input_ml_model : str, default='spec'
    Defines the model input/output configuration.
    Options:
    - 'spec'     : spectra → spectra
    - 'specwind' : spectra + wind → spectra
    - 'wind'     : wind → spectra
    - 'int_params'      : integrated parameters → integrated parameters

neural_network : str, default='MLP'
    Neural network architecture.
    Options:
    - 'MLP' : Multilayer Perceptron
    - 'CNN' : Convolutional Neural Network

mode : str, default='run_model'
    Execution mode.
    Options:
    - 'run_model'              : run the model, generate predictions, and save
                                results to a NetCDF file
    - 'hyperparameter_tuning' : perform grid search to identify optimal
                                hyperparameters

period : tuple or None, default=None
    Time range used to filter the dataset.
    - (start_time, end_time) : filter between start and end
    - (start_time,)         : select a single timestamp
    - None                  : use the full available time range

    start_time and end_time may be strings or datetime-like objects.

target_data : bool, default=False
    If True, returns the target (ground truth) data as a NetCDF file aligned
    with the prediction timestamps.

time_shift_hrs : int, default=0
    Number of hours by which the target data is shifted relative to the input data.

hyperparameters : dict, optional
    Hyperparameter configuration. Defaults depend on the selected neural
    network and input configuration.

    MLP:
    - linear_layers : list[int]
        Defines the fully connected architecture. The length of the list
        determines the number of layers, and each value specifies the number
        of neurons in that layer.

    CNN:
    - kernel_size_conv : list[list[int]]
        Convolution kernel sizes for each branch (e.g., spectra, wind, map).
        Each sublist represents a sequence of convolutional layers; its length
        defines the number of layers, and each value specifies the kernel size.

    - kernel_size_pool : list[list[int]]
        Pooling configuration for each branch.
        Each sublist defines pooling applied after convolutional blocks.
        A convolutional block consists of all layers defined in
        `kernel_size_conv` for that branch.

        If a sublist contains multiple values, the corresponding convolutional
        block is repeated, with pooling applied after each repetition.

    - linear_layers : list[list[int]]
        Fully connected layers applied after convolution for each branch.
        Each sublist defines the number and size of layers for the
        corresponding branch.
'''

## Next time: Add hindcast, add windmap, add only map

########## Optional parameters ##########
location = 'Vartdalsfjorden' 
input_ml_model = 'spec'
neural_network = 'CNN'
mode = 'run_model'
period = None
target_data = False
time_shift_hrs = 0


########## Upload data ##########
input_spec = xr.open_dataset('/data/NORA3_wave_spec_lon5.6591934_lat62.7611818_20170101_20230228.nc')
target_Sulafjorden = xr.open_dataset('/data/NORAC_wave_spec_lon6.076_lat62.4_20170101_20230228.nc')
target_Vartdalsfjorden = xr.open_dataset('/data/NORAC_wave_spec_lon6.0189248_lat62.2958587_20170101_20230228.nc')

wind_f1_Sulafjorden = xr.open_dataset('/data/NORA3_wind_sub_lon5.778573_lat62.515031_20170101_20230228.nc')
wind_f2_Sulafjorden = xr.open_dataset('/data/NORA3_wind_sub_lon6.307565_lat62.380851_20170101_20230228.nc')
wind_f1_Vartdalsfjorden = xr.open_dataset('data/NORA3_wind_sub_lon6.189561_lat62.363432_20170101_20230228.nc')
wind_f2_Vartdalsfjorden = xr.open_dataset('/data/NORA3_wind_sub_lon5.913653_lat62.226543_20170101_20230228.nc')

map_Sulafjorden = xr.load_dataset(r'/data/gebco_2025_n62.45_s62.3_w6.0_e6.15.nc')
map_Vartdalsfjorden = xr.load_dataset(r'/data/gebco_2025_n62.35_s62.2_w5.95_e6.1.nc')



########## Output file names ########## 
if period is None:
        period_start, period_end = pd.to_datetime(target_Sulafjorden.time.min().values).strftime("%Y-%m-%d"), pd.to_datetime(target_Sulafjorden.time.max().values).strftime("%Y-%m-%d")
else:
        period_start, period_end = period[0], period[1]

output_file = f'{mode}_{neural_network}_{input_ml_model}_{location}_({period_start} to {period_end})_test'
output_file_target = f'Target_spec_{location}_{period}.nc'

########## Hyperparameters ##########
if mode == 'run_model':
        if neural_network == 'MLP':
                hyperparameters = {
                        'no_epochs' : [25],
                        'batch_size' : 64,
                        'activ' : 'relu',
                        'layers' : [128,128,128],
                        'dropout' : 0.05
                }
        elif neural_network == 'CNN':
                hyperparameters = {
                        'no_epochs' : 25,
                        'batch_size' : 64,
                        'hidden_channels' : 10,
                        'kernel_size_conv' : [[2,2], [2,2], [2,2]],   # [spec, wind, map]
                        'kernel_size_pool' : [[2], [2], [2]],   # [spec, wind, map]
                        'linear_layers' : [[250,250], [250,250], [250,250]],  # [spec, wind, map]
                        'activ' : 'relu',
                        'dropout' : 0.05,
                        'pool' : 'maxpool',
                        }                



elif mode == 'hyperparameter_tuning':
        if neural_network == 'MLP':
                # hyperparameters = {
                #         'no_epochs' : [10, 25, 40],
                #         'batch_size' : [32, 64],
                #         'activ' : ['relu', 'lrelu', 'tanh'],
                #         'layers' : [[500],
                #                 [128, 128],
                #                 [250, 250],
                #                 [128, 128, 128],
                #                 [250, 250, 250]],
                #         'dropout' : [0.05, 0.1]
                # }

                hyperparameters = {
                        'no_epochs' : 2,
                        'batch_size' : [32, 64],
                        'activ' : ['relu'],
                        'layers' : [[128]],
                        'dropout' : [0.05]
                }


        elif neural_network == 'CNN':

                if input_ml_model == 'int_params':
                        hyperparameters = {
                                'no_epochs' : [10, 25, 40],
                                'batch_size' : [32, 64],
                                'hidden_channels' : [5,10],
                                'kernel_size_conv' : [  [[2]],
                                                        # [[3]],
                                                        [[2,2]],
                                                        # [[3,3]]    
                                                        ],   # [spec, wind, map]
                                'kernel_size_pool' : [
                                                        [[2]],
                                                        [[2,2]],
                                                        ],   # [spec, wind, map]
                                'linear_layers' : [
                                                [[250]],
                                                # [[128,128]],
                                                [[250,250]],

                                                ],  # [spec, wind, map]
                                'activ' : 'relu',
                                'dropout' : [0.05, 0.1],
                                'pool' : 'maxpool',
                                }

                elif input_ml_model == 'spec': 
                        hyperparameters = {
                                'no_epochs' : [10, 25, 40],
                                'batch_size' : [32, 64],
                                'hidden_channels' : [5,10],
                                'kernel_size_conv' : [  [[2]],
                                                        [[3]],
                                                        [[2,2]],
                                                        [[3,3]]    
                                                        ],   # [spec, wind, map]
                                'kernel_size_pool' : [
                                                        [[2]],
                                                        [[2,2]],
                                                        ],   # [spec, wind, map]
                                'linear_layers' : [
                                                [[250]],
                                                # [[128,128]],
                                                [[250,250]],

                                                ],  # [spec, wind, map]
                                'activ' : 'relu',
                                'dropout' : [0.05, 0.1],
                                'pool' : 'maxpool',
                                }
                        
                elif input_ml_model == 'specwind':
                        hyperparameters = {
                                'no_epochs' : [10, 25, 40],
                                'batch_size' : [32, 64],
                                'hidden_channels' : [5, 10],
                                'kernel_size_conv' : [
                                                        [[2], [2]],
                                                        [[3], [2]],
                                                        [[2,2], [2,2]],
                                                        [[3,3], [2,2]]
                                                        ],   # [spec, wind, map]
                                'kernel_size_pool' : [
                                                        [[2], [2]],
                                                        [[2,2], [2,2]],
                                                        ],   # [spec, wind, map]
                                'linear_layers' : [
                                                [[250], [250]],
                                                # [[128,128], [128,128]],
                                                [[250,250], [250, 250]],

                                                ],  # [spec, wind, map]
                                'activ' : 'relu',
                                'dropout' : [0.05, 0.1],
                                'pool' : 'maxpool',
                                }


                elif input_ml_model == 'wind':
                        hyperparameters = {
                                'no_epochs' : [10, 25, 40],
                                'batch_size' : [32, 64],
                                'hidden_channels' : [5, 10],
                                'kernel_size_conv' : [
                                                        [[None], [2]],
                                                        # [[None], [3]],
                                                        [[None], [2,2]],
                                                        # [[None], [3,3]]      
                                                        ],   # [spec, wind, map]
                                'kernel_size_pool' : [
                                                        [[None], [2]],
                                                        [[None], [2,2]],
                                                        ],   # [spec, wind, map]
                                'linear_layers' : [
                                                [[None], [250]],
                                                # [[None], [128,128]],
                                                [[None], [250, 250]],

                                                ],  # [spec, wind, map]
                                'activ' : 'relu',
                                'dropout' : [0.05, 0.1],
                                'pool' : 'maxpool',
                                }



########## Locations ##########
if location == 'Sulafjorden':
        target_datasets = [target_Sulafjorden]
        input_datasets_wind = [wind_f1_Sulafjorden, wind_f2_Sulafjorden]
        input_datasets_maps = [map_Sulafjorden]
        loc_nr = [0]

elif location == 'Vartdalsfjorden':
        target_datasets = [target_Vartdalsfjorden]
        input_datasets_wind = [wind_f1_Vartdalsfjorden, wind_f2_Vartdalsfjorden]
        input_datasets_maps = [map_Vartdalsfjorden]
        loc_nr = [1]

else:
        target_datasets = [target_Sulafjorden, target_Vartdalsfjorden]
        input_datasets_wind = [wind_f1_Sulafjorden, wind_f2_Sulafjorden, wind_f1_Vartdalsfjorden, wind_f2_Vartdalsfjorden]
        input_datasets_maps = [map_Sulafjorden, map_Vartdalsfjorden]
        loc_nr = [1,2]


########## Set up model class ##########
ml_model_class = model_class.ml_model_class(
                                neural_network = neural_network,
                                input_ml_model = input_ml_model,
                                target_datasets_spec=target_datasets, 
                                input_datasets_spec=input_spec, 
                                input_datasets_wind=input_datasets_wind, 
                                input_datasets_maps=input_datasets_maps,
                                hyperparameters=hyperparameters, 
                                period = period, 
                                time_shift=True,
                                time_shift_hrs=time_shift_hrs                        
                                )


########## Run class ##########
if mode == 'hyperparameter_tuning':

        if input_ml_model == 'int_params':
                # ml_model_class.hyperparameter_tuning(int_params=['Hs', 'peak_freq', 'peak_dir'], output_file=output_file_int)
                ml_model_class.hyperparameter_tuning(int_params=['Hs', 'peak_freq', 'peak_dir'], output_file=output_file)

        else:
                # ml_model_class.hyperparameter_tuning(int_params=['Hs', 'peak_freq', 'peak_dir', 'mean_freq', 'mean_dir'], output_file=output_file_models)
                ml_model_class.hyperparameter_tuning(int_params=['Hs', 'peak_freq', 'peak_dir', 'mean_freq', 'mean_dir'], output_file=output_file)


elif mode == 'run_model':
        if target_data:
                pred_values, training_time, time_test, ml_model, target_test = ml_model_class.run_model(target_data=target_data)  
        else:
                pred_values, training_time, time_test, ml_model = ml_model_class.run_model()      
  

        print('Predicted values shape: ', pred_values.shape)
        print('Training time [min]: ', training_time/60)  

        if input_ml_model == 'int_params':
                pred_da = xr.Dataset(
                        {
                        "Hs": (("location","time"), pred_values[:,:,0]),
                        "peak_freq": (("location","time"), pred_values[:,:,1]),    
                        "peak_dir": (("location","time"), pred_values[:,:,2]),
                        },
                        coords={"location":np.arange(pred_values.shape[0]), "time": time_test},
                )

        else:
                pred_da = xr.DataArray(
                    pred_values,
                    dims=['loc', "time", "frequency", "direction"],
                    coords={
                        'loc': loc_nr,
                        "time": time_test,
                        "frequency": target_datasets[0].frequency.values,
                        "direction": target_datasets[0].direction,
                        "longitude": ("loc", [ds.longitude.values[0] for ds in target_datasets]),
                        "latitude":  ("loc", [ds.latitude.values[0] for ds in target_datasets]),
                    },
                    name="SPEC"
                )


        if target_data:

                target_da = xr.DataArray(
                    target_test,
                    dims=['loc', "time", "frequency", "direction"],
                    coords={
                        'loc': loc_nr,
                        "time": time_test,
                        "frequency": target_datasets[0].frequency.values,
                        "direction": target_datasets[0].direction,
                        "longitude": ("loc", [ds.longitude.values[0] for ds in target_datasets]),
                        "latitude":  ("loc", [ds.latitude.values[0] for ds in target_datasets]),
                    },
                    name="SPEC"
                )

                target_da.to_netcdf(output_file_target)


        pred_da.attrs["hyperparameters"] = json.dumps(hyperparameters)
        pred_da.attrs["ml_model"] = str(ml_model)
        pred_da.attrs['training_time [min]'] = training_time/60
        print('Results saved in: ', output_file, '\n')
        pred_da.to_netcdf(output_file)
