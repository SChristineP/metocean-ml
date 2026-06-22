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
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn

from metocean_ml import dataset, trainer, models, spectra_tools

def standardize_array(data:pd.DataFrame|xr.DataArray|np.ndarray, named_columns = True, dtype=np.float64):
    """
    Assumes an array of data, with time in the first dimension.
    Attempts to standardize the format to a pandas dataframe (time x features).
    
    Returns
    --------
    pd.DataFrame
        The standardized dataframe.
    list[list]
        A list of coordinates, corresponding to the dimensions except time.
        E.g. [[latitudes],[longitudes],[frequencies],[directions]].
        These can be used to restore the original shape of the array.
    """


    # The goal of this section is to retrieve the values,
    # the time-index, and the remaining indices
    if isinstance(data,pd.DataFrame):
        values = data.values
        time = data.index
        coords = {"feature":list(data.columns)}
    elif isinstance(data,xr.DataArray):
        time = data[data.dims[0]].data
        if len(data.dims)==1: # just one feature
            if data.name:
                coords = {"feature":[data.name]}
            else:
                coords = {"feature":["Var 0"]}
        elif len(data.dims)>1:
            coords = {dim:list(data[dim].values) for dim in list(data.dims)[1:]}
        else:
            raise ValueError("Zero-dimensional array not allowed.")
        values = data.values
    elif isinstance(data,xr.Dataset):
        if np.any([len(data[v].dims)>1 for v in data]):
            print(data)
            raise TypeError("Dataset is only accepted with a single coordinate (time)")
        data = data.to_dataframe()
        values = data.values
        coords = {"feature":list(data.columns)}
        time = data.index
    elif isinstance(data,np.ndarray):
        if len(data.shape)==1:
            data = np.expand_dims(data,1)
        coords = {f"dim {i+1}": list(np.arange(d)) for i,d in enumerate(data.shape[1:])}
        time = np.arange(len(data))
        values = data
    else:
        raise TypeError(f"Unknown data type: {type(data)}.")

    # Transform coordinates to a single dimension of column names
    if named_columns:
        if len(coords)==1 and "feature" in coords: # Simplified names if original array is 2D
            columns = list(coords["feature"])
        else:   # generalized naming scheme for ND array
            dims, features = zip(*coords.items())
            columns = [str(dict(zip(dims, v))) for v in itertools.product(*features)]
    else:
        columns = np.arange(values.shape[1])

    # Reshape values to 2D (time x columns) and create dataframe
    values = values.reshape(len(time),-1)
    values = pd.DataFrame(values,index=time,columns=columns).sort_index().astype(dtype,errors="ignore")

    # This part simply checks for columns that were not converted to the requested dtype (float),
    not_converted = values.dtypes!=dtype
    if np.any(not_converted):
        not_converted = values.columns[not_converted].values
        drop_dict = {k:values[k].dtype for k in not_converted}
        if len(coords)>1:
            raise TypeError(f"Encountered problematic datatype(s) with columns {drop_dict}, which could not be converted to {dtype}.",
                            "The column(s) could not be dropped due to being part of a multidimensional array.")
        print(f"Warning: Columns dropped due to problematic data type: {drop_dict}")
        values = values.drop(not_converted,axis=1)
        print(coords)
        coords = {next(iter(coords)):{c for c in coords[next(iter(coords))] if c not in not_converted}}

    return values, coords



def restore_array(data,coords):
    """
    Restore an array to its original ND shape, using the coords from format_array.
    """
    if len(coords) == 1: # Array is 2D.
        return xr.Dataset.from_dataframe(data)

    # else:
    time = data.index
    shape = [len(time)]
    for k,v in coords.items():
        shape.append(len(v))
    data = data.values.reshape(shape)
    coords = {**{"time":time},**coords}
    return xr.DataArray(data,coords)



def merge_datasets(data:list[pd.DataFrame|np.ndarray|xr.DataArray|xr.Dataset],
                   keys=None,
                   resample = "1h",
                   join = "inner",
                   ):
    """
    Standardize a list of datasets,
    optionally resample and merge them to one dataframe.
    
    Parameters
    -----------
    data : list[array]
        List of arrays (numpy, pandas, xarray) to merge.
    keys : list[str]
        A list of names for the datasets, for identification.
    resample : str
        Resampling frequency for the pandas resample function, to standardize time index of datasets.
    join : str
        Index join method on merge - "inner" or "outer".
        
    Returns
    -------
    pd.DataFrame or list[pd.DataFrame]
        The input arrays, transformed to 2D pandas tables and merged (if join is not None)
    dict[dict]
        Dictionary of metadata containing the coordinates of the original ND arrays,
        which can be used to restore the original shape from the dataframe.
    """

    for i,d in enumerate(data):
        if isinstance(d,pd.DataFrame) and isinstance(d.columns,(pd.MultiIndex,list,tuple)):
            print(f"WARNING: Dataset {i} has multiindex columns. These will be merged into single level string columns. ")
        
    if keys and (len(keys) != len(data)):
        raise ValueError(f"Got len(data)=={len(data)} while len(keys)=={len(keys)}.")
    if not keys:
        keys = [f"Dataset {i}" for i in range(len(data))]

    metadata = {}
    for i in range(len(data)):
        df, coords = standardize_array(data[i])
        metadata[keys[i]] = {"coords":coords}
        if resample:
            df.index = pd.to_datetime(df.index)
            df = df.resample(resample).interpolate("linear")
        data[i] = df

    if join:
        data = pd.concat(data,axis=1,join=join,keys=keys)
    
    return data, metadata



def align_dataframes(input_data:pd.DataFrame,target_data:pd.DataFrame):
    """
    Align input and target data.

    Returns
    --------
    pd.DataFrame
        Input data for training, validation and testing, aligned with the target data timestamps.
    pd.DataFrame
        Target data for training, validation and testing, aligned with the input data timestamps.
    pd.DataFrame
        Inference input data, which has no corresponding timestamps in the target data, to be used for prediction.
    """
    if not isinstance(input_data,pd.DataFrame):
        raise TypeError(f"Parameter input_data must be pandas DataFrame, got {type(input_data)}.")
    if not isinstance(target_data,pd.DataFrame):
        raise TypeError(f"Parameter target_data must be pandas DataFrame, got {type(target_data)}.")

    input_data.index = pd.to_datetime(input_data.index)
    target_data.index = pd.to_datetime(target_data.index)

    train_X, train_Y = input_data.align(target_data,join="inner",axis=0)
    if len(train_X) == 0 or len(train_Y) == 0:
        raise ValueError("No matching timestamps found. Check dataframe index timestamps.")

    inference_data = input_data.drop(train_X.index,axis=0)
    return train_X, train_Y, inference_data



def standardize_wave_dataset(data):
    '''
    Standardize a 2D wave spectrum dataset to match the WINDSURFER/NORA3 format.

    Parameters:
    - data : xarray.Dataset
        Input dataset to standardize.

    Returns:
    - xarray.Dataset
        The standardized dataset.

    '''

    # Detect NORAC product
    if 'product_name' in data.attrs and data.attrs['product_name'].startswith("ww3"):
        # Rename dims and vars
        data = data.rename({
            'frequency': 'freq',
            'efth': 'SPEC'})

    return data

def filter_period(data, period):
    '''
    Filters the dataset to a specified time period.

    Parameters
    - data : xarray.Dataset or xarray.DataArray
        The input dataset containing a 'time' dimension.
    period : tuple or None
        A tuple specifying the desired time range.
        - (start_time, end_time): Filters between start_time and end_time.
        - (start_time,): Filters to a single timestamp.
        - None: Uses the full time range available in data.
        Both start_time and end_time may be strings or datetime-like objects.

    Returns
    filtered_data : xarray.Dataset or xarray.DataArray
        Subset of the data within the specified time range.
    period_label : str
        A string label describing the filtered time period.
        If start and end times are equal, only that time is returned.
    '''

    # Finds data time range
    data_start = pd.to_datetime(data.time.min().values)
    data_end = pd.to_datetime(data.time.max().values)

    # Uses the full dataset if period is None
    if period == None:
        start_time, end_time = data_start, data_end
    elif period[0] is None:
        start_time = data_start
    elif period[1] is None:
        end_time = data_end
    
    # Uses the choosen period range
    elif isinstance(period,list):
        start_time, end_time = pd.to_datetime(period[0]), pd.to_datetime(period[1])

        if start_time < data_start or end_time > data_end:
            raise ValueError(f"Period {start_time} to {end_time} is outside data range {data_start} to {data_end}.")
    elif not isinstance(period, list):
        raise ValueError (f'Period must be a list.')

    # Uses one timestamp if choosen
    else:
        start_time = end_time = pd.to_datetime(period)
        if start_time not in data.time.values:
            raise ValueError(f"({start_time}) is outside data range {data_start} to {data_end}.")

    filtered_data = data.sel(time=slice(start_time, end_time))

    if start_time == end_time:
        period_label = f"{start_time.strftime('%Y-%m-%dT%H')}Z"
    else:
        period_label = f"{start_time.strftime('%Y-%m-%dT%H')}Z to {end_time.strftime('%Y-%m-%dT%H')}Z"

    return filtered_data, period_label

def error_metrics_1D(target,pred):
    return {
        "mse": mean_squared_error(target, pred),
        "mae": mean_absolute_error(target, pred),
        "r2": r2_score(target, pred),
        "rmse":root_mean_squared_error(target,pred)}

def dir_error(target,pred):
    diff = np.abs(target-pred)%360
    diff = np.minimum(diff,360-diff)
    return error_metrics_1D(np.zeros_like(diff),diff)

def spec_prediction_performance(spec_pred, spec_target, freq, dir,upsample=1000, int_params = ['Hs', 'peak_freq', 'peak_dir']):
    params_pred = integrated_parameters_dict(spec_pred,freq,dir, params=int_params)
    params_target = integrated_parameters_dict(spec_target,freq,dir, params=int_params)
    perf = {}
    for k,pred in params_pred.items():
        target = params_target[k]
        if "dir" in k:
            perf[k] = dir_error(target,pred)
        else:
            perf[k] = error_metrics_1D(target,pred)
    return params_pred, params_target, pd.DataFrame(perf)

def Dataset_to_DF(dataset, prefix, time):
    if isinstance(dataset, dict):
        df = pd.DataFrame(dataset).add_prefix(prefix)
        df = df.set_index(time)
    else:
        df = dataset.to_dataframe().add_prefix(prefix)
    df.index.name = None
    return df




def integrated_parameters_dict(
    spec:       np.ndarray|xr.DataArray, 
    frequencies:np.ndarray|xr.DataArray, 
    directions: np.ndarray|xr.DataArray,
    params: list = ['Hs', 'peak_freq', 'peak_dir']) -> dict:
    """
    Calculate the integrated parameters of a 2D wave spectrum, 
    or some array/list of spectra. Uses simpsons integration rule.

    Implemented: Hs, peak dir, peak freq.

    a = ∫∫ cos(dir) * F(freq, dir) dfreq ddir
    b = ∫∫ sin(dir) * F(freq, dir) dfreq ddir
    
    Arguments
    ---------
    spec : np.ndarray or xr.DataArray
        An array of spectra. The shape must be either 
        [..., frequencies, directions] or [..., frequencies*directions].
    frequencies : np.ndarray or xr.DataArray
        Array of spectra frequencies.
    directions: np.ndarray or xr.DataArray
        Array of spectra directions.
        
    Returns
    -------
    spec_parameters : dict[str, np.ndarray]
        A dict with keys Hs, peak_freq, peak_dir, and values are arrays
        of the integrated parameter.
    """
    # spec = spec_dataset['SPEC']

    spec_parameters = {}

    # Make sure all arrays are numpy.
    if isinstance(spec, xr.DataArray):
        spec = spec.data
    if isinstance(frequencies, xr.DataArray):
        frequencies = frequencies.data
    if isinstance(directions, xr.DataArray):
        directions = directions.data

    params_list = {'Hs', 'peak_freq', 'peak_period', 'peak_dir', 'mean_freq', 'mean_period', 'mean_dir'}
    if invalid:=set(params)-params_list: 
        raise ValueError(f'Invalid parameters: {invalid}. This function only calculates: {params_list}')

    # Check if spec values and shape are OK
    if np.any(spec < 0):
        print("Warning: negative spectra values set to 0")
        spec = np.clip(spec, a_min=0, a_max=None)

    flat_check = (len(spec.shape)<2)
    freq_check = (len(frequencies) != spec.shape[-2])
    dir_check = (len(directions) != spec.shape[-1])
    if flat_check or freq_check or dir_check:
        try:
            spec = spec.reshape(spec.shape[:-1]+(len(frequencies),len(directions)))
        except Exception:
            raise IndexError("Spec shape does not match frequencies and directions.")

    if 'peak_freq' in params or 'peak_dir' in params:
        # Use argmax to find indices of largest value of each spectrum.
        peak_dir_freq = np.array([np.unravel_index(s.argmax(),s.shape) 
            for s in spec.reshape(-1,len(frequencies),len(directions))])
        peak_dir_freq = peak_dir_freq.reshape(spec.shape[:-2]+(2,))
        peak_freq = frequencies[peak_dir_freq[...,0]]
        peak_dir = directions[peak_dir_freq[...,1]] if 'peak_dir'in params else None
        spec_parameters['peak_freq'] = peak_freq
        spec_parameters['peak_dir'] = peak_dir
        spec_parameters['peak_dir_rad'] = np.deg2rad(peak_dir)

    
    if 'Hs' in params or 'mean_dir' in params or 'mean_freq' in params or 'mean_period' in params:
        # Integration requires radians
        if np.max(directions) > 2*np.pi: 
            directions = np.deg2rad(directions)
        
        # Sort on direction before integration
        sorted_indices = np.argsort(directions)
        directions = directions[sorted_indices]
        spec = spec[...,sorted_indices]
        
        # Integration with simpson's rule
        S_f = simpson(spec, x=directions)
        m0 = simpson(S_f, x=frequencies)
        Hs = 4 * np.sqrt(m0)

        spec_parameters['Hs'] = Hs

        if 'mean_freq' in params or 'mean_period' in params:
            m1 = simpson(frequencies*S_f, x=frequencies)
            mean_freq = (m1/m0)
            spec_parameters['mean_freq'] = mean_freq
            if 'mean_period' in params:  
                mean_period = (m0/m1)
                spec_parameters['mean_period'] = mean_period
    
    if 'mean_dir' in params:
        spec_dataset = xr.DataArray(
            spec,
            dims=["time", "frequency", "direction"],
            coords={
                "time" : np.arange(spec.shape[0]),
                "frequency": frequencies,
                "direction": directions,
            },
            name="SPEC")
        
        mean_dir = np.rad2deg(compute_mean_wave_direction(spec_dataset))
        spec_parameters['mean_dir'] = mean_dir

    return spec_parameters


def compute_mean_wave_direction(data, var='SPEC', mean_pdir=False):
    '''
    Compute the mean wave direction from a directional wave energy spectrum.

    This function calculates the mean wave direction dir_mean based on the 
    discrete approximation of the integrals:

        a = ∫∫ cos(dir) * F(freq, dir) dfreq ddir
        b = ∫∫ sin(dir) * F(freq, dir) dfreq ddir
        dir_mean = arctan2(b, a)

    where:
        - F(freq, dir) is the spectral energy density as a function of frequency and direction.
        - The integrals are approximated by summations over the frequency and
          direction bins weighted by the bin widths.

    If mean_pdir=True, the mean peak direction is calculated instead. 
          
    Parameters:
    - data : xarray.Dataset or xarray.DataArray
        Wave spectrum with dimensions including 'freq' and 'direction'. If a Dataset,
        the spectral variable is specified by `var`.
    - var : str, optional, default = 'SPEC'
        Name of the spectral variable in `data` if `data` is a Dataset.
    - mean_pdir : bool, optional, default = False,
        False : calculates mean wave direction
        True : calculated mean peak wave direction

    Returns:
    - mean_dir_rad : xarray.DataArray
        Mean (peak) wave direction in radians using the mathematical convention:
        0 = East, positive counter-clockwise (CCW).

    Notes:
    - Frequency and direction bin widths are computed using gradients and used to weight the integration.
    - The final direction is based on vector summation (a, b) and converted using arctangent.
    - Based on the method in the WAVEWATCH III User Manual (v6.07, NOAA/NCEP, 2019).
    '''

    try:
        spectrum = data[var].sortby('direction')
    except KeyError :
        spectrum = data.sortby('direction')

    direction = data['pdir'] if mean_pdir else spectrum['direction']

    # directions_rad = np.deg2rad((450 - direction) % 360)                                                        # Convert to mathematical convention (radians, pointing to East counterclockwise)
    directions_rad = np.deg2rad(direction)
    if not mean_pdir:
        # Full 2D integration
        delta_freq = np.gradient(spectrum.frequency.values)                                                          # Calculate frequency and dir bin widths
        delta_dir = np.gradient(spectrum['direction'])
        delta_dir_rad = np.deg2rad(delta_dir)                   
        dfreq_2d = xr.DataArray(delta_freq, dims=['frequency'])                                                      # Create DataArrays for bin widths to broadcast over spectrum dims
        ddir_2d = xr.DataArray(delta_dir_rad, dims=['direction'])
        area_element = dfreq_2d.broadcast_like(spectrum) * ddir_2d.broadcast_like(spectrum)                     # Compute the area element dfreq ddir for each frequency-direction bin by outer product


    if mean_pdir:
        peak_directions_rad = np.deg2rad(450 - (spectrum.integrate('frequency').idxmax(dim='direction')) % 360)      # Computes mean peak direction
        a = xr.ufuncs.cos(peak_directions_rad) 
        b = xr.ufuncs.sin(peak_directions_rad)

    else:
        a = (xr.ufuncs.cos(directions_rad) * spectrum * area_element).sum(dim=['frequency', 'direction'])            # Compute weighted sums a and b over freq and direction dimensions
        b = (xr.ufuncs.sin(directions_rad) * spectrum * area_element).sum(dim=['frequency', 'direction'])

    mean_dir_rad = np.arctan2(b, a)

    return mean_dir_rad


def taylor_diagram(df,var_ref,var_comp,norm_std=True, colors=[],output_file='Taylor_diagram.png'):
    """
    Plot a Taylor diagram
    df: dataframe with all timeseries
    var_ref: list of string with the name of the timeseries of reference
    var_comp: list of strings with the names of the timeseries to be compared with the reference
    norm_std: option to define normalized or non-normalized standard deviation

    Option 1: #[[A,3],[B,3],[C,3]]
    var_ref   = ['hs_sulaA','hs_sulaB','hs_sulaC'] 
    var_comp = ['hs_nora3']
    norm_std = True #can only run with this option

    Option 2 : #Originalen [[A,3],[A,4],[A,5]]
    var_ref   = ['hs_sulaA']
    var_comp = ['hs_nora3','hs_nora4','hs_nora5']
    norm_std = True/False #can run with both options

    Option 3 : #[[A,3],[B,4],[C,5]]
    var_ref   = ['hs_sulaA','hs_sulaB','hs_sulaC']
    var_comp = ['hs_nora3','hs_nora4','hs_nora5']
    norm_std = True #can only run with this option

    """

    def run_taylor(var_ref,var_comp,maxx,index):
        def correlation(var_ref,var_comp,max_std,radius):
            # Calculate the coordinates of the points x and y
            # Correlation coefficient between the reference and the other(s)
            ccf=np.zeros((len(var_comp)+1))
            ccf[0]=np.corrcoef(df[var_ref[0]].to_numpy(),df[var_ref[0]].to_numpy())[0,1] # Should be 1
            for i in range(len(var_comp)):
                ccf[i+1]=np.corrcoef(df[var_ref[0]].to_numpy(),df[var_comp[i]].to_numpy())[0,1]     

            # Coordinates of the lines for the correlation
            xbc1=np.arange(0.0,max_std+0.015,0.001)
            corr=np.array([0.2,0.4,0.6,0.8,0.9,0.95,0.99])
            ycr=np.zeros((len(corr),len(xbc1)))
            for r in range(len(corr)):
                for a in range(len(xbc1)):
                    ycr[r,a]=np.tan(np.arccos(corr[r]))*xbc1[a]
                    d=np.sqrt(ycr[r,a]**2+xbc1[a]**2)
                    if d>np.max(radius):
                        ycr[r,a]=np.nan
                    del d

            return ccf,xbc1,ycr,corr

        def set_axes_and_std(var_ref,var_comp,maxx):
            std=np.zeros((len(var_comp)+1))
            std[0]=np.std(df[var_ref].to_numpy())
            for i in range(len(var_comp)):
                std[i+1]=np.std(df[var_comp[i]].to_numpy())
            if norm_std is True:
                std=std/std[0]

            # Coordinates of the big circles
            min_std=0
            max_std=maxx + 0.5 #to set the max of x-y

            if max_std<=5:
                step=0.5
            elif ((max_std>5) & (max_std<=10)):
                step=1
            elif ((max_std>10) & (max_std<=20)):
                step=3
            else:
                step=5

            radius=np.arange(min_std+step,max_std,step)
            radius=np.concatenate([radius,np.array([max_std])])
            radius1=radius
            xbc=np.arange(0.0,max_std+0.01,0.0001)
            ybc=np.zeros((len(radius),len(xbc)))
            ysc=np.zeros((len(radius),len(xbc)))
            for r in range(len(radius)):
                for a in range(len(xbc)):
                    val_ybc = radius[r]**2-xbc[a]**2
                    ybc[r,a]=np.sqrt(val_ybc) if val_ybc > 0 else np.nan
                    val_ysc = radius1[r]**2-(xbc[a]-std[0])**2
                    ysc[r,a]=np.sqrt(val_ysc) if val_ysc > 0 else np.nan

                    if np.isnan(ysc[r,a]):
                        d=np.sqrt(ysc[r,a]**2+xbc[a]**2)
                        if d>np.max(radius):
                            ysc[r,a]=np.nan
                        del d

            return std,max_std,radius,xbc,ybc,step

        def plotting(var_ref,var_comp,maxx,index):
            #Plot the data
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)

            #Get the data
            std,max_std,radius,xbc,ybc,step=set_axes_and_std(var_ref,var_comp,maxx)
            ccf,xbc1,ycr,corr = correlation(var_ref,var_comp,max_std,radius)

            # print('std:', std)
            # print('correlation ', ccf)

            # centered RMSE (matches Taylor diagram)
            rmse_centered = np.zeros(len(var_comp))

            for i in range(len(var_comp)):
                rmse_centered[i] = np.sqrt(
                    std[0]**2 +
                    std[i+1]**2 -
                    2 * std[0] * std[i+1] * ccf[i+1]
                )

            # print("Centered RMSE (Taylor):", rmse_centered)  

            return std, ccf, rmse_centered        

        std, ccf, rmse_centered = plotting(var_ref,var_comp,maxx,index)
        return std, ccf, rmse_centered
    
    legends = []
    if (len(var_ref)<len(var_comp)) and (len(var_ref)==1): #for option 2
        # fig, ax = plt.subplots(figsize=(8, 8))
        model_ref = df[var_ref[0]] 
        std_mod = df[model_ref.name].std()
        var=[*var_ref,*var_comp] 
        #to find the max of the variables to set the len of axis 
        maxx=int(np.max(df[var].std())/std_mod if norm_std else np.max(df[var].std())) #max value on the x-y axis

        show=True #Always true in this case
        index=0        
        var_ref = np.array(var_ref)
        run_taylor(var_ref,var_comp,maxx,index)

    elif (len(var_ref)>len(var_comp)) and (len(var_comp)==1): #for option 1
        if norm_std is not True:
            print('This option can only be run with normalized standard deviation as True.')
            return
        # fig, ax = plt.subplots(figsize=(10, 10))
        var=[*var_ref,*var_comp]
        i_end = len(var_ref)
        minn = np.nanmin(df[var_ref].std())
        stdd_c = df[var_comp].std()
        maxx = int(np.max(stdd_c/minn))

        index = 0
        #loop over len of var_ref
        for i in range(len(var_ref)):
            var_ref1 = var_ref[i]
            var_comp1 = var_comp[0]
            model_ref = df[var_ref1]
            std_mod = df[model_ref.name].std()
            if i==i_end-1:
                show=True
                index = index+1
                run_taylor([var_ref1],[var_comp1],maxx,index)
            else:
                show=False
                index = index + 1
                run_taylor([var_ref1],[var_comp1],maxx,index)

    elif len(var_ref)==len(var_comp): #for option 3
        if norm_std is not True:
            print('This option can only be run with normalized standard deviation as True.')
            return
        # fig, ax = plt.subplots(figsize=(10, 10))
        var=[*var_ref,*var_comp]
        i_end = len(var_ref)
        #to find the max value to set on the x-y axis
        std_max = []
        for i in range(len(var_ref)):
            std_m = (df[var_comp[i]].std()/df[var_ref[i]].std())
            std_max.append(std_m)
        maxx = np.nanmax(std_max)
        index = 0
        #loop over len of var_ref
        for i in range(len(var_ref)):
            var_ref1 = var_ref[i]
            var_comp1 = var_comp[i]
            model_ref = df[var_ref1]
            std_mod = df[model_ref.name].std()

            if i==i_end-1:
                index = index + 1
                show=True
                std, ccf, rmse_centered = run_taylor([var_ref1],[var_comp1],maxx,index)
            else:
                index = index + 1 
                show=False
                std, ccf, rmse_centered = run_taylor([var_ref1],[var_comp1],maxx,index)
    
    return std, ccf, rmse_centered



def accuracy_fn_np(y_true, y_pred, tol):
    diff = np.abs(y_true - y_pred)
    correct = np.sum(diff <= tol)
    total = y_true.size   # total number of elements
    return (correct / total) * 100