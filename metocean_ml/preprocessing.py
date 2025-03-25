import itertools

import pandas as pd
import numpy as np
import xarray as xr

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



def merge_datasets(data:list[pd.DataFrame,np.ndarray,xr.DataArray,xr.Dataset],
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
    train_X, train_Y = input_data.align(target_data,join="inner",axis=0)
    inference_data = input_data.drop(train_X.index,axis=0)
    return train_X, train_Y, inference_data