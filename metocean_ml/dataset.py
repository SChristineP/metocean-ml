import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader

class TimeseriesDataset(Dataset):
    '''
    A pytorch class to represent a timeseries of input and target data.
    The time dimension of input and target data should be equal,
    sampled at uniform intervals, and sorted ascending.
    
    Parameters
    ----------
    input_data : array
        The input data.
    target_data : array
        The target data.
    input_timestamps : int
        Number of input timestamps to include (>=1).
    time_offset : int
        How far ahead in time the target should be.
        0 means contemporary to the last input timestamp.
    '''
    def __init__(self,
                 input_data: np.ndarray | pd.DataFrame | xr.DataArray,
                 target_data: np.ndarray | pd.DataFrame | xr.DataArray,
                 input_timestamps: int = 1,
                 time_offset:int = 0):

        if input_timestamps < 1: raise ValueError("There must have at least one input.")
        
        if isinstance(input_data,(pd.DataFrame,xr.DataArray)): input_data=input_data.values
        if isinstance(target_data,(pd.DataFrame,xr.DataArray)): target_data=target_data.values

        self.X = torch.tensor(input_data,dtype=torch.float)
        self.Y = torch.tensor(target_data,dtype=torch.float)
        
        self.input_timestamps = input_timestamps
        self.time_offset = time_offset
        
    def __getitem__(self, index):
        x = self.X[index:index+self.input_timestamps]
        y = self.Y[index+self.input_timestamps+self.time_offset-1]
        return torch.squeeze(x), torch.squeeze(y)
    
    def __len__(self):
        return len(self.X)-self.input_timestamps-self.time_offset

class TimeseriesWithContext(Dataset):
    '''
    A pytorch class to represent a timeseries of input and target data,
    with supporting context data such as local geography.
    
    Parameters
    ----------
    input_timeseries : array
        Time-varying input data, shape (node,time,features).
    input_context : array
        Time-independent input data, shape (node,features).
    target_data : array
        Time-varying target data, shape (node,time,features).
    metadata : any, optional
        
    input_timestamps : int
        Number of input timestamps to include (>=1).
    '''
    def __init__(self,
                 input_data: np.ndarray | pd.DataFrame | xr.DataArray,
                 input_context : np.ndarray | pd.DataFrame | xr.DataArray,
                 target_data: np.ndarray | pd.DataFrame | xr.DataArray,
                 input_timestamps: int = 1):

        if input_timestamps < 1: raise ValueError("There must have at least one input.")
        
        if isinstance(input_data,(pd.DataFrame,xr.DataArray)): input_data=input_data.values
        if isinstance(target_data,(pd.DataFrame,xr.DataArray)): target_data=target_data.values

        self.X = torch.tensor(input_data,dtype=torch.float)
        self.C = torch.tensor(input_context,dtype=torch.float)
        self.Y = torch.tensor(target_data,dtype=torch.float)
        
        self.input_timestamps = input_timestamps

        self.nodes, self.time, _ = self.X.shape
        self.total_samples = self.nodes * (self.time-self.input_timestamps)

    def __getitem__(self, index):
        node_index = index//(self.time-self.input_timestamps)
        time_index = index%(self.time-self.input_timestamps)

        x = self.X[node_index,time_index:time_index+self.input_timestamps]
        c = self.C[node_index]
        y = self.Y[node_index,time_index+self.input_timestamps-1]
        return torch.cat([torch.flatten(x),torch.flatten(c)]), torch.squeeze(y)

    def __len__(self):
        return self.total_samples

