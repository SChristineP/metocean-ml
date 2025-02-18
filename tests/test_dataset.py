import pytest
from metocean_ml.trainer import Trainer
from metocean_ml.dataset import TimeseriesDataset,TimeseriesWithContext
import numpy as np

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

# Helper function to create some dummy data for testing
def create_dummy_data():
    input_data = np.random.randn(100, 10)  # 100 samples, 10 features
    target_data = np.random.randn(100, 10)  # 100 samples, 10 targets
    return input_data, target_data

def create_context_data():
    input_data = np.random.randn(5, 10, 10)  # 5 nodes, 10 time steps, 10 features
    context_data = np.random.randn(5, 5)  # 5 nodes, 5 context features
    target_data = np.random.randn(5, 10, 10)  # 5 nodes, 10 time steps, 10 target features
    return input_data, context_data, target_data

# Test 1: Initialization and basic functionality of TimeseriesDataset
def test_timeseries_dataset_initialization():
    input_data, target_data = create_dummy_data()
    
    # Test for TimeseriesDataset with numpy array
    dataset = TimeseriesDataset(input_data, target_data, input_timestamps=3, time_offset=1)
    assert isinstance(dataset, TimeseriesDataset), "Failed to create TimeseriesDataset"
    
    # Check length of dataset
    assert len(dataset) == len(input_data) - 3 - 1, "Dataset length mismatch"
    
    # Check output shape from __getitem__
    x, y = dataset[0]
    assert x.shape == (3, 10), "Input shape mismatch"  # 3 time steps, 10 features
    assert y.shape == (10,), "Target shape mismatch"  # 10 target features

# Test 2: Initialization and basic functionality of TimeseriesWithContext
def test_timeseries_with_context_initialization():
    input_data, context_data, target_data = create_context_data()
    
    # Test for TimeseriesWithContext
    dataset = TimeseriesWithContext(input_data, context_data, target_data, input_timestamps=3)
    assert isinstance(dataset, TimeseriesWithContext), "Failed to create TimeseriesWithContext"
    
    # Check length of dataset
    assert len(dataset) == 5 * (10 - 3), "Dataset length mismatch for context"
    
    # Check output shape from __getitem__
    x, y = dataset[0]
    assert x.shape == (3 * 10 + 5,), "Input shape mismatch"  # (3 time steps * 10 features) + (5 context features)
    assert y.shape == (10,), "Target shape mismatch"  # 10 target features

# Test 3: Handling of edge cases (e.g., ValueError for invalid input_timestamps)
def test_timeseries_dataset_invalid_input_timestamps():
    input_data, target_data = create_dummy_data()
    
    # Test for invalid input_timestamps (less than 1)
    with pytest.raises(ValueError):
        TimeseriesDataset(input_data, target_data, input_timestamps=0)

    # Test for valid input_timestamps
    dataset = TimeseriesDataset(input_data, target_data, input_timestamps=1)
    assert len(dataset) == len(input_data) - 1 - 0, "Dataset length mismatch with input_timestamps=1"
    
    # Ensure proper dataset creation
    x, y = dataset[0]
    assert x.shape == (10,), "Input shape mismatch with input_timestamps=1"
    assert y.shape == (10,), "Target shape mismatch with input_timestamps=1"

