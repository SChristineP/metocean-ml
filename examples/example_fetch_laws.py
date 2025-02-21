import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from ..metocean_ml import physics, geospatial

# Load data
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../tests/data/E39_F_vartdalsfjorden.nc')
data = xr.open_dataset(os.path.join(dirname,filename))
lat,lon = data['latitude'].data, data['longitude'].data

# Convert to pandas
data = data.to_dataframe().iloc[:,:4]
data.columns = ['wind_speed','wind_direction','hs','tp']
data['wind_speed'] = data['wind_speed'].clip(lower=1e-10)

# Calculate fetch distances per direction
fetch = geospatial.get_fetch(lat,lon)

# Calculate effective fetch for the timeseries of wind
effective_fetch = physics.effective_fetch(fetch,data['wind_direction'], sector=30)

# Calculate wave parameters from fetch and wind speed
# Fetch laws: Holthuijsen, KahmaCalkoen, JONSWAP
wave_parameters = physics.fetch_laws(data['wind_speed'],fetch,laws='holthuijsen')

# Compare with observations
results = pd.DataFrame({
    'hs_fetch_law': wave_parameters['hs'],
    'hs_observed': data['hs'],
    'tp_fetch_law': wave_parameters['tp'],
    'tp_observed': data['hs']
}, index = data.index)

print(results.head(10))

print(results.describe())