# Imports
import xarray as xr
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from metocean_ml import preprocessing
import seaborn as sns

# Load data
data_path = os.path.join(os.path.dirname(__file__),"../tests/data/")
input_west = xr.load_dataset(data_path+"origin_nora3_62_5.nc")
input_north = xr.load_dataset(data_path+"origin_nora3_63_6.nc")
target_dataset = xr.load_dataset(data_path+"target_norac_62_6.nc")

# Select input data and merge to a DataFrame.
input_data = [input_west["SPEC"],input_north["SPEC"]]
keys = ["Spectra West","Spectra North"]
input_data, metadata = preprocessing.merge_datasets(input_data,keys=keys)

# Also make the target spectra into a 2D dataframe.
target_data,coords = preprocessing.standardize_array(target_dataset["efth"])

# Align timeseries, and split data
input_data, target_data, inference = preprocessing.align_dataframes(input_data,target_data)
X_train, X_test, y_train, y_test = train_test_split(input_data, target_data,shuffle=False,test_size=0.5)

# Transform variables to normal distribution
input_scaler = QuantileTransformer(output_distribution="normal")
target_scaler = QuantileTransformer(output_distribution="normal")
X_train_scaled = input_scaler.fit_transform(X_train)
X_test_scaled = input_scaler.transform(X_test)
y_train_scaled = target_scaler.fit_transform(y_train)
y_test_scaled = target_scaler.transform(y_test)

# Select model, fit and predict. Any sklearn model should work
model = LinearRegression()
model.fit(X_train_scaled,y_train_scaled)
y_predicted_scaled = model.predict(X_test_scaled)
y_predicted = target_scaler.inverse_transform(y_predicted_scaled)

# Calculate RMSE loss
print(f"RMSE: {root_mean_squared_error(y_test,y_predicted)}")

# Restore arrays to 3D shape (time, frequencies, directions)
y_predicted = pd.DataFrame(y_predicted,index=y_test.index,columns=y_test.columns)
predicted_spectra = preprocessing.restore_array(y_predicted,coords=coords)
target_spectra = preprocessing.restore_array(y_test,coords=coords)

# Visualize example
fig,ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(14,6))

t = target_spectra.sum(("frequency","direction")).idxmax("time") # the highest energy timestamp

yticks = target_spectra['frequency'].values.round(2)
xticks = target_spectra['direction'].values.round(2)
cmap = "Blues"

sns.heatmap(predicted_spectra.sel(time=t),cmap=cmap,xticklabels=xticks,yticklabels=yticks,ax=ax[0])
sns.heatmap(target_spectra.sel(time=t),cmap=cmap,xticklabels=xticks,yticklabels=yticks,ax=ax[1])
fig.suptitle(f"time = {t.values}")
ax[0].set_title("Predicted spectrum")
ax[1].set_title("Target spectrum")
ax[0].set_ylabel("Frequency")
ax[0].set_xlabel("Direction")
ax[1].set_xlabel("Direction")
plt.show()