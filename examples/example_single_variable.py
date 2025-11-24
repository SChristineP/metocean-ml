import matplotlib.pyplot as plt
from metocean_ml import pipelines
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import os

data_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data')
print(data_path)
df_norac = pd.read_csv(os.path.join(data_path,'NORAC_test.csv'),comment='#',index_col=0, parse_dates=True)
df_nora3 = pd.read_csv(os.path.join(data_path,'NORA3_test.csv'),comment='#',index_col=0, parse_dates=True)

# Define training and validation period:
start_training = '2019-01-01'
end_training   = '2019-12-31'
start_valid    = '2018-01-01'
end_valid      = '2018-12-31'

# Select method and variables for ML model:
model='GBR' # 'SVR_RBF', 'LSTM', GBR
var_origin = ['hs','tp','Pdir'] # ['hs','tp','Pdir', 'hs_sea', 'hs_swell'] 
var_train  = ['hs']
# Run ML model:
ts_pred = pipelines.predict_ts(ts_origin=df_nora3,var_origin=var_origin,ts_train=df_norac.loc[start_training:end_training],var_train=var_train, model=model)
# Plotting a month of data:
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(12, 6),gridspec_kw={'top': 0.95,'bottom': 0.150,'left': 0.05,'right': 0.990,'hspace': 0.2,'wspace': 0.2})
plt.title('Model: '+model+',Training Variables: '+','.join(var_origin))
plt.plot(df_nora3['hs'].loc['2017-12-30':'2018-01-30'],'o',label='NORA3')
plt.plot(ts_pred.loc['2017-12-30':'2018-01-30'],'x',label='NORAC_pred')
plt.ylabel('Hs[m]',fontsize=20)
plt.plot(df_norac['hs'].loc['2017-12-30':'2018-01-30'].asfreq('h'),'.',label='NORAC')
plt.grid()
plt.legend()
plt.savefig(model+'-'+'_'.join(var_origin)+'ts.png')
plt.close()

#Plot all the data:
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(12, 6),gridspec_kw={'top': 0.95,'bottom': 0.150,'left': 0.05,'right': 0.990,'hspace': 0.2,'wspace': 0.2})
plt.title('Model: '+model+',Training Variables: '+','.join(var_origin))
plt.plot(df_nora3['hs'],'o',label='NORA3')
plt.plot(ts_pred,'x',label='NORAC_pred')
plt.ylabel('Hs[m]',fontsize=20)
plt.plot(df_norac['hs'],'.',label='NORAC')
plt.grid()
plt.legend()
plt.savefig(model+'-'+'_'.join(var_origin)+'ts_all.png')
plt.close()

# Scatter plot and metrics:
plt.scatter(df_norac['hs'].loc[start_valid:end_valid], ts_pred.loc[start_valid:end_valid], color='black')
plt.title('scatter:'+model+'-'+'_'.join(var_origin))
plt.text(0, 1.0,'ΜΑΕ:'+str(np.round(mean_absolute_error(df_norac['hs'].loc[start_valid:end_valid], ts_pred.loc[start_valid:end_valid]),3)))
plt.text(0, 0.8,'$R²$:'+str(np.round(r2_score(df_norac['hs'].loc[start_valid:end_valid], ts_pred.loc[start_valid:end_valid]),3)))
plt.text(0, 0.6,'RMSE:'+str(np.round(mean_squared_error(df_norac['hs'].loc[start_valid:end_valid], ts_pred.loc[start_valid:end_valid])**0.5,3)))
plt.xlabel('Hs from NORAC')
plt.ylabel('Hs from NORAC_pred')
plt.savefig(model+'-'+'_'.join(var_origin)+'scatter.png')
plt.close()
