import matplotlib.pyplot as plt
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df_norac = pd.read_csv('../tests/data/NORAC_test.csv',comment='#',index_col=0, parse_dates=True)
df_nora3 = pd.read_csv('../tests/data/NORA3_test.csv',comment='#',index_col=0, parse_dates=True)

# Define training and validation period:
start_training = '2019-01-01'
end_training   = '2019-12-31'
start_valid    = '2018-01-01'
end_valid      = '2018-12-31'

# Select method and variables for ML model:
model='GBR' # 'SVR_RBF', 'LSTM', GBR
var_origin = ['hs','tp','Pdir']
var_train  = ['hs']

# Run ML model:
def predict_ts(ts_origin,var_origin, ts_train,var_train, model='GBR'):
    """
    Input:
    ts_origin: pandas DataFrame
    var_origin: variable name (str) e.g., ['hs','tp','Pdir','hs_swell']
    ts_train: pandas DataFrame
    var_train: variable name (str) e.g., ['hs']
    model = 'GBR', 'SVR_RBF', LSTM
    Output:
    ts_pred: pandas DataFrame
    """
    Y_pred = pd.DataFrame(columns=[var_train], index=ts_origin.index)

    def _add_suffix(var_origin, suffix='_x', exclude=['time']):
        if isinstance(var_origin, list) or isinstance(var_origin, pd.Index):
            return [var + suffix if var not in exclude else var for var in var_origin]
        elif isinstance(var_origin, str):
            return var_origin + suffix if var_origin not in exclude else var_origin
        else:
            return "Unsupported data type"


    # Add extension _x, _y
    ts_origin.columns = _add_suffix(ts_origin.columns, suffix='_x') 
    ts_train.columns  = _add_suffix(ts_train.columns, suffix='_y') 

    var_origin = _add_suffix(var_origin, suffix='_x')  
    var_train =  _add_suffix(var_train, suffix='_y')  

    # Merge or join the dataframes based on time
    #ts_origin.set_index('time', inplace=True)
    #ts_train.set_index('time', inplace=True)
    merged_data = pd.merge(ts_origin[var_origin], ts_train[var_train], how='inner', left_on='time', right_on='time')

    # Handling missing values if any
    merged_data = merged_data.dropna()
    # Extracting features and target variables
    X = merged_data[var_origin]
    Y = merged_data[var_train]
    
    # Splitting the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1)

    # Creating and fitting the linear regression model
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.ensemble import GradientBoostingRegressor
    if model == 'LinearRegression':
        model = LinearRegression()
        model.fit(X_train, Y_train)
        Y_pred[:] = model.predict(ts_origin[var_origin].values).reshape(-1, 1)    
    elif model == 'SVR_RBF':    
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        model.fit(X_train, Y_train)
        Y_pred[:] = model.predict(ts_origin[var_origin].values).reshape(-1, 1)    
    elif model == 'SVR_LINEAR':            
        model = SVR(kernel="linear", C=100, gamma="auto")
        model.fit(X_train, Y_train)
        Y_pred[:] = model.predict(ts_origin[var_origin].values).reshape(-1, 1)    
    elif model == 'SVR_POLY':    
        model = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
        model.fit(X_train, Y_train)
        Y_pred[:] = model.predict(ts_origin[var_origin].values).reshape(-1, 1)    
    elif model == 'GBR':
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
        model.fit(X_train, Y_train)
        Y_pred[:] = model.predict(ts_origin[var_origin].values).reshape(-1, 1)    
    # elif model == 'LSTM':
        # from keras.models import Sequential
        # from keras.layers import LSTM, Dense
        # from keras.optimizers import Adam
        # optimizer = Adam(learning_rate=0.001)
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        # model = Sequential()
        # model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        # model.add(LSTM(units=50, return_sequences=False))
        # model.add(Dense(units=1))
        # # Compile the model
        # model.compile(optimizer=optimizer, loss='mean_squared_error')
        # model.fit(X_train, Y_train, epochs=40, batch_size=32, validation_data=(X_test, Y_test), verbose=1)
        # Y_pred[:] = model.predict(ts_origin[var_origin].values).reshape(-1, 1) 

    # remove back extension _x, _y
    ts_origin.columns = [col.replace('_x', '') for col in ts_origin.columns]
    ts_train.columns = [col.replace('_y', '') for col in ts_train.columns]
    
    return Y_pred

ts_pred = predict_ts(ts_origin=df_nora3,var_origin=var_origin,ts_train=df_norac.loc[start_training:end_training],var_train=var_train, model=model)

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
plt.text(0, 1.0,'ΜΑΕ:'+str(np.round(sklearn.metrics.mean_absolute_error(df_norac['hs'].loc[start_valid:end_valid], ts_pred.loc[start_valid:end_valid]),3)))
plt.text(0, 0.8,'$R²$:'+str(np.round(sklearn.metrics.r2_score(df_norac['hs'].loc[start_valid:end_valid], ts_pred.loc[start_valid:end_valid]),3)))
plt.text(0, 0.6,'RMSE:'+str(np.round(sklearn.metrics.mean_squared_error(df_norac['hs'].loc[start_valid:end_valid], ts_pred.loc[start_valid:end_valid])**0.5,3)))
plt.xlabel('Hs from NORAC')
plt.ylabel('Hs from NORAC_pred')
plt.savefig(model+'-'+'_'.join(var_origin)+'scatter.png')
plt.close()
