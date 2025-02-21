from metocean_ml import examples as ml
import pandas as pd
import os

#Import  NORA3 test data

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data/NORA3_test.csv')
df = pd.read_csv(filename,comment='#',index_col=0, parse_dates=True)

def test_predict_ts_GBR(df=df):
    ml.predict_ts(ts_origin=df,var_origin=['hs','tp'],ts_train=df.loc['2015-01-01':'2016-12-31'],var_train=['hs'], model='GBR')

def test_predict_ts_SVR(df=df):
    ml.predict_ts(ts_origin=df,var_origin=['hs','tp'],ts_train=df.loc['2015-01-01':'2016-12-31'],var_train=['hs'], model='SVR_RBF')

def test_predict_ts_LSTM(df=df):
    ml.predict_ts(ts_origin=df,var_origin=['hs','tp'],ts_train=df.loc['2015-01-01':'2016-12-31'],var_train=['hs'], model='LSTM')


  
