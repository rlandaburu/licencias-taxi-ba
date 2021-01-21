#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:39:32 2021

@author: federicolandaburu
"""
import streamlit as st
import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric
import base64

data = pd.read_csv("Taxis.csv", sep=";")
data['ds'] = pd.to_datetime(data['ds'],errors='coerce') 

max_date = data['ds'].max()

periods_input = 12
m = Prophet(seasonality_mode='multiplicative')
m.fit(data)
 
future = m.make_future_dataframe(periods=periods_input, freq= 'MS')
    
forecast = m.predict(future)
fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

fcst_filtered =  fcst[fcst['ds'] > max_date] 

fig1 = m.plot(forecast)   
fig2 = m.plot_components(forecast)


st.write(fcst_filtered)
st.write(fig1)
st.write(fig2)





