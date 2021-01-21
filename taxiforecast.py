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

st.sidebar.header('Elegi cuantos meses queres proyectar')
periods_input = st.sidebar.selectbox("Meses", (12,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24))

m = Prophet(seasonality_mode='multiplicative')
m.fit(data)
 
future = m.make_future_dataframe(periods=periods_input, freq= 'MS')
    
forecast = m.predict(future)
fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

fcst_filtered =  fcst[fcst['ds'] > max_date] 

fig1 = m.plot(forecast)   
fig2 = m.plot_components(forecast)


st.title("Prediccion de venta de licencias de taxi en 2021 en base ventas pasadas")
st.write(round(fcst_filtered['yhat'].sum(),0), 'Licencias estimadas en 2021')

st.subheader('Cantidad de licencias estimadas a vender por mes')
st.write(fcst_filtered[['ds','yhat']])
st.subheader('Licencias vendidas en negro y estimacion en azul')
st.write(fig1)
st.subheader('Tendencia de los ultimos años')
st.write(fig2)








