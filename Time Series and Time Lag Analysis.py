# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:58:47 2023

@author: joeal
"""
#First, we'll need to import the necessary libraries and download the historical data using yfinance:
    
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download the historical data for three cryptocurrencies
btc = yf.download("BTC-USD", start="2017-01-01", end="2023-05-07")
eth = yf.download("ETH-USD", start="2017-01-01", end="2023-05-07")
xrp = yf.download("XRP-USD", start="2017-01-01", end="2023-05-07")
    
# Combine the data into a single DataFrame
data = pd.concat([btc['Close'], eth['Close'], xrp['Close']], axis=1)
data.columns = ['BTC', 'ETH', 'XRP']

#Next, let's take a look at the data to see what it looks like:
    
print(data.head())
print(data.tail())

#This will print the first and last 5 rows of the DataFrame. We can also plot the data to visualize how the cryptocurrencies have performed over time:

#The code seems fine except that it does not handle missing or infinite values. To fix this, we can add the following line after downloading the data:
# Handle missing or infinite values
data = data.replace([np.inf, -np.inf], np.nan).dropna()
#This code replaces any infinite values with NaN and then drops any rows containing NaN values from the DataFrame. This ensures that the data is free of missing or infinite values before performing any analysis.

fig, ax = plt.subplots(3, 1)
ax[0].plot(data['BTC'],color='red',label='BTC')
ax[0].set_ylabel('Price (USD)')
ax[0].set_title('BTC Daily Closing Price')
ax[1].plot(data['ETH'],color='green',label='ETH')
ax[1].set_ylabel('Price (USD)')
ax[1].set_title('ETH Daily Closing Price')
ax[2].plot(data['XRP'],color='blue',label='XRP')
ax[2].set_ylabel('Price (USD)')
ax[2].set_title('XRP Daily Closing Price')
plt.show()



#This will generate a plot of the price of each cryptocurrency over time.


#To perform time series analysis, we need to convert the data into a time series by setting the date as the index of the DataFrame:
    
data.index = pd.to_datetime(data.index)

#Now that we have a time series, we can calculate the rolling mean and rolling standard deviation to identify trends and volatility in the data:

# Calculate the monthly and biannual rolling mean and standard deviation
rolmean30 = data.rolling(window=30).mean()
rolstd30 = data.rolling(window=30).std()
rolmean180 = data.rolling(window=180).mean()
rolstd180 = data.rolling(window=180).std()

# Plot the rolling statistics
fig, ax = plt.subplots(3, 1, figsize=(10,5))
ax[0].plot(data['BTC'],label='BTC')
ax[0].plot(rolmean30['BTC'],label='Monthly Rolling Mean')
ax[0].plot(rolstd30['BTC'],label='Monthly Rolling Standard Deviation')
ax[0].plot(rolmean180['BTC'],label='Biannual Rolling Mean')
ax[0].plot(rolstd180['BTC'],label='Binannual Rolling Standard Deviation')
ax[0].set_title('Rolling Mean and Rolling Standard Deviation (BTC)')
ax[0].set_ylabel('Price (USD)')
ax[0].legend()
ax[1].plot(data['ETH'],label='ETH')
ax[1].plot(rolmean30['ETH'],label='Monthly Rolling Mean')
ax[1].plot(rolstd30['ETH'],label='Monthly Rolling Standard Deviation')
ax[1].plot(rolmean180['ETH'],label='Biannual Rolling Mean')
ax[1].plot(rolstd180['ETH'],label='Binannual Rolling Standard Deviation')
ax[1].set_title('Rolling Mean and Rolling Standard Deviation (ETH)')
ax[1].set_ylabel('Price (USD)')
ax[1].legend()
ax[2].plot(data['XRP'],label='XRP')
ax[2].plot(rolmean30['XRP'],label='Monthly Rolling Mean')
ax[2].plot(rolstd30['XRP'],label='Monthly Rolling Standard Deviation')
ax[2].plot(rolmean180['XRP'],label='Biannual Rolling Mean')
ax[2].plot(rolstd180['XRP'],label='Binannual Rolling Standard Deviation')
ax[2].set_title('Rolling Mean and Rolling Standard Deviation (XRP)')
ax[2].set_ylabel('Price (USD)')
ax[2].legend()

plt.show()


#This will generate a plot of the original data along with the rolling mean and rolling standard deviation. We can see how the price of each cryptocurrency has changed over time and identify any trends or patterns.

plt.figure()


#We can also perform a Dickey-Fuller test to test for stationarity in the data:

from statsmodels.tsa.stattools import adfuller

# Perform Dickey-Fuller test
result_btc = adfuller(data['BTC'].dropna())
result_eth = adfuller(data['ETH'].dropna())
result_xrp = adfuller(data['XRP'].dropna())

# Define the critical values at certain significance levels
critical_values = {
    '1%': -3.430,
    '5%': -2.862,
    '10%': -2.567
}


# Print the test statistic and p-value
print(f'BTC ADF Statistic: {result_btc[0]}')
print(f'BTC p-value: {result_btc[1]}')
print(f'ETH ADF Statistic: {result_eth[0]}')
print(f'ETH p-value: {result_eth[1]}')
print(f'XRP ADF Statistic: {result_xrp[0]}')
print(f'XRP p-value: {result_xrp[1]}')

# Perform differencing to transform non-stationary series into stationary
data_diff = data.diff().dropna()

# Perform Dickey-Fuller test on the differenced series
result_diff_btc = adfuller(data_diff['BTC'].dropna())
result_diff_eth = adfuller(data_diff['ETH'].dropna())
result_diff_xrp = adfuller(data_diff['XRP'].dropna())

# Print the test statistic, p-value, and critical values after differencing
print(f'Differenced BTC ADF Statistic: {result_diff_btc[0]}')
print(f'Differenced BTC p-value: {result_diff_btc[1]}')
for level, crit_value in critical_values.items():
    print(f'Differenced BTC Critical Value at {level}: {crit_value}')
    if result_diff_btc[0] < crit_value:
        print('Differenced BTC ADF Statistic is under the critical value at', level)
    else:
        print('Differenced BTC ADF Statistic is not under the critical value at', level)
print()

print(f'Differenced ETH ADF Statistic: {result_diff_eth[0]}')
print(f'Differenced ETH p-value: {result_diff_eth[1]}')
for level, crit_value in critical_values.items():
    print(f'Differenced ETH Critical Value at {level}: {crit_value}')
    if result_diff_eth[0] < crit_value:
        print('Differenced ETH ADF Statistic is under the critical value at', level)
    else:
        print('Differenced ETH ADF Statistic is not under the critical value at', level)
print()

print(f'Differenced XRP ADF Statistic: {result_diff_xrp[0]}')
print(f'Differenced XRP p-value: {result_diff_xrp[1]}')
for level, crit_value in critical_values.items():
    print(f'Differenced XRP Critical Value at {level}: {crit_value}')
    if result_diff_xrp[0] < crit_value:
        print('Differenced XRP ADF Statistic is under the critical value at', level)
    else:
        print('Differenced XRP ADF Statistic is not under the critical value at', level)



#The Dickey-Fuller test helps us determine if the data is stationary or not. A p-value less than 0.05 indicates that the data is stationary. If the p-value is greater than 0.05, we can assume that the data is non-stationary.

#We can also plot the autocorrelation and partial autocorrelation functions to identify any seasonality or trends in the data:
    
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot the autocorrelation and partial autocorrelation functions
fig, ax = plt.subplots(3, 3, figsize=(10,5))
ax[0, 0].plot(data_diff['BTC'],color='red'); ax[0, 0].set_title('BTC Diff')
ax[0, 1].plot(data_diff['ETH'],color='green'); ax[0, 1].set_title('ETH Diff')
ax[0, 2].plot(data_diff['XRP'],color='blue'); ax[0, 2].set_title('XRP Diff')
plot_acf(data_diff['BTC'], ax=ax[1, 0],color='red')
plot_acf(data_diff['ETH'], ax=ax[1, 1],color='green')
plot_acf(data_diff['XRP'], ax=ax[1, 2],color='blue')
plot_pacf(data_diff['BTC'], ax=ax[2, 0],color='red')
plot_pacf(data_diff['ETH'], ax=ax[2, 1],color='green')
plot_pacf(data_diff['XRP'], ax=ax[2, 2],color='blue')

plt.show()

#This will generate a plot of the differenced data for each cryptocurrency and the autocorrelation function for each differenced series.

plt.figure()





#Finally, we can fit a time series model to the data using an ARIMA model.

from statsmodels.tsa.arima_model import ARIMA

# Choose the lag order based on the significant spikes in the autocorrelation plot
order_btc = (1, 1, 1)  # ARIMA(p, d, q) order for BTC
order_eth = (1, 1, 1)  # ARIMA(p, d, q) order for ETH
order_xrp = (1, 1, 1)  # ARIMA(p, d, q) order for XRP

# Fit the ARIMA model to the data
model_btc = ARIMA(data['BTC'], order=order_btc)
model_eth = ARIMA(data['ETH'], order=order_eth)
model_xrp = ARIMA(data['XRP'], order=order_xrp)

# Fit the model to the data
model_btc_fit = model_btc.fit()
model_eth_fit = model_eth.fit()
model_xrp_fit = model_xrp.fit()

# Forecast the next 30 days
forecast_btc_30 = model_btc_fit.forecast(steps=30)[0]
forecast_eth_30 = model_eth_fit.forecast(steps=30)[0]
forecast_xrp_30 = model_xrp_fit.forecast(steps=30)[0]

# Forecast the next 6 months
forecast_btc_180 = model_btc_fit.forecast(steps=180)[0]
forecast_eth_180 = model_eth_fit.forecast(steps=180)[0]
forecast_xrp_180 = model_xrp_fit.forecast(steps=180)[0]

# Generate the dates for the forecast
last_date = data.index[-1]
forecast_dates_30 = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=30, freq='D')
forecast_dates_180 = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=180, freq='D')

# Plot the forecasts along with the original data
fig, ax = plt.subplots(3, 1, figsize=(10, 5))
ax[0].plot(data.index, data['BTC'], label='Original')
ax[0].plot(forecast_dates_30, forecast_btc_30, label='Monthly Forecast')
ax[0].plot(forecast_dates_180, forecast_btc_180, label='6 Month Forecast')
ax[0].set_title('BTC Forecast')
ax[1].plot(data.index, data['ETH'], label='Original')
ax[1].plot(forecast_dates_30, forecast_eth_30, label='Monthly Forecast')
ax[1].plot(forecast_dates_180, forecast_eth_180, label='6 Month Forecast')
ax[1].set_title('ETH Forecast')
ax[2].plot(data.index, data['XRP'], label='Original')
ax[2].plot(forecast_dates_30, forecast_xrp_30, label='Monthly Forecast')
ax[2].plot(forecast_dates_180, forecast_xrp_180, label='6 Month Forecast')
ax[2].set_title('XRP Forecast')
plt.legend()
plt.show()

plt.figure()
# Extract the residuals from the fitted models
residuals_btc = pd.DataFrame(model_btc_fit.resid)
residuals_eth = pd.DataFrame(model_eth_fit.resid)
residuals_xrp = pd.DataFrame(model_xrp_fit.resid)

# Plot the residuals
fig, ax = plt.subplots(3, 1, figsize=(10, 5))
ax[0].plot(residuals_btc, label='BTC', color='red')
ax[1].plot(residuals_eth, label='ETH', color='green')
ax[2].plot(residuals_xrp, label='XRP', color='blue')
ax[0].set_title('BTC Residuals')
ax[1].set_title('ETH Residuals')
ax[2].set_title('XRP Residuals')
plt.show()




    
