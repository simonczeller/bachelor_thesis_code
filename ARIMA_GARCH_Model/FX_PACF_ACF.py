import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams.update({'font.size': 20})

figsmall = (5.6,4)
figlarge = (6,4)
os.chdir('...')

resultspath = '...'

#--------------------------------------------DATA PREPROCESSING--------------------------------------------
# Load raw data
df_rawEURUSD = pd.read_csv('...')

# Keep only data from Sunday 21:00 to Friday 21:00
df_rawEURUSD['Date'] = pd.to_datetime(df_rawEURUSD['datetime'])
df_EURUSD = df_rawEURUSD[~df_rawEURUSD['Date'].apply(lambda x: (x.weekday() == 4 and x.hour > 21) or (x.weekday() == 5) or (x.weekday() == 6 and x.hour < 21))]
df_EURUSD.reset_index(inplace=True)

#create USD/EUR dataframe
df_USDEUR = df_EURUSD.copy()
df_USDEUR['Schluss'] = 1/df_USDEUR['bid_close']

df_USDEUR.dropna(inplace=True)

#remove all columns except 'Date' and 'Schluss'
df_USDEUR = df_USDEUR[['Date', 'Schluss']]

#rename 'Schluss' column to 'USDEUR Exchange Rate'
df_USDEUR.rename(columns={'Schluss':'USDEUR Exchange Rate'}, inplace=True)

#sort by date
df_USDEUR.sort_values(by='Date', inplace=True)
df_USDEUR.reset_index(drop=True, inplace=True)

#create differenced series
df_USDEUR_diff = df_USDEUR.copy()
df_USDEUR_diff['USDEUR Exchange Rate'] = df_USDEUR_diff['USDEUR Exchange Rate'].diff()
df_USDEUR_diff.dropna(inplace=True)
df_USDEUR_diff.rename(columns={'USDEUR Exchange Rate':'USDEUR Exchange Rate Difference'}, inplace=True)

#---------------------------------------------------------------------------------------------------------

#----------------------------------------------ACF AND PACF-----------------------------------------------

#save plot_acf and plot_pacf to png
fig, ax = plt.subplots(figsize=figsmall)
plot_acf(df_USDEUR_diff['USDEUR Exchange Rate Difference'], lags=range(1,101), title='', auto_ylims=True, ax=ax)
plt.savefig(resultspath+'ACF_USDEUR_15.pdf', dpi=600)

fig, ax = plt.subplots(figsize=figsmall)
plot_pacf(df_USDEUR_diff['USDEUR Exchange Rate Difference'], lags=range(1,101), title='', auto_ylims=True, ax=ax)
plt.savefig(resultspath+'PACF_USDEUR_15.pdf', dpi=600)
