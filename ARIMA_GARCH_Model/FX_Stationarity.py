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
plt.rcParams.update({'font.size': 10})

figsmall = (5.6,4)
figlarge = (6,2.5)
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

#create differenced
df_USDEUR_diff = df_USDEUR.copy()
df_USDEUR_diff['USDEUR Exchange Rate'] = df_USDEUR_diff['USDEUR Exchange Rate'].diff()
df_USDEUR_diff.dropna(inplace=True)
df_USDEUR_diff.rename(columns={'USDEUR Exchange Rate':'USDEUR Exchange Rate Difference'}, inplace=True)

#---------------------------------------------------------------------------------------------------------


#--------------------------------------------STATIONARITY TEST--------------------------------------------
#define function to test stationarity
def test_stationarity(timeseries, filename):
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    
    # Creating a DataFrame for the results
    dfoutput = pd.DataFrame(dftest[0:4]).transpose()
    dfoutput.columns = ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    dfoutput.index = ['Value']
    critical_values = {f'Critical Value ({key})': value for key, value in dftest[4].items()}
    critical_values_df = pd.DataFrame(critical_values, index=['Value'])
    results_df = pd.concat([dfoutput, critical_values_df], axis=1)
    
    # Formatting the DataFrame for LaTeX export
    results_df.to_latex(resultspath + filename,
                        index=False, 
                        float_format="{:0.3f}".format, 
                        caption="Results of Dickey-Fuller Test.",
                        label="tab:stationarityResults",
                        column_format="lcccccc",
                        header=True)

#run ADF test on df_USDEUR
test_stationarity(df_USDEUR['USDEUR Exchange Rate'], 'USD_EUR_exchange_rate_ADF.tex')

#run ADF test on df_USDEUR_diff
test_stationarity(df_USDEUR_diff['USDEUR Exchange Rate Difference'], 'USD_EUR_exchange_rate_diff_ADF.tex')

#plot df_USDEUR and df_USDEUR_diff in two different plots
plt.figure(figsize=figlarge)
plt.plot(df_USDEUR.set_index('Date', inplace=False), linewidth=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(resultspath+'USD_EUR_exchange_rate.pdf', dpi=600)

plt.figure(figsize=figlarge)
plt.plot(df_USDEUR_diff.set_index('Date', inplace=False), linewidth=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(resultspath+'USD_EUR_exchange_rate_difference.pdf', dpi=600)

plt.show()
