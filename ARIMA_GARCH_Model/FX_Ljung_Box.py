import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
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

#---------------------------------------------LJUNG BOX TEST----------------------------------------------
ljung_box_result = acorr_ljungbox(df_USDEUR_diff['USDEUR Exchange Rate Difference'], lags=20, return_df=True)


#print results
print('------RESULTS FOR 15 MIN FREQUENCY------')
print(ljung_box_result)

#save results to latex file 
ljung_box_result.to_latex(resultspath+'Ljung_Box.tex')

#plot Ljung Box for first 20 lags as bar plot
plt.figure(figsize=figsmall)
plt.scatter(ljung_box_result.index, ljung_box_result['lb_pvalue'])
plt.xlabel('Lag')   
plt.xticks(range(0,21,5))
plt.ylabel('')
plt.ylim(0,1)
plt.axhline(y=0.05, color='blue', linestyle='--')
plt.tight_layout()
plt.savefig(resultspath+'Ljung_Box_15.pdf', dpi=600)


print('--------------------------------------------ARIMA(0,1,1)--------------------------------------------')
#fit ARIMA(0,1,1) model to df_USDEUR
model = ARIMA(df_USDEUR_diff['USDEUR Exchange Rate Difference'], order=(0,0,1))
model_fit = model.fit()
print(model_fit.summary())



#test residuals for autocorrelation
residuals = model_fit.resid
ljung_box_result_residuals = acorr_ljungbox(residuals, lags=20, return_df=True)
ljung_box_result_residuals.to_latex(resultspath+'Ljung_Box_residuals011.tex')

plt.figure(figsize=figsmall)
plt.scatter(ljung_box_result_residuals.index, ljung_box_result_residuals['lb_pvalue'])
plt.xlabel('Lag')
plt.xticks(range(0,21,5))
plt.ylabel('')
plt.ylim(0,1)
plt.axhline(y=0.05, color='blue', linestyle='--')
plt.tight_layout()
plt.savefig(resultspath+'Ljung_Box_residuals011.pdf', dpi=600)



#print results
print('------RESULTS FOR RESIDUALS------')
print(ljung_box_result_residuals)


print('--------------------------------------------ARIMA(9,1,9)--------------------------------------------')
#fit ARIMA(9,1,9) model to df_USDEUR
model = ARIMA(df_USDEUR['USDEUR Exchange Rate'], order=(9,1,9))
model_fit = model.fit()
print(model_fit.summary())

pd.DataFrame(model_fit.summary().tables[1].data[1:], columns=model_fit.summary().tables[1].data[0]).to_latex(resultspath+'ARIMA_9_1_9_summary.tex')

#test residuals for autocorrelation
model = ARIMA(df_USDEUR_diff['USDEUR Exchange Rate Difference'], order=(9,0,9))
model_fit = model.fit()

residuals = model_fit.resid
ljung_box_result_residuals = acorr_ljungbox(residuals, lags=20, return_df=True)

plt.figure(figsize=figsmall)
plt.scatter(ljung_box_result_residuals.index, ljung_box_result_residuals['lb_pvalue'])
plt.xlabel('Lag')
plt.xticks(range(0,21,5))
plt.ylabel('')
plt.ylim(0,1)
plt.axhline(y=0.05, color='blue', linestyle='--')
plt.tight_layout()
plt.savefig(resultspath+'Ljung_Box_residuals_9_1_9.pdf', dpi=600)


plt.show()