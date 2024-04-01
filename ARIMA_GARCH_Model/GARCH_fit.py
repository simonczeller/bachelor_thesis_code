import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch
from arch.unitroot import VarianceRatio
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model

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


#--------------------------------------------ARIMA fitting--------------------------------------------

#fit model
print('Fitting ARIMA(9,1,9) model...')
model919 = ARIMA(df_USDEUR_diff['USDEUR Exchange Rate Difference'], order=(9,0,9))
model919_fit = model919.fit()

#plot residuals
plt.figure(figsize=figsmall)
plt.plot(df_USDEUR['Date'][1:],model919_fit.resid)
plt.xticks(rotation=45)
x_label = plt.xlabel('')
y_label = plt.ylabel('')
plt.title('')
plt.tight_layout()

plt.savefig(resultspath+'ARIMA919_residuals.pdf', dpi=600)

#save plot_acf and plot_pacf of residuals to png
fig, ax = plt.subplots(figsize=figsmall)
plot_acf(model919_fit.resid**2, lags=range(1,101), title='', auto_ylims=True, ax=ax)
plt.savefig(resultspath+'ACFresid_919.pdf', dpi=600)

fig, ax = plt.subplots(figsize=figsmall)
plot_pacf(model919_fit.resid**2, lags=range(1,101), title='', auto_ylims=True, ax=ax)
plt.savefig(resultspath+'PACFresid_919.pdf', dpi=600)

#arch test on residuals
for lags in range(1,21):
    arch_test_result919 = het_arch(model919_fit.resid, nlags=lags, ddof=18)
    print("Lags:", lags, "P-Value:", arch_test_result919[1])


#fit model
print('Fitting ARIMA(0,1,1) model...')
model011 = ARIMA(df_USDEUR_diff['USDEUR Exchange Rate Difference'], order=(0,0,1))
model011_fit = model011.fit()

#plot residuals
plt.figure(figsize=figsmall)
plt.plot(df_USDEUR['Date'][1:],model011_fit.resid)
plt.xticks(rotation=45)
x_label = plt.xlabel('')
y_label = plt.ylabel('')
plt.title('')
plt.tight_layout()

plt.savefig(resultspath+'ARIMA011_residuals.pdf', dpi=600)

#arch test on residuals
for lags in range(1,21):
    arch_test_result011 = het_arch(model011_fit.resid, nlags=lags, ddof=1)
    print("Lags:", lags, "P-Value:", arch_test_result011[1])

#save plot_acf and plot_pacf to png
fig, ax = plt.subplots(figsize=figsmall)
plot_acf(model011_fit.resid**2, lags=range(1,101), title='', auto_ylims=True, ax=ax)
plt.savefig(resultspath+'ACFresid_011.pdf', dpi=600)

fig, ax = plt.subplots(figsize=figsmall)
plot_pacf(model011_fit.resid**2, lags=range(1,101), title='', auto_ylims=True, ax=ax)
plt.savefig(resultspath+'PACFresid_011.pdf', dpi=600)



#--------------------------------------------GARCH fitting (9,1,9)--------------------------------------------
#fit GARCH(1,2) model to residuals
print('Fitting GARCH(1,2) model...')
garch_model = arch_model(model919_fit.resid, vol='Garch', p=1, q=2)
garch_model_fit = garch_model.fit()
print(garch_model_fit.summary())

# Extract the parameter estimates table from the GARCH model summary
table = garch_model_fit.summary().tables[2]

# Convert the table to a pandas DataFrame
df_garch_summary = pd.DataFrame(data=[row[:len(table.data[0])] for row in table.data[1:]], columns=table.data[0])

# Save the DataFrame as a LaTeX file
df_garch_summary.to_latex(resultspath+'GARCH12_919summary.tex')

#diagnostics
garch_residuals_standardized = garch_model_fit.std_resid

#plot standardized residuals
plt.figure(figsize=figsmall)
plt.plot(df_USDEUR['Date'][1:],garch_residuals_standardized)
plt.xticks(rotation=45)
x_label = plt.xlabel('')
y_label = plt.ylabel('')
plt.title('')
plt.tight_layout()

plt.savefig(resultspath+'GARCH12_919standardized_residuals.pdf', dpi=600)

#ljung box for standardized residuals
ljung_box_result_residuals11919 = acorr_ljungbox(garch_residuals_standardized, lags=20, return_df=True)
ljung_box_result_residuals11919.to_latex(resultspath+'Ljung_Box_residuals919_GARCH_1_2.tex')

plt.figure(figsize=figsmall)
plt.scatter(ljung_box_result_residuals11919.index, ljung_box_result_residuals11919['lb_pvalue'])
plt.xlabel('Lag')
plt.xticks(range(0,21,5))
plt.ylabel('p-value')
plt.ylim(0,1)
plt.axhline(y=0.05, color='blue', linestyle='--')
plt.tight_layout()
plt.savefig(resultspath+'Ljung_Box_residuals12919.pdf', dpi=600)

print(ljung_box_result_residuals11919)


#--------------------------------------------GARCH fitting (0,1,1)--------------------------------------------
#fit GARCH(1,2) model to residuals
print('Fitting GARCH(1,2) model...')
garch_model = arch_model(model011_fit.resid, vol='Garch', p=1, q=2)
garch_model_fit = garch_model.fit()
print(garch_model_fit.summary())

# Extract the parameter estimates table from the GARCH model summary
table = garch_model_fit.summary().tables[2]

# Convert the table to a pandas DataFrame
df_garch_summary = pd.DataFrame(data=[row[:len(table.data[0])] for row in table.data[1:]], columns=table.data[0])


# Save the DataFrame as a LaTeX file
df_garch_summary.to_latex(resultspath+'GARCH12_011summary.tex')



#diagnostics
garch_residuals_standardized = garch_model_fit.std_resid

#plot standardized residuals
plt.figure(figsize=figsmall)
plt.plot(df_USDEUR['Date'][1:],garch_residuals_standardized)
plt.xticks(rotation=45)
x_label = plt.xlabel('')
y_label = plt.ylabel('')
plt.title('')
plt.tight_layout()

plt.savefig(resultspath+'GARCH11_011standardized_residuals.pdf', dpi=600)

#ljung box for standardized residuals
ljung_box_result_residuals12011 = acorr_ljungbox(garch_residuals_standardized, lags=20, return_df=True)
ljung_box_result_residuals12011.to_latex(resultspath+'Ljung_Box_residuals011_GARCH_1_3.tex')

plt.figure(figsize=figsmall)
plt.scatter(ljung_box_result_residuals12011.index, ljung_box_result_residuals12011['lb_pvalue'])
plt.xlabel('Lag')
plt.xticks(range(0,21,5))
plt.ylabel('p-value')
plt.ylim(0,1)
plt.axhline(y=0.05, color='blue', linestyle='--')
plt.tight_layout()
plt.savefig(resultspath+'Ljung_Box_residuals12011.pdf', dpi=600)

print(ljung_box_result_residuals12011)



#bruteforce GARCH model fitting for different p and q and save list with order, aic and bic
results919 = []
print("brute force")
for p in range(1, 6):
    for q in range(0, 6):
        print(f'Fitting GARCH({p},{q}) model...')
        garch_model = arch_model(model919_fit.resid, vol='Garch', p=p, q=q)
        garch_model_fit = garch_model.fit()
        results919.append({'Order': (p, q), 'AIC': garch_model_fit.aic, 'BIC': garch_model_fit.bic})

results919 = pd.DataFrame(results919)

# Find and print the order with minimum AIC and BIC
min_aic_order919 = results919.loc[results919['AIC'].idxmin(), 'Order']
min_bic_order919 = results919.loc[results919['BIC'].idxmin(), 'Order']


# Subtract minimum AIC and BIC from all AIC and BIC values
min_aic = results919['AIC'].min()
min_bic = results919['BIC'].min()
results919['AIC'] = results919['AIC'] - min_aic
results919['BIC'] = results919['BIC'] - min_bic


#bruteforce GARCH model fitting for different p and q and save list with order, aic and bic
results = []
print("brute force")
for p in range(1, 6):
    for q in range(0, 6):
        print(f'Fitting GARCH({p},{q}) model...')
        garch_model = arch_model(model011_fit.resid, vol='Garch', p=p, q=q)
        garch_model_fit = garch_model.fit()
        results.append({'Order': (p, q), 'AIC': garch_model_fit.aic, 'BIC': garch_model_fit.bic})

results = pd.DataFrame(results)

# Find and print the order with minimum AIC and BIC
min_aic_order = results.loc[results['AIC'].idxmin(), 'Order']
min_bic_order = results.loc[results['BIC'].idxmin(), 'Order']


# Subtract minimum AIC and BIC from all AIC and BIC values
min_aic = results['AIC'].min()
min_bic = results['BIC'].min()
results['AIC'] = results['AIC'] - min_aic
results['BIC'] = results['BIC'] - min_bic


print("Minimum AIC Order:", min_aic_order919)
print("Minimum BIC Order:", min_bic_order919)

print(results919)


print("Minimum AIC Order:", min_aic_order)
print("Minimum BIC Order:", min_bic_order)

print(results)




plt.show()