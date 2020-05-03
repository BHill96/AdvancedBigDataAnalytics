# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:34:42 2020

@author: Joe
"""

import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv, read_excel, concat, merge_asof, DataFrame, to_datetime

def turnDaily(stock, info):
    daily = []
    colLabel = info.columns[1]
    i=len(info)-1
    j=len(stock)-1
    while j > -1 and i > -1:
        if info['Date'][i] < stock['Date'][j]:
            daily.append(info[colLabel][i])
            j = j-1
        else:
            i = i-1
    return daily[::-1]

unh = pd.read_csv('AdvancedBigDataAnalytics/Final/Data/Stocks/UNH_data.csv')
unh['Date'] = pd.to_datetime(unh['Date'])
libor = pd.read_csv('AdvancedBigDataAnalytics/Final/Data/liborfinal.csv')
gdp = pd.read_csv('AdvancedBigDataAnalytics/Final/Data/GDPC1.csv')
cpi = pd.read_csv('AdvancedBigDataAnalytics/Final/Data/CPIAUCSL.csv')
inflation = pd.read_csv('AdvancedBigDataAnalytics/Final/Data/MICH.csv')
unemployment = pd.read_csv('AdvancedBigDataAnalytics/Final/Data/UNRATENSA.csv')

cpi['Date'] = pd.to_datetime(cpi['Date'])
cpi['Date'] = cpi['Date'].dt.normalize()


gdp['Date'] = pd.to_datetime(gdp['Date'])
gdp['Date'] = gdp['Date'].dt.normalize()

inflation['Date'] = pd.to_datetime(inflation['Date'])
inflation['Date'] = inflation['Date'].dt.normalize()


unemployment['Date'] = pd.to_datetime(unemployment['Date'])
unemployment['Date'] = unemployment['Date'].dt.normalize()
        
gdpDaily = turnDaily(unh, gdp)
gdpDaily = DataFrame(gdpDaily)    
gdpDaily.dropna()
gdpDaily.rename(columns = {'0':'GDP'}, inplace = True)

cpiDaily = turnDaily(unh, cpi)
cpiDaily = DataFrame(cpiDaily)    
cpiDaily.dropna()
cpiDaily.rename(columns = {'0':'CPI'}, inplace = True)

inflationDaily = turnDaily(unh, inflation)
inflationDaily = DataFrame(inflationDaily)    
inflationDaily.dropna()
inflationDaily.rename(columns = {'0':'Inflation'}, inplace = True)

unemploymentDaily = turnDaily(unh, unemployment)
unemploymentDaily = DataFrame(unemploymentDaily)    
unemploymentDaily.dropna()
unemploymentDaily.rename(columns = {'0':'Unempl'}, inplace = True)

libor = read_csv('AdvancedBigDataAnalytics/Final/Data/liborfinal.csv')
libor['Date'] = pd.to_datetime(libor['Date'])
libor['Date'] = libor['Date'].dt.normalize()




#Read in files
s_and_p = ['MMM','ABT','ABBV','ACN','ATVI','AYI','ADBE','AMD','AAP','AES','AET',
        'AMG','AFL','A','APD','AKAM','ALK','ALB','ARE','ALXN','ALGN','ALLE',
        'AGN','ADS','LNT','ALL','GOOGL','GOOG','MO','AMZN','AEE','AAL','AEP',
        'AXP','AIG','AMT','AWK','AMP','ABC','AME','AMGN','APH','ADI','ANDV',
        'ANSS','ANTM','AON','AOS','APA','AIV','AAPL','AMAT','APTV','ADM',
        'AJG','AIZ','T','ADSK','ADP','AZO','AVB','AVY','BLL','BAC','BK',
        'BAX','BDX','BBY','BIIB','BLK','HRB','BA','BWA','BXP','BSX'
        ,'BMY','AVGO','CHRW','CA','COG','CDNS','CPB','COF','CAH','CBOE',
        'KMX','CCL','CAT','CBS','CNC','CNP','CTL','CERN','CF','SCHW',
        'CHTR','CHK','CVX','CMG','CB','CHD','CI','XEC','CINF','CTAS','CSCO','C','CFG',
        'CTXS','CLX','CME','CMS','KO','CTSH','CL','CMCSA','CMA','CAG','CXO','COP',
        'ED','STZ','COO','GLW','COST','COTY','CCI','CSX','CMI','CVS','DHI',
        'DHR','DRI','DVA','DE','DAL','XRAY','DVN','DLR','DFS','DISCA','DISCK','DISH',
        'DG','DLTR','D','DOV','DTE','DRE','DUK','DXC','ETFC','EMN','ETN',
        'EBAY','ECL','EIX','EW','EA','EMR','ETR','EVHC','EOG','EQT','EFX','EQIX','EQR',
        'ESS','EL','ES','RE','EXC','EXPE','EXPD','ESRX','EXR','XOM','FFIV','FB','FAST',
        'FRT','FDX','FIS','FITB','FE','FISV','FLIR','FLS','FLR','FMC','FL','F',
		'FBHS','BEN','FCX','GPS','GRMN','IT','GD','GE','GIS','GM','GPC','GILD',
		'GPN','GS','GT','GWW','HAL','HBI','HOG','HIG','HAS','HCA','HP','HSIC',
		'HSY','HES','HLT','HOLX','HD','HON','HRL','HST','HPQ','HUM','HBAN','HII',
		'IDXX','INFO','ITW','ILMN','INTC','ICE','IBM','INCY','IP','IPG','IFF','INTU',
		'ISRG','IVZ','IQV','IRM','JEC','JBHT','SJM','JNJ','JCI','JPM','JNPR','KSU','K','KEY',
		'KMB','KIM','KMI','KLAC','KSS','KR','LB','LH','LRCX','LEG','LEN',
		'LLY','LNC','LKQ','LMT','L','LOW','LYB','MTB','MAC','M','MRO','MPC','MAR','MMC','MLM',
		'MAS','MA','MAT','MKC','MCD','MCK','MDT','MRK','MET','MTD','MGM','MCHP','MU',
		'MSFT','MAA','MHK','TAP','MDLZ','MON','MNST','MCO','MS','MOS','MSI','MYL','NDAQ',
		'NOV','NAVI','NTAP','NFLX','NWL','NFX','NEM','NWSA','NWS','NEE','NLSN','NKE','NI',
		'NBL','JWN','NSC','NTRS','NOC','NCLH','NRG','NUE','NVDA','ORLY','OXY','OMC','OKE',
		'ORCL','PCAR','PKG','PH','PDCO','PAYX','PNR','PBCT','PEP','PKI','PRGO','PFE',
		'PCG','PM','PSX','PNW','PXD','PNC','RL','PPG','PPL','PFG','PG','PGR',
		'PLD','PRU','PEG','PSA','PHM','PVH','PWR','QCOM','DGX','RRC','RJF','RTN','O',
		'REG','REGN','RF','RSG','RMD','RHI','ROK','COL','ROP','ROST','RCL','CRM','SBAC',
		'SLB','SNI','STX','SEE','SRE','SHW','SIG','SPG','SWKS','SLG','SNA','SO','LUV',
		'SPGI','SWK','SBUX','STT','SRCL','SYK','STI','SYF','SNPS','SYY','TROW','TPR',
		'TGT','TEL','FTI','TXN','TXT','TMO','TIF','TWX','TJX','TSCO','TDG','TRV',
		'TRIP','TSN','UDR','ULTA','USB','UAA','UNP','UAL','UNH','UPS','URI',
		'UTX','UHS','UNM','VFC','VLO','VAR','VTR','VRSN','VRSK','VZ','VRTX','V','VNO',
		'VMC','WMT','WBA','DIS','WM','WAT','WEC','WFC','WDC','WU','WY','WHR','WMB',
		'WLTW','WYNN','XEL','XRX','XLNX','XL','XYL','YUM','ZBH','ZION','ZTS']

stocks = pd.DataFrame()
x = pd.read_csv('AdvancedBigDataAnalytics/Final/Data/Stocks/CI_data.csv').drop(['Open','High','Low','Close','Adj Close', 'Volume', 'Name'], axis = 1)
stocks['Date'] = x['Date']
for i in s_and_p:
    s = pd.read_csv('AdvancedBigDataAnalytics/Final/Data/Stocks/' + i + '_data.csv').drop(['Open','High','Low','Adj Close','Volume', 'Name'], axis = 1)
    apd = s[s['Date'] >= '1994-01-01'].copy()
    stocks[i] = apd['Close']

stocks.dropna(axis = 'columns', inplace = True)


stocks['libor'] = libor[' value']
stocks['cpi'] = cpiDaily
stocks['gdp'] = gdpDaily
stocks['I'] = inflationDaily
stocks['U'] = unemploymentDaily



#%%
data_training1 = stocks[stocks['Date']<'2009-01-01'].copy()
data_test = stocks[stocks['Date']>='2009-01-01'].copy()

data_training1 = data_training1.drop(data_training1.loc[:, 'Date':'TGT'], axis = 1)
data_training = np.array(data_training1)

#Create X,Y Train Set
X_train = np.expand_dims(data_training, axis = 2)
y_train = data_training1.drop(['libor','cpi','gdp','I','U'], axis = 1)
y_train = np.array(y_train)

#Create X test Set, y actual set
data_test = data_test.drop(data_test.loc[:,'Date':'TGT'], axis = 1)
data_test = np.array(data_test)
X_test = np.expand_dims(data_test, axis = 2)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import optimizers

# All parameter gradients will be clipped to
# a maximum norm of 1.


#Adjust shape as needed for data

regressior = Sequential()

regressior.add(LSTM(units = 256, activation = 'relu',  return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 128, activation = 'relu', return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 128, activation = 'relu', return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 64, activation = 'relu'))
regressior.add(Dropout(0.2))

regressior.add(Dense(units = 35))
regressior.summary()

#opt = optimizers.SGD(lr=0.001, clipnorm = 1)
regressior.compile(optimizer='adam', loss = 'mean_absolute_error')
regressior.fit(X_train, y_train, epochs=5, batch_size=8)

#%%
#Run LSTM on Test, output predictions

y_pred = regressior.predict(X_test)