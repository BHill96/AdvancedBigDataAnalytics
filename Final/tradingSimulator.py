#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 21:04:23 2020

@author: blakehillier
"""

from copy import deepcopy
import pandas as pd
from tqdm import tqdm
import datetime
from dateutil.relativedelta import relativedelta
#import XLNetFed
import numpy as np
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import optimizers

# Silence tensorflow and pandas warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
pd.options.mode.chained_assignment = None  # default='warn'

def turnDaily(stock, info):
    stockDate = stock.columns[0]
    infoDate = info.columns[0]
    daily = []
    colLabel = info.columns[1]
    i=len(info)-1
    j=len(stock)-1
    while j > -1 and i > -1:
        if info[infoDate][i] < stock[stockDate][j]:
            #print('{2}: {0}<{1}'.format(info[infoDate][i], stock[stockDate][j], j))
            daily.append(info[colLabel][i])
            j = j-1
        else:
            #print('{2}: {0}>{1}'.format(info[infoDate][i], stock[stockDate][j], i))
            i = i-1
    return daily[::-1]

"""
Loads all the macro data from csv files and converts to daily timeframe.
daily is the name of a stock
 - Must be stored in Data/Stocks directory
macro is the name of the macro data
 - Must be stored in the Data directory
* You do not need the file extension since we are using .csv only *
"""
def loadMacro(daily, macro):
    dailyData = pd.read_csv('Data/Stocks/'+daily+'.csv')
    dailyData.Date = pd.to_datetime(dailyData['Date'])
    dailyData.Date = dailyData['Date'].dt.normalize()
    dailyData.drop(columns=dailyData.columns[1:], inplace=True)
    # print(dailyData)

    for fileName in tqdm(macro):
        #print('Loading '+fileName)
        data = pd.read_csv('Data/'+fileName+'.csv').dropna()
        data.DATE = pd.to_datetime(data['DATE'])
        data.DATE = data['DATE'].dt.normalize()
        # print(data)
        # print(type(data.DATE[0]),type(data[data.columns[1]][0]))
        dailyData[fileName] = turnDaily(dailyData, data)

    return dailyData


def loadStocks():
    #Read in files
    # Should use the text file
    s_and_p = ['MMM','ABT','ABBV','ACN','ATVI','AYI','ADBE','AMD','AAP','AES','AET',
               'AMG','AFL','A','APD','AKAM','ALK','ALB','ARE','ALXN','ALGN','ALLE',
               'AGN','ADS','LNT','ALL','GOOGL','GOOG','MO','AMZN','AEE','AAL','AEP',
               'AXP','AIG','AMT','AWK','AMP','ABC','AME','AMGN','APH','ADI','ANDV',
               'ANSS','ANTM','AON','AOS','APA','AIV','AAPL','AMAT','APTV','ADM',
               'AJG','AIZ','T','ADSK','ADP','AZO','AVB','AVY','BLL','BAC','BK',
               'BAX','BDX','BBY','BIIB','BLK','HRB','BA','BWA','BXP','BSX',
               'BMY','AVGO','CHRW','CA','COG','CDNS','CPB','COF','CAH','CBOE',
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

    stocks = pd.read_csv('Data/Stocks/CI_data.csv').drop(['Open','High','Low','Close','Adj Close',
                                                          'Volume', 'Name'], axis=1)
    stocks.Date = pd.to_datetime(stocks['Date'])
    stocks.Date = stocks['Date'].dt.normalize()
    for tick in tqdm(s_and_p):
        stock = pd.read_csv('Data/Stocks/'+tick+'_data.csv').drop(['Open','High','Low','Adj Close',
                                                                   'Volume', 'Name'], axis=1)
        # stock = stock[stock['Date'] >= '1994-01-01'] ??
        apd = stock[stock['Date'] >= '1994-01-01'].copy()
        stocks[tick] = apd['Close']
    return stocks

def riskBins(stocks, numStocks, numRiskLevels):
    # Should we normalize it first?
    variance = stocks.var(axis=0)[-numStocks:].sort_values(ascending=True, axis=0)
    # If numRiskLevels does not easily divide numStocks, the riskiest bin will be smaller
    binSize = int(np.ceil(numStocks/numRiskLevels))
    bins = []
    for i in range(0, numRiskLevels-1):
        lb = i*binSize
        ub = (i+1)*binSize
        bins.append(variance.index[lb:ub])
    bins.append(variance.index[ub:])
    return bins

def createModel(shape):
    regressior = Sequential()
    regressior.add(LSTM(units=256, activation='relu', return_sequences=True, input_shape=(shape[1], 1)))
    regressior.add(Dropout(0.2))
    regressior.add(LSTM(units=128, activation='relu', return_sequences=True))
    regressior.add(Dropout(0.2))
    regressior.add(LSTM(units=64, activation='relu', return_sequences=True))
    regressior.add(Dropout(0.2))
    regressior.add(LSTM(units=32, activation='relu'))
    regressior.add(Dropout(0.2))
    regressior.add(Dense(units=1))
    return regressior

def lstm(data, ticker, epochs=5, batch=64):
    training = data.drop(columns=['DATE'], axis=1)
    training.drop(training.tail(1).index, inplace=True)

    #Create X,Y Train Set
    X = np.expand_dims(np.array(training), axis = 2)
    Y = np.array(data[ticker])[1:]
    # Build network Structure
    model = createModel(X.shape)

    # Compile and train model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Verbose=0 makes it silent
    model.fit(X, Y, epochs=epochs, batch_size=batch, verbose=0)
    return model

"""
Simulates the market from tBegin to tEnd while implimenting a long trading strategy
for each t it trains the XLNet and the LSTM based on 'current' data,
then forcasts to t+dt and calculates the expected return for each stock.
Ranks each stock by return and selects the top n stocks to invest in.
This repeats until tEnd.

T = [tBegin, tEnd] the interval of simulation (string yyyy/mm/dd)
dt = time until portfolio needs to be reavaluated (months)
n = number of best stocks to trade with after ranking
riskLevel = how risky you want the portfolio to be from 0 to numRiskLevels-1
xlnetMetric = name of macro file to use for sentiment analysis
xlnetMetricType = 'Daily' or 'Quarterly'
"""
def simulateMarket(T, dt, n, riskLevel, numRiskLevels, xlnetMetric, xlnetMetricType='Quarterly',
                   MAX_LEN=128, batch=24, epochs=10):
    # Turn to datetime
    T = pd.to_datetime(T)
    dt = relativedelta(months=+dt)

    # Load and merge datasets
    macroFiles = ['liborfinal','GDPC1','CPIAUCSL','MICH','UNRATENSA']
    macroDaily = loadMacro('UNH_data', macroFiles)
    stocks = loadStocks()
    forcastData = pd.merge(macroDaily, stocks, on='Date', how='outer').dropna(axis=1)
    forcastData.sort_values(['Date'], inplace=True, axis=0, ascending=True)
    forcastData.reset_index(inplace=True)
    forcastData.drop('index', axis=1, inplace=True)
    forcastData.rename(columns={'Date':'DATE'}, inplace=True)
    # Keep text separate
    text = pd.read_csv('Data/FedTextData.csv', names=['Date','Text'])
    text.Date = pd.to_datetime(text['Date'])
    text.Date = text['Date'].dt.normalize()
    text.sort_values(['Date'], inplace=True, axis=0, ascending=True)
    text.reset_index(inplace=True)
    text.drop('index', axis=1, inplace=True)

    # Prep data for simulation
    t = T[0]
    currentText = text[text.Date < t]
    currentNum = forcastData[forcastData.DATE < t]
    # Find number of stocks
    numStocks = len(forcastData.columns) - (len(macroFiles)+1)
    while t < T[1]:
        print('Selecting stocks based on risk...')
        bins = riskBins(stocks=currentNum, numStocks=numStocks, numRiskLevels=numRiskLevels)
        usableStockTickers = bins[riskLevel].tolist()
        usableStockTickers.append('DATE')
        usableStocks = currentNum[usableStockTickers]

        # Requires GPU
        print('Training XLNet...')
        """sentiment = XLNetFed.CalcSentiment(currentText, currentNum[['DATE',xlnetMetric]],
                                           metricType=xlnetMetricType)
        inpt, attMsk = XLNetFed.TextPrep(currentText, MAX_LEN=MAX_LEN)
        model = XLNetFed.Train(inpt[:-1], attMsk[:-1], list(sentiment.Econ_Perf), batch_size=batch,
                               epochs=epochs)
        print('Trained')"
        sentiment = XLNetFed.Predict(model, inpt, attMsk, batch)"""
        sentiment = [1 for _ in range(0, len(currentText))]
        currentText['Sentiment'] = sentiment
        currentNum['Sentiment'] = turnDaily(usableStocks[['DATE', usableStockTickers[0]]],
                                            currentText[['Date','Sentiment']])

        print('LSTM training...')
        for stock in tqdm(usableStocks):
            lstmColumns = np.append(macroFiles, ['DATE',stock,'Sentiment'])
            modelData = currentNum[lstmColumns]
            model = lstm(modelData, stock)
            print('trained')
            history = [modelData.iloc[-1]]
            print(history)
            #for date in forcastData[(forcastData.Date > t) & (forcastData.DATE <= t+dt)]:
             #   model

        print('Caclulating expected returns...')

        print('Ranking Stocks...')

        print('Buying best n stocks...')

        print('Updating data...')
        t += dt
        currentText = text[text.Date < t]
        currentNum = forcastData[forcastData.DATE < t]

        print('Calculating actual returns...')
