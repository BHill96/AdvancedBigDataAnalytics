#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 21:04:23 2020

@author: blakehillier
"""

import pandas as pd
from tqdm import tqdm

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
    dailyData['Date'] = pd.to_datetime(dailyData['Date'])
    dailyData['Date'] = dailyData['Date'].dt.normalize()
    dailyData.drop(columns=dailyData.columns[1:], inplace=True)
    # print(dailyData)

    for fileName in tqdm(macro):
        #print('Loading '+fileName)
        data = pd.read_csv('Data/'+fileName+'.csv').dropna()
        data['DATE'] = pd.to_datetime(data['DATE'])
        data['DATE'] = data['DATE'].dt.normalize()
        # print(data)
        # print(type(data.DATE[0]),type(data[data.columns[1]][0]))
        dailyData[fileName] = turnDaily(dailyData, data)

    return dailyData


def loadStocks():
    #Read in files
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

    stocks = pd.read_csv('Data/Stocks/CI_data.csv').drop(['Open','High','Low','Close','Adj Close', 'Volume', 'Name'], axis=1)
    stocks['Date'] = pd.to_datetime(stocks['Date'])
    stocks['Date'] = stocks['Date'].dt.normalize()
    for tick in tqdm(s_and_p):
        stock = pd.read_csv('Data/Stocks/'+tick+'_data.csv').drop(['Open','High','Low','Adj Close','Volume', 'Name'], axis=1)
        # stock = stock[stock['Date'] >= '1994-01-01'] ??
        apd = stock[stock['Date'] >= '1994-01-01'].copy()
        stocks[tick] = apd['Close']
    return stocks

def turnDaily(stock, info):
    daily = []
    colLabel = info.columns[1]
    i=len(info)-1
    j=len(stock)-1
    while j > -1 and i > -1:
        if info['DATE'][i] < stock['Date'][j]:
            # print('{2}: {0}<{1}'.format(info['DATE'][i], stock['Date'][j], j))
            daily.append(info[colLabel][i])
            j = j-1
        else:
            # print('{2}: {0}>{1}'.format(info['DATE'][i], stock['Date'][j], i))
            i = i-1
    return daily[::-1]



macroDaily = loadMacro('unh_data', ['liborfinal','GDPC1','CPIAUCSL','MICH','UNRATENSA'])

stocks = loadStocks()
forcastData = pd.merge(macroDaily, stocks, on='Date', how='outer').dropna()
text = pd.read_csv('Data/FedTextData.csv', names=['Date','Text'])
text['Date'] = pd.to_datetime(text['Date'])
text['Date'] = text['Date'].dt.normalize()

"""
Load Text
Combine into one data frame
Separate into two dataframes, one with date <= tBegin and the other date > tBegin

t = tBegin
while t < tEnd
    Train XLNet
    Train LSTM
    Forcast stocks to time period t+1
    Calculate expected returns
    Rank stocks by returns
    Pick best n stocks to buy
    Jump to t+1
"""