#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 01:01:13 2020

@author: blakehillier
"""

from pandas import read_csv, read_excel, concat, merge_asof, DataFrame, to_datetime
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from XLNetFed import CalcSentiment, TextPrep
from transformers import XLNetForSequenceClassification, AdamW
from tqdm import trange
import tqdm
import numpy as np
from numpy import argmax
from tensorflow.keras import Sequential, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

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
    training.drop(training.tail(1).index, inplace=True)

    #Create X,Y Train Set
    X = np.expand_dims(np.array(training), axis=2)
    Y = np.array(data[ticker])[1:]
    # Build network Structure
    model = createModel(X.shape)

    # Compile and train model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Verbose=0 makes it silent
    model.fit(X, Y, epochs=epochs, batch_size=batch, verbose=0)
    return model

# Train XLNet
EPOCHS = 10
BATCH_SIZE = 28
MAX_LEN = 128
dataDr = 'Data/'
print('XLNet...')
textData = CalcSentiment(read_csv(dataDr+'FedTextData.csv', names=['Date','Text']),
                         read_csv=read_csv(dataDr+'liborfinal.csv'))
inpts, attMsks = TextPrep(textData, MAX_LEN=MAX_LEN)

# Turn data into torch tensors
inpts = torch.tensor(inpts)
labels = torch.tensor(list(textData.Econ_Perf))
masks = torch.tensor(attMsks)

# Create Iterators of the datasets
trainData = TensorDataset(inpts, masks, labels)
trainSampler = RandomSampler(trainData)
trainDataloader = DataLoader(trainData, sampler=trainSampler, batch_size=BATCH_SIZE)

model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)
# Loads model into GPU memory
model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias','gamma','beta']
optimizer_grouped_parameters = [
    {'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate':0.01},
    {'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate':0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

# Find GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for _ in trange(EPOCHS, desc='Epoch'):
    # Train
    model.train()
    for step, batch in enumerate(trainDataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        # Forward pass and loss calculation
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        logits = outputs[1]
        # Calculate gradients
        loss.backward()
        # Update weights using gradients
        optimizer.step()

# Evaluate text
model.eval()
evalData = TensorDataset(inpts, masks, labels)
evalDataloader = DataLoader(evalData, batch_size=BATCH_SIZE)
labels = np.array([])
for batch in evalDataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    # Don't calculate gradients since we are evaluating the model
    with torch.no_grad():
        output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = output[0]
    logits = logits.detach().cpu().numpy()
    labels = np.append(labels, argmax(logits, axis=1).flatten())

textPred = DataFrame(textData.Date, columns=['Date'])
textPred['Label'] = labels
unh = pd.read_csv('Data/Stocks/UNH_data.csv')
unh = pd.to_datetime(unh['Date'])
unh = unh['Date'].dt.normalize()
unh.drop(columns=unh.columns[1:], inplace=True)
textDaily = turnDaily(unh, textPred)
textDaily = DataFrame(textDaily)
textDaily.dropna()
textDaily.rename(columns = {'0':'EconPerf'}, inplace = True)

# Begin RNN LSTM
print('Loading Stocks...')
stocks = loadStocks()

# Read in files
macroFiles = ['liborfinal','GDPC1','CPIAUCSL','MICH','UNRATENSA']
macroDaily = loadMacro('UNH_data', macroFiles)

data = pd.merge(macroDaily, stocks, on='Date', how='outer').dropna(axis=1)
data.sort_values(['Date'], inplace=True, axis=0, ascending=True)
data.reset_index(inplace=True)
data.drop('index', axis=1, inplace=True)
data.rename(columns={'Date':'DATE'}, inplace=True)

data_training = data[data['DATE']<'2009-01-01']
data_test = data[data['DATE']>='2009-01-01']

#Create test and training sets
mae = []
print('Testing Stocks...')
for stock in stocks.columns:
    modelColumns = np.append(macroFiles, stock)
    training = data_training[modelColumns]
    test = data_test[modelColumns]

    model = lstm(training, stock)

    # Test Results
    actual = np.array(test[stock])[1:]
    test = test.drop(test.tail(1).index)
    X = np.expand_dims(np.array(test), axis=2)

    pred = model.predict(X)
    predict = pred.tolist()
    mae.append([stock, mean_absolute_error(predict,actual)])
    print('Stock {0} MAE {1}'.format(stock, mae))

mae = pd.DataFrame(mae)
mae.to_csv('Data/finalModelMAE.csv')


"""
def prediction(predict, actual, stockname):
    plt.figure(figsize=(14,5))
    plt.plot(actual, color = 'red', label = 'Real' + stockname + 'Stock Price')
    plt.plot(predict, color = 'blue', label = 'Predicted' + stockname + 'Stock Price')
    plt.title(stockname + ' Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(stockname +  ' Stock Price')
    plt.legend()
    plt.show()

prediction(y_pred[:,0],unh1,'UNH')
prediction(y_pred[:,1],ci1,'CI')
prediction(y_pred[:,2],gild1,'GILD')
prediction(y_pred[:,3],hum1,'HUM')
prediction(y_pred[:,4],pfe1,'PFE')

#Plot all stock predicts and actuals
plt.figure(figsize=(14,5))
plt.plot(pfe1, color = 'red', label = 'Real UNH Stock Price')
plt.plot(unh1, color = 'cyan', label = 'Real UNH Stock Price')
plt.plot(gild1, color = 'green', label = 'Real GILD Stock Price')
plt.plot(hum1, color = 'magenta', label = 'Real HUM Stock Price')
plt.plot(ci1, color = 'yellow', label = 'Real CI Stock Price')
plt.plot(y_pred[:,0], color = 'blue', label = 'Predicted UNH Stock Price')
plt.plot(y_pred[:,1], color = 'black', label = 'Predicted CI Stock Price')
plt.plot(y_pred[:,2], color = '0.9', label = 'Predicted GILD Stock Price')
plt.plot(y_pred[:,3], color = '0.75', label = 'Predicted HUM Stock Price')
plt.plot(y_pred[:,4], color = '0.5', label = 'Predicted PFE Stock Price')
plt.title('UNH Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('UNH Stock Price')
plt.legend()
plt.show()


# PDF is just for quick checking of figure
plt.savefig('StockPredXLNet.pdf')
plt.savefig('StockPredXLNet.pgf', transparent=True)
plt.show()

# Print MSE for each stock and Residual Plot
def metrics(predict, actual, stock):
    res = []
    for i,j in zip(predict,actual):
        res.append(abs(i-j))
    plt.plot(res)
    plt.ylabel('Residual Squared')
    plt.xlabel('Time (Days)')
    plt.title(stock + ' Residuals')
    plt.show()
    #Determine MSE
    predict = predict.tolist()
    actual = actual.tolist()
    print(mean_squared_error(predict,actual))


metrics(y_pred[:,0],unh1, 'UNH')
metrics(y_pred[:,1],ci1, 'CI')
metrics(y_pred[:,2],gild1, 'GILD')
metrics(y_pred[:,3],hum1, 'HUM')
metrics(y_pred[:,4],pfe1, 'PFE')
"""