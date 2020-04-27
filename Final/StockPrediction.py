#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 01:01:13 2020

@author: blakehillier
"""

from pandas import read_csv, read_excel, concat, merge_asof, DataFrame, to_datetime
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from XLNetFed import CalcSentiment, TextPrep
from pytorch_transformers import XLNetForSequenceClassification, AdamW
from tqdm import trange
import numpy as np
from numpy import argmax
from tensorflow.keras import Sequential, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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

# Train XLNet
EPOCHS = 10
BATCH_SIZE = 24
dataDr = 'Data/'
unh = read_csv(dataDr+'UNH.csv')
textData = CalcSentiment(read_csv(dataDr+'FedTextData.csv', names=['Date','Text']), unh)
inpts, attMsks = TextPrep(textData)

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
textDaily = turnDaily(unh, textPred)
textDaily = DataFrame(textDaily)    
textDaily.dropna()
textDaily.rename(columns = {'0':'EconPerf'}, inplace = True)

# Begin RNN LSTM
# Read in files
cpi = read_csv(dataDr+'CPIMonthly.csv')
cpi['Date'] = to_datetime(cpi['Date'])
cpi['Date'] = cpi['Date'].dt.normalize()

gdp = read_csv(dataDr+'GDPQuarterly.csv')
gdp['Date'] = to_datetime(gdp['Date'])
gdp['Date'] = gdp['Date'].dt.normalize()

inflation = read_csv(dataDr+'InflationRateMonthly.csv')
inflation['Date'] = to_datetime(inflation['Date'])
inflation['Date'] = inflation['Date'].dt.normalize()

unemployment = read_csv(dataDr+'UnemploymentRateMonthly.csv')
unemployment['Date'] = to_datetime(unemployment['Date'])
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


frames = [unemploymentDaily, inflationDaily, gdpDaily, textDaily]
econ = concat(frames, axis = 1)

tnx = read_excel(dataDr+'TNX Daily.xlsx')
tnx.dropna(inplace=True)

libor = read_csv(dataDr+'LIBORDaily.csv').drop(['Ln Changes'], axis=1)
libor['Date'] = to_datetime(libor['Date'])
libor['Date'] = libor['Date'].dt.normalize()

# Merge Dataframes
final = merge_asof(unh, tnx, on = 'Date')
final = final.iloc[1:]
data = merge_asof(final, libor, on='Date')

data = concat((data, econ), axis = 1)

# Create test and training sets
data_training1 = data[data['Date']<'2007-01-01'].copy()
data_test = data[data['Date']>='2007-01-01'].copy()

data_training1 = data_training1.drop(['Date'], axis = 1)

data_training = np.array(data_training1)

# Create X,Y Train Set
X_train = np.expand_dims(data_training, axis = 2)
y_train = data_training1['Close_x']
y_train = np.array(y_train)

# Create X test Set, y actual set
data_test = data_test.drop(['Date'], axis = 1)
data_test = np.array(data_test)
X_test = np.expand_dims(data_test, axis = 2)

# Adjust shape as needed for data
# Create RNN LSTM
regressior = Sequential()

regressior.add(LSTM(units = 256, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 128, activation = 'relu', return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 64, activation = 'relu', return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 32, activation = 'relu'))
regressior.add(Dropout(0.2))

regressior.add(Dense(units = 1))
regressior.summary()

regressior.compile(optimizer='adam', loss = 'mean_squared_error')
# This prevents overfitting, only use during final testing and training
callback = callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.01)
regressior.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=[callback])

# Test Results
y_pred = regressior.predict(X_test)

unhClose = unh['Close']
unhClose = unhClose[1759:3773,]
unhClose = np.array(unhClose)

# Plot predicted vs actual
plt.figure(figsize=(14,5))
plt.plot(unhClose, color = 'red', label = 'Real UNH Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted UNH Stock Price')
plt.title('UNH Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('UNH Stock Price')
plt.legend()
# PDF is just for quick checking of figure
plt.savefig('StockPredXLNet.pdf')
plt.savefig('StockPredXLNet.pgf', transparent=True)
plt.show()

# Find MSE
tom = y_pred[:,0]
tom = tom.tolist()
print('MSE: {0}'.format(mean_squared_error(unhClose,tom)))