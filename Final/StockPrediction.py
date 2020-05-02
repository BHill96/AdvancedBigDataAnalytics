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
textData = CalcSentiment(read_csv(dataDr+'FedTextData.csv', names=['Date','Text']), read_csv=read_csv(dir+'GDP.csv'))
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
gild = pd.read_csv('GILD.csv').drop(['Open','High','Low','Adj Close','Volume'], axis = 1)
ci = pd.read_csv('CI.csv').drop(['Open','High','Low','Adj Close','Volume'], axis = 1)
pfe = pd.read_csv('PFE.csv').drop(['Open','High','Low','Adj Close','Volume'], axis = 1)
hum = pd.read_csv('HUM.csv').drop(['Open','High','Low','Adj Close','Volume'], axis = 1)
Unh = pd.read_csv('UNH.csv').drop(['Open','High','Low','Adj Close','Volume'], axis = 1)
tnx = pd.read_csv('TNX1.csv').drop(['Open','High','Low','Adj Close','Volume'], axis = 1)
tnx.dropna(inplace = True)
libor = pd.read_csv('liborfinal.csv')
gdp = pd.read_csv('GDPC1.csv')
cpi = pd.read_csv('CPIAUCSL.csv')
inflation = pd.read_csv('MICH.csv')
unemployment = pd.read_csv('UNRATENSA.csv')

#Create Empytlists for conversion to daily data
gdpdaily = []
cpidaily = []
inflationdaily = []
unemploymentdaily = []


def dataclean(predictor, emptylist, filename):
    i = 0
    j = 0
    while i < len(predictor['Date']):
        if predictor['Date'][i] < predictor['DATE'][j]:
            emptylist.append(predictor[filename][j])
            i = i + 1
        else:
            j = j + 1


dataclean(gdp, gdpdaily, 'GDPC1')
dataclean(cpi, cpidaily, 'CPIAUCSL')
dataclean(inflation, inflationdaily, 'MICH')
dataclean(unemployment, unemploymentdaily, 'UNRATENSA')
        
Unh['ci'] = ci['Close']
Unh['gild'] = gild['Close']
Unh['hum'] = hum['Close']
Unh['pfe'] = pfe['Close']
#Unh['tnx'] = tnx['Close']
Unh['libor'] = libor[' value']
Unh['cpi'] = cpidaily
Unh['gdp'] = gdpdaily
Unh['I'] = inflationdaily
Unh['U'] = unemploymentdaily
Unh['Txt'] = textDaily['EconPerf']

#Create test and training sets
data_training1 = Unh[Unh['Date']<'2009-01-01'].copy()
data_test = Unh[Unh['Date']>='2009-01-01'].copy()

data_training1 = data_training1.drop(['Date'], axis = 1)
data_training = np.array(data_training1)

#Create X,Y Train Set
X_train = np.expand_dims(data_training, axis = 2)
y_train = data_training1[['Close','ci','gild','hum','pfe']]
y_train = np.array(y_train)

#Create X test Set, y actual set
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

regressior.add(Dense(units = 5))
regressior.summary()

regressior.compile(optimizer='adam', loss = 'mean_squared_error')
# This prevents overfitting, only use during final testing and training
callback = callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.01)
regressior.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=[callback])

# Test Results
y_pred = regressior.predict(X_test)

#Adjust actual stock value to have on same time scale on x axis
unh1 = np.array(Unh['Close'][(5287-len(data_test)):5287])
ci1 = np.array(Unh['ci'][(5287-len(data_test)):5287])
gild1 = np.array(Unh['gild'][(5287-len(data_test)):5287])
hum1 = np.array(Unh['hum'][(5287-len(data_test)):5287])
pfe1 = np.array(Unh['pfe'][(5287-len(data_test)):5287])

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

#Create Covariance and Correlation matrices, Show heatmap
mat = Unh.drop(columns = 'Date')
mat.rename(columns = {'Close' : 'unh'}, inplace = True)
sns.heatmap(mat.corr(), annot = True)

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
