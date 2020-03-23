# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 21:15:54 2020

@author: Joe
"""
import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


#Read in files

jpm = pd.read_excel('JPM.xlsx').drop(['LN Price','Volume','LN Volume'], axis=1)
tnx = pd.read_excel('TNX Daily.xlsx')
libor = pd.read_excel('1 Year Libor Daily.xlsx').drop(['Ln Changes'], axis=1)

tnx.dropna(inplace=True)

#Merge Dataframes

final = pd.merge_asof(jpm,tnx, on = 'Date')
data = pd.merge_asof(final,libor, on='Date')

#Create test and training sets

data_training1 = data[data['Date']<'2007-01-01'].copy()
data_test = data[data['Date']>='2007-01-01'].copy()

data_training1 = data_training1.drop(['Date'], axis = 1)

data_training = np.array(data_training1)

#Create X,Y Train Set
X_train = np.expand_dims(data_training, axis = 2)
y_train = data_training1['Close_x']
y_train = np.array(y_train)

#Create X test Set, y actual set
y_test = data_test['Close_x']
data_test = data_test.drop(['Date'], axis = 1)
data_test = np.array(data_test)
X_test = np.expand_dims(data_test, axis = 2)


#%%
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

#Adjust shape as needed for data

regressior = Sequential()

regressior.add(LSTM(units = 256, activation = 'linear', return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 128, activation = 'linear', return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 64, activation = 'linear', return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 32, activation = 'linear'))
regressior.add(Dropout(0.2))

regressior.add(Dense(units = 1))
regressior.summary()

regressior.compile(optimizer='adam', loss = 'mean_squared_error')
regressior.fit(X_train, y_train, epochs=5, batch_size=32)

#%%
#Run LSTM on Test, output predictions

y_pred = regressior.predict(X_test)
#%%
#Adjust actual stock value to have on same time scale on x axis
jpm = jpm['Close']
jpm = jpm[1760:3773,]
jpm = np.array(jpm)

#%%
#Plot predicted vs actual
plt.figure(figsize=(14,5))
plt.plot(jpm, color = 'red', label = 'Real JPM Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted JPM Stock Price')
plt.title('JPM Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('JPM Stock Price')
plt.legend()
plt.show()







