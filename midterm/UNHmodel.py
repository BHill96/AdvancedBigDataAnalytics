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

#os.chdir('C:\\Users\\Joe\\Documents\\CGU\\Advanced Big Data Analytics\\Project')
#Read in files

#y_test = pd.read_excel('UNH.xlsx').drop(['LN Close','Volume','LN Volume'], axis = 1)

gdpdaily = []
cpidaily = []
inflationdaily = []
unemploymentdaily = []
cpi = pd.read_excel('CPI Monthly.xlsx')
gdp = pd.read_excel('Real GDP Quarterly.xlsx')
inflation = pd.read_excel('Inflation Rate Monthly.xlsx')
unemployment = pd.read_excel('Unemployment Rate Monthly.xlsx')

i=0
j=0
while i < len(gdp['Date']):
    if gdp['Date'][i] < gdp['Date.1'][j]:
        gdpdaily.append(gdp['GDPC1'][j])
        i = i+1
    else:
        j = j+1
        
gdpdaily = pd.DataFrame(gdpdaily)    
gdpdaily.dropna()
gdpdaily.rename(columns = {'0':'GDP'}, inplace = True)

i=0
j=0
while i < len(cpi['Date']):
    if cpi['Date'][i] < cpi['Date.1'][j]:
        cpidaily.append(cpi['CPIAUCSL'][j])
        i = i+1
    else:
        j = j+1
        
cpidaily = pd.DataFrame(cpidaily)    
cpidaily.dropna()


i=0
j=0
while i < len(inflation['Date']):
    if inflation['Date'][i] < inflation['Date.1'][j]:
        inflationdaily.append(inflation['Inflation'][j])
        i = i+1
    else:
        j = j+1
        
inflationdaily = pd.DataFrame(inflationdaily)    
inflationdaily.dropna()
inflationdaily.rename(columns = {'0':'I'}, inplace = True)

i=0
j=0
while i < len(unemployment['Date']):
    if unemployment['Date'][i] < unemployment['Date.1'][j]:
        unemploymentdaily.append(unemployment['UNRATE'][j])
        i = i+1
    else:
        j = j+1
        
unemploymentdaily = pd.DataFrame(unemploymentdaily)    
unemploymentdaily.dropna()
unemploymentdaily.rename(columns = {'0':'U'}, inplace = True)
#%%
frames = [unemploymentdaily, inflationdaily, gdpdaily]
econ = pd.concat(frames, axis = 1)

unh = pd.read_excel('UNH.xlsx').drop(['LN Close','Volume','LN Volume'], axis=1)
tnx = pd.read_excel('TNX Daily.xlsx')
libor = pd.read_excel('1 Year Libor Daily.xlsx').drop(['Ln Changes'], axis=1)

tnx.dropna(inplace=True)

#Merge Dataframes
final = pd.merge_asof(unh,tnx, on = 'Date')
final = final.iloc[1:]
data = pd.merge_asof(final,libor, on='Date')


data=pd.concat((data,econ), axis = 1)
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
data_test = data_test.drop(['Date'], axis = 1)
data_test = np.array(data_test)
X_test = np.expand_dims(data_test, axis = 2)
#%%
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

#Adjust shape as needed for data

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
regressior.fit(X_train, y_train, epochs=50, batch_size=32)

#%%
#Run LSTM on Test, output predictions

y_pred = regressior.predict(X_test)
#%%
#Adjust actual stock value to have on same time scale on x axis
unh1 = unh['Close']
unh1 = unh1[1759:3773,]
unh1 = np.array(unh1)

#%%
#Plot predicted vs actual
plt.figure(figsize=(14,5))
plt.plot(unh1, color = 'red', label = 'Real UNH Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted UNH Stock Price')
plt.title('UNH Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('UNH Stock Price')
plt.legend()
plt.show()

#%%
from sklearn.metrics import mean_squared_error
tom = y_pred[:,0]
tom = tom.tolist()
unh1 = unh1.tolist()
mean_squared_error(unh1,tom)





