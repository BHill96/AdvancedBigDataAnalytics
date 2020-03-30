# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 21:15:54 2020

@author: Joe
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#Read in files
dataDr = 'Data/'

cpi = pd.read_csv(dataDr+'CPIMonthly.csv')
cpi['Date'] = pd.to_datetime(cpi['Date'])
cpi['Date'] = cpi['Date'].dt.normalize()

gdp = pd.read_csv(dataDr+'GDPQuarterly.csv')
gdp['Date'] = pd.to_datetime(gdp['Date'])
gdp['Date'] = gdp['Date'].dt.normalize()

inflation = pd.read_csv(dataDr+'InflationRateMonthly.csv')
inflation['Date'] = pd.to_datetime(inflation['Date'])
inflation['Date'] = inflation['Date'].dt.normalize()

unemployment = pd.read_csv(dataDr+'UnemploymentRateMonthly.csv')
unemployment['Date'] = pd.to_datetime(unemployment['Date'])
unemployment['Date'] = unemployment['Date'].dt.normalize()

unh = pd.read_csv(dataDr+'UNH.csv').drop(['LN Close','Volume','LN Volume'], axis=1)
unh['Date'] = pd.to_datetime(unh['Date'])
unh['Date'] = unh['Date'].dt.normalize()

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
        
gdpDaily = turnDaily(unh, gdp)
gdpDaily = pd.DataFrame(gdpDaily)    
gdpDaily.dropna()
gdpDaily.rename(columns = {'0':'GDP'}, inplace = True)

cpiDaily = turnDaily(unh, cpi)
cpiDaily = pd.DataFrame(cpiDaily)    
cpiDaily.dropna()
cpiDaily.rename(columns = {'0':'GDP'}, inplace = True)

inflationDaily = turnDaily(unh, inflation)
inflationDaily = pd.DataFrame(inflationDaily)    
inflationDaily.dropna()
inflationDaily.rename(columns = {'0':'GDP'}, inplace = True)

unemploymentDaily = turnDaily(unh, unemployment)
unemploymentDaily = pd.DataFrame(unemploymentDaily)    
unemploymentDaily.dropna()
unemploymentDaily.rename(columns = {'0':'GDP'}, inplace = True)
#%%
frames = [unemploymentDaily, inflationDaily, gdpDaily]
econ = pd.concat(frames, axis = 1)

tnx = pd.read_excel(dataDr+'TNX Daily.xlsx')
tnx.dropna(inplace=True)

libor = pd.read_csv(dataDr+'LiborDaily.csv').drop(['Ln Changes'], axis=1)
libor['Date'] = pd.to_datetime(libor['Date'])
libor['Date'] = libor['Date'].dt.normalize()

unh['Date'] = pd.to_datetime(unh['Date'])
unh['Date'] = unh['Date'].dt.normalize()

#Merge Dataframes
final = pd.merge_asof(unh,tnx, on = 'Date')
print(final)
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
from tensorflow.keras import Sequential, callbacks
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
# This prevents overfitting, only use during final testing and training
callback = callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.05)
regressior.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=[callback])

#%%
#Run LSTM on Test, output predictions

y_pred = regressior.predict(X_test)
#%%
#Adjust actual stock value to have on same time scale on x axis
unhClose = unh['Close']
unhClose = unhClose[1759:3773,]
unhClose = np.array(unhClose)

#%%
#Plot predicted vs actual
plt.figure(figsize=(14,5))
plt.plot(unhClose, color = 'red', label = 'Real UNH Stock Price')
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
mean_squared_error(unhClose,tom)





