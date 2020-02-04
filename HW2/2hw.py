# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:03:30 2020

@author: Joe
         Blake
"""

import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.cluster import KMeans
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
#%%
random.seed()


full= ['abt','adp','aep','aig','axp','ba','bac','bax','bdx','bk','bmy','bp','cat','cl','clx'
       ,'cmi','cop','cpb','csx','cvs','cvx','d','de','dhr','duk','eix','emr','es','etn','etr',
       'exc','fdx','fitb','gd','ge1','gis', 'glw','gpc','gww','hal','hban','hpq','hsy','hum','ibm1',
       'iff','intc','ip','jnj1','jpm','k','klac','kmb','ko', 'kr','l','leg','lhx','lly','lnc','mkc','mlhr',
       'mmm','mo','mrk','msi','mtb','nem','nsc','ntrs','omc','pcar','peg','pep','pfe','pg','pnw','ppg',
       'ppl','rtn','shw','slb','so','spgi','tgt','tmo','tsn','txn','unp','usb','utx','vfc','wba',
       'wfc','whr','wmb','wy','xel','xom','xrx']

#comb = combinations(full,3)
sample = []

comb = []
while len(full) > 2:
    temp = random.sample(range(0,len(full)),3)
    comb.append([full[temp[0]],full[temp[1]],full[temp[2]]])
    full.remove(comb[-1][0])
    full.remove(comb[-1][1])
    full.remove(comb[-1][2])

for i in comb:

  
    data1 = pd.read_excel(i[0] + '.xlsx').drop(['Date', 'PX_VOLUME'], axis=1)   
    data2 = pd.read_excel(i[1] + '.xlsx').drop(['Date', 'PX_VOLUME'], axis=1)
    data3 = pd.read_excel(i[2] + '.xlsx').drop(['Date', 'PX_VOLUME'], axis=1)
    data = data1.merge(data2, left_index=True, right_index=True).merge(data3, left_index=True, right_index=True)
    
    
    data = data.pct_change()
    data.replace([np.inf, -np.inf], np.nan)
    data.dropna(inplace=True)
    
    
    covariance = data.cov()
    U, D, V= np.linalg.svd(covariance)
    
    sample.append([i,D])
    # only for recreacion of matrix
    #U = U.T
    #V = V.T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for j in sample:
    ax.scatter(j[1][0], j[1][1], j[1][2], color='blue')

plt.show()
#%%

df = pd.DataFrame(sample,columns = ['stocks', 'sv'])
df.info()


#%%
cluster = KMeans(n_clusters = 2, n_jobs = -1).fit(df['sv'])
print(cluster.labels_)


