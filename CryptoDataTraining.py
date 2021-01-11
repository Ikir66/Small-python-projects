import pandas as pd
import numpy as np
from numpy import genfromtxt
import scipy as sp
from sklearn import preprocessing
import matplotlib.pyplot as plt

btcdf = pd.read_csv('~/Desktop/money/data/BTC-USD.csv')
xmrdf = pd.read_csv('~/Desktop/money/data/XMR-USD.csv')
ltcdf = pd.read_csv('~/Desktop/money/data/LTC-USD.csv')

btcM = btcdf.to_numpy()
xmrM = xmrdf.to_numpy()
ltcM = ltcdf.to_numpy()

dates = btcM[:,[0]]
btcPrices = np.transpose(btcM[:,[4]])
xmrPrices = np.transpose(xmrM[:,[4]])
ltcPrices = np.transpose(ltcM[:,[4]])

# 2255 datapoints

def maVector (vector, matime, time):
    i = matime - 1
    maVector = np.zeros([1,time])
    while i < time:
        sum = 0
        j = i - matime + 1
        c = 0
        while c < matime:
            sum += vector[0,j]
            c += 1
            j += 1
        maVector[0,i] = sum/matime
        i += 1
    return maVector

btcMA110 = maVector(btcPrices, 110, 2255)

print(btcMA110)

fig = plt.figure()
plt.tight_layout()
ax = plt.axes()
ax.plot(dates, btcPrices, label = "BTC Price");

#python3.7 nov2020.py
