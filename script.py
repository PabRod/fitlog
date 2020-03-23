""" 
This is a quick and dirty script for fitting a time series to a logistic curve

It is intended to be read side by side with ./docs/Logistic.pdf
"""

## Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

## Import data
data = pd.read_csv('data.csv')
ts = data['Days since dead #1']
ys = data['Dead']
N = len(ts)

## Build functions
def logistic(t, y0, r, k):
    """ Logistic function (see ./docs/Logistic.pdf) """
    return y0*k / (y0 - (y0 - k) * np.exp(-r*t))

def r2(p):
    """ Fitting function (see ./docs/Logistic.pdf) """
    r, k = p
    return sum((ys - logistic(ts, ys[0], r, k))**2)

## Minimize square distances
p0 = (1, 2) # Initial guess
res = minimize(r2, p0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})

## Extract the optimal parameters
rmin = res.x[0]
kmin = res.x[1]

## Plot results
fig, ax = plt.subplots(1, 1)
plt.plot(ts, ys, '.') # Plot the data
plt.plot(ts, logistic(ts, ys[0], rmin, kmin)) # Plot the optimal logistic curve
ax.set_title('Cases vs. time')
ax.set_xlabel('Days since first death')
ax.set_ylabel('Deaths')