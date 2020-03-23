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
    return y0*k / (y0 - (y0 - k) * np.exp(-r*t))

def r2(p):
    r, k = p
    return sum((ys - logistic(ts, ys[0], r, k))**2)

## Minimize square distances
p0 = (1, 2) # Initial guess
res = minimize(r2, p0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
rmin = res.x[0]
kmin = res.x[1]

## Plot results
fig, ax = plt.subplots(1, 1)
plt.plot(ts, ys, '.')
plt.plot(ts, logistic(ts, ys[0], rmin, kmin))