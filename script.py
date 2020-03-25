#!/usr/bin/env python
"""
This is a quick and dirty script for fitting a time series to a logistic curve

It is intended to be read side by side with ./docs/Logistic.pdf
"""

## Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.optimize import minimize

## Import data
data = pd.read_csv('data.csv')
days_since_death = data['Days since dead #1']
deaths = data['Dead']
hospitalized = data['Hospitalized']
date = re.sub(r'\/','_', data['Date'][len(data['Date'])-1])
N = len(days_since_death)

## Build functions
def logistic(t, y0, r, k):
    """ Logistic function (see ./docs/Logistic.pdf) """
    return y0*k / (y0 - (y0 - k) * np.exp(-r*t))

def deaths_r2(p):
    """ Fitting function (see ./docs/Logistic.pdf) """
    r, k = p
    return sum((deaths - logistic(days_since_death, deaths[0], r, k))**2)

def hospitalized_r2(p):
    r, k = p
    return sum((hospitalized - logistic(days_since_death, hospitalized[0], r, k))**2)

## Minimize square distances
p0 = (1, 2) # Initial guess
deaths_res = minimize(deaths_r2, p0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})

p0 = (1, 10) # Initial guess
hospitalized_res = minimize(hospitalized_r2, p0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})

## Extract the optimal parameters
deaths_rmin = deaths_res.x[0]
deaths_kmin = deaths_res.x[1]
hospitalized_rmin = hospitalized_res.x[0]
hospitalized_kmin = hospitalized_res.x[1]

## Plot results for deaths
fig, ax = plt.subplots(1, 1)
plt.plot(days_since_death, deaths, '.') # Plot the data
plt.plot(days_since_death, logistic(days_since_death, deaths[0], deaths_rmin, deaths_kmin)) # Plot the optimal logistic curve
ax.set_title('Covid-19 deaths in the Netherlands')
ax.set_xlabel('Days since first death')
ax.set_ylabel('Deaths')

plt.savefig('figs/'+date+'_deaths.png')

fig, ax = plt.subplots(1, 1)
plt.plot(days_since_death, hospitalized, '.') # Plot the data
plt.plot(days_since_death, logistic(days_since_death, hospitalized[0], hospitalized_rmin, hospitalized_kmin)) # Plot the optimal logistic curve
ax.set_title('Covid-19 hospitalizations in the Netherlands')
ax.set_xlabel('Days since first death')
ax.set_ylabel('Hospitalizations (cumulative)')

plt.savefig('figs/'+date+'_hospitalizations.png')

print('deaths_rmin: {}'.format(deaths_rmin))
print('deaths_kmin: {}'.format(deaths_kmin))

print('hospitalized_rmin: {}'.format(hospitalized_rmin))
print('hospitalized_kmin: {}'.format(hospitalized_kmin))
