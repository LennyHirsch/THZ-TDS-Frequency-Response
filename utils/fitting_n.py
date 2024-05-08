import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Parsons.csv')

power = lambda x, a, b, c: a*(x**b) + c

a = 1346
b = -2.373
c = 3.34

x = np.linspace(min(df['wl']), max(df['wl']))

fit = power(x, a, b, c)

print(df.head())
plt.plot(df['wl'], df['n'])
plt.plot(x, fit)
plt.legend(['data', 'fit'])
plt.xlabel('Wavelength')
plt.ylabel('n')
plt.show()