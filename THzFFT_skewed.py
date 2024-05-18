from scipy.io import loadmat
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft, rfft, rfftfreq, fftfreq
from scipy.integrate import cumulative_trapezoid
from scipy.special import erf
import numpy as np
import pandas as pd
# import xarray as xr
import csv

def fft_thz(sig, pos, pad, step_size, start, end):
    c = 299792458
    dt = step_size / c
    zeros = [0.0] * pad

    diff = len(sig) - end # difference between full signal and single-cycle pulse
    zero_diff = [0.0] * diff

    sig = zeros + sig[start:end] + zeros + zero_diff # slice only the single cycle, then pad
    # sig = zeros + sig + zeros
    time = [s / c for s in pos]
    time = [max(time) - t for t in time]

    time_pad = []
    for t in range(pad*2):
        time_pad.append(max(time) + t*dt)

    time = time + time_pad

    duration = max(time)
    n = len(sig)
    sampleRate = n/duration

    yf = rfft(sig)
    xf = rfftfreq(n, 1/sampleRate)

    return (xf, yf, sig, time)

def normalise(array):
    maximum = max(np.abs(array))
    return [a/maximum for a in array]

# IMPORT RAW DATA

path = './experimental-data/'
# filename = 'datastruct_uncompressed - 23-11-14.csv'
# filename = 'datastruct_uncompressed - prelim 1 - 23-11-14.csv'
# filename = 'datastruct_uncompressed - 23-06-23-unpurged.csv'
filename = 'datastruct_uncompressed - 1299 chop - 23-06-24.csv'
# filename = 'datastruct_uncompressed - 22-07-11.csv'
# filename = 'datastruct_compressed - 23-05-02.csv'

sig = []
pos = []
std = []

with open (path + filename, newline='') as file:
    reader = csv.reader(file, delimiter=',', quotechar='|')
    for row in reader:
        if len(row):
            sig.append(float(row[0]))
            pos.append(float(row[1]))
            std.append(float(row[2]))

sig = [s + 2.2e-4 for s in sig] # ADD ZERO OFFSET

# FOURIER TRANSFORM SIGNAL, JUST TO GET PADDED TIME AND SIGNAL
pad = 500
step_size = 2e-6

(freq, fft, sig_padded, time_padded) = fft_thz(sig, pos, pad, step_size, 20, 740)

sig_padded = np.array(sig_padded)
time_padded = np.array(time_padded)

# DEFINING SKEWED GAUSSIAN
width = lambda tau: tau*np.sqrt(np.pi)
gaussian = lambda t, t0, w: np.exp(-(t-t0)**2 / w**2)
cfd = lambda t: 0.5*(1 + erf(t/np.sqrt(2)))

def skewed_gaussian(t, t0, tau, a):
    phi = 2/width(tau) * gaussian(t,t0,width(tau)) * cfd(a*(t-t0 / width(tau)))
    return phi

# MODIFYING RESULTS
a = 1
t0 = 6e-12
tau = 0.9e-12
factor = 0.8758

# mod = [skewed_gaussian(t,t0,tau,a) for t in time_padded] # GENERATE SKEWED GAUSSIAN MODIFYING FUNCTION
mod = skewed_gaussian(time_padded,t0,tau,a)
mod = normalise(mod) # NORMALISE SO FACTOR WORKS AS EXPECTED
mod = [m*factor for m in mod] # CHANGE MODIFYING FUNCTION BY SOME FACTOR

sig_modded = [s*(1 + m) for s, m in zip(sig_padded, mod)] # GENERATE MODIFIED SIGNAL
sig_modded = normalise(sig_modded) # NORMALISE MODIFIED SIGNAL FOR NICER PLOTTING. TODO: SHOULDN'T MAKE ANY MATERIAL DIFFERENCE?
integ = cumulative_trapezoid(sig_modded, time_padded[0:1730]) # INTEGRATE MODIFIED SIGNAL

print(integ[-1])

plt.plot(time_padded[0:1730], sig_padded)
plt.title('Padded signal')
plt.show()

plt.plot(time_padded, ( mod ))
plt.plot(time_padded[0:1730], normalise( sig_modded ))
plt.title('Modifying function and Modified signal')
plt.legend(['Modifying function', 'Modified signal'])
plt.show()

plt.plot(integ)
plt.title('Integrated modified signal')
plt.show()

# PLOT FFTS

yf = rfft(sig_modded)
duration = len(time_padded[0:1730])
sampleRate = len(sig_modded)/duration
xf = rfftfreq(len(sig_modded), 1/sampleRate)

plt.plot(xf, np.abs(yf))

yf = rfft(normalise( sig_padded ))
sampleRate = len(sig_padded)/len(time_padded[0:1730])
xf = rfftfreq(len(sig_padded), 1/sampleRate)
plt.plot(xf, np.abs(yf))
plt.yscale('log')
plt.legend([ 'Modified', 'Original' ])
plt.show()

