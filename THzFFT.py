from scipy.io import loadmat
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft, rfft, rfftfreq, fftfreq
from scipy.integrate import cumulative_trapezoid
import numpy as np
import pandas as pd
import xarray as xr
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

def import_matlab_thz_struct(filepath):
    trace = loadmat(filepath)['dataStruct']
    sig = [item.flat[0] for item in trace[0][0]['x_mean'][0]] #in V
    pos = [item.flat[0] / 1000 for item in trace[0][0]['pos'][0]] #in m
    std = [item.flat[0] for item in trace[0][0]['x_std'][0]] #in V

    return (sig, pos, std)

path = './'
# filename = 'datastruct_uncompressed - 23-11-14.mat'
# filename = 'datastruct_uncompressed - prelim 1 - 23-11-14.mat'
# filename = 'datastruct_uncompressed - 23-06-23-unpurged.mat'
filename = 'datastruct_uncompressed - 1299 chop - 23-06-24.mat'
# filename = 'datastruct_uncompressed - 22-07-11.mat'
# filename = 'datastruct_compressed - 23-05-02.mat'

(sig, pos, std) = import_matlab_thz_struct(path + filename)

rows = zip(sig, pos, std)
with open(filename.removesuffix('.mat') + '.csv', "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)


sig = [s + 2.2e-4 for s in sig]

pad = 500
step_size = 2e-6

(freq, fft, sig_padded, time_padded) = fft_thz(sig, pos, pad, step_size, 20, 740)


t_0 = 6e-12
integration_fact = 0.9e-12
fun = [np.exp(-((t-t_0)**2 )/integration_fact**2 * np.pi) for t in time_padded]

factor = 0.885
fun = [f*factor for f in fun]

# plt.plot(fun)
# plt.show()
sig_fun = [s*(1 + f) for s, f in zip(sig_padded, fun)]

integ = cumulative_trapezoid(sig_fun, time_padded[0:1730])
print(integ[-1])

# plt.plot(time_padded[0:1729], integ)
# plt.plot(sig_fun)
# plt.plot(integ)
# plt.show()

#WIP

yf = rfft(sig_fun)
duration = len(time_padded[0:1730])
sampleRate = len(sig_fun)/duration
xf = rfftfreq(len(sig_fun), 1/sampleRate)

plt.plot(xf, np.abs(yf))
plt.yscale('log')
plt.show()

#NORMALISE
maximum = max(fft)
fft_norm = [f/maximum for f in fft]

# PLOTTING
plt.figure(1)
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
plt.grid(which='minor', color='#EEEEEE', linewidth=0.4)
plt.minorticks_on()
plt.plot(range(len(pos)), sig, 'gx')
plt.xlabel('Position (mm)')
plt.ylabel('Amplitude (V)')
plt.title(filename + '  Time domain')

plt.figure(2)
# plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
# plt.grid(which='minor', color='#EEEEEE', linewidth=0.4)
plt.minorticks_on()
plt.plot(freq, np.abs([f**2 for f in fft_norm]))
plt.ylim([1e-4, 2])
plt.xlim([0,1e13])
plt.xlabel('Frequency (Hz)')
plt.yscale('log')

plt.ylabel('Amplitude^2 (a.u.)')
plt.title(filename + ' FFT')

# plt.show()