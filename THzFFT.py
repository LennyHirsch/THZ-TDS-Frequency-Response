from scipy.io import loadmat
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft, rfft, rfftfreq, fftfreq
from scipy.integrate import cumulative_trapezoid
import numpy as np
import pandas as pd
# import xarray as xr
import csv

def fft_thz(sig, pos, pad, step_size, start, end):
    c = 299792458
    dt = step_size / c
    zeros = [0.0] * pad

    # diff = len(sig) - end - start # difference between full signal and single-cycle pulse
    diff = len(sig) - (end - start) # difference between full signal and single-cycle pulse
    zero_diff = [0.0] * diff

    sig = zeros + sig[start:end] + zeros + zero_diff # slice only the single cycle, then time_pad

    time = [i*dt for i in range(len(sig))]

    # pos.reverse()
    #
    # time = [p / c for p in pos]
    # # time = [max(time) - t for t in time]
    #
    # time_pad = []
    # for t in range(pad*2):
    #     time_pad.append(max(time) + t*dt)
    #
    # time = time + time_pad

    duration = max(time)
    n = len(sig)
    sampleRate = n/duration

    fft = rfft(sig)
    freq = rfftfreq(n, 1/sampleRate)

    return (freq, fft, sig, time)

def normalise(array):
    maximum = max(np.abs(array))
    return [a/maximum for a in array]


path = './experimental-data/'
# filename = 'datastruct_uncompressed - 23-11-14.csv'
# filename = 'datastruct_uncompressed - prelim 1 - 23-11-14.csv'
# filename = 'datastruct_uncompressed - 23-06-23-unpurged.csv'
# filename = 'datastruct_uncompressed - 1299 chop - 23-06-24.csv'
# filename = 'datastruct_uncompressed - 22-07-11.csv'
filename = 'datastruct_compressed - 23-05-02.csv'

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

# pos.reverse()

pad = 500
step_size = 2e-6

(freq, fft, sig, time) = fft_thz(sig, pos, pad, step_size, 20, 740)

# # FROM MEETING WITH MATTEO

plt.plot(time, sig)
plt.show()
# sig = [s + 2.2e-4 for s in sig]

# 1299 chop
t_0 = 5.685e-12 # 1299 chop
width = 0.9e-12 
factor = 0.919785

# compressed
t_0 = 3.872e-12 # 1299 chop
width = 8.81e-13 
factor = 1.1
gauss = [np.exp(-((t-t_0)**2 ) * np.pi / width**2) for t in time]

gauss = [g*factor for g in gauss]

sig_gauss = [s*(1 + g) for s, g in zip(sig, gauss)]

integ = cumulative_trapezoid(sig_gauss, time)
print(f"last cum sum value: {integ[-1]}")
print(f"gauss max: {time[gauss.index(max(gauss))]}")

plt.plot(time[0:-1], integ)
# plt.plot(time, gauss)
plt.show()

plt.plot(time, sig_gauss)
plt.show()

#WIP

fft_edit = rfft(sig_gauss)
duration = len(time)
sampleRate = len(sig_gauss)/duration
freq_edit = rfftfreq(len(sig_gauss), 1/sampleRate)

# NORMALISE
fft = normalise(fft)
fft_edit = normalise(fft_edit)
sig = normalise(sig)
gauss = normalise(gauss)

# plt.plot(time, sig)
# plt.plot(time, gauss)
# plt.show()
#
plt.plot(freq, np.abs(fft))
plt.plot(freq, np.abs(fft_edit))
plt.yscale('log')
plt.xlim([-1e12,1e13])
plt.show()

#
# # PLOTTING
# plt.figure(1)
# plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
# plt.grid(which='minor', color='#EEEEEE', linewidth=0.4)
# plt.minorticks_on()
# plt.plot(range(len(pos)), sig, 'gx')
# plt.xlabel('Position (mm)')
# plt.ylabel('Amplitude (V)')
# plt.title(filename + '  Time domain')
#
# plt.figure(2)
# # plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
# # plt.grid(which='minor', color='#EEEEEE', linewidth=0.4)
# plt.minorticks_on()
# plt.plot(freq, np.abs([f**2 for f in fft_norm]))
# plt.ylim([1e-4, 2])
# plt.xlim([0,1e13])
# plt.xlabel('Frequency (Hz)')
# plt.yscale('log')
#
# plt.ylabel('Amplitude^2 (a.u.)')
# plt.title(filename + ' FFT')
#
# # plt.show()
