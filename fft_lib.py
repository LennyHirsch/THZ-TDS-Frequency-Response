import numpy as np
from scipy.special import erf
from scipy.fft import rfft, rfftfreq
from scipy.integrate import cumulative_trapezoid as cum_trap
from matplotlib import pyplot as plt
import csv

def import_thz(path):
    sig = []
    pos = []
    std = []

    with open (path, newline='') as file:
        reader = csv.reader(file, delimiter=',', quotechar='|')
        for row in reader:
            if len(row):
                sig.append(float(row[0]))
                pos.append(float(row[1]))
                std.append(float(row[2]))

    pos.reverse()

    return(np.array(sig), np.array(pos), np.array(std))

def pos2time(pos): # convert position to time
    c = 299792458
    return(pos/c)

def normalise(array):
    maximum = max(np.abs(array))
    return array/maximum

def normalise_time(time, sig): # set time at THz peak to zero
    max_index = np.argmax(np.abs(sig)) # find index of maximum signal 
    t0 = time[max_index]
    return(time-t0)

def pad_sig(sig, pad, start, end):
    zeros = [0.0] * pad
    sig = sig[start:end]

    sig = np.concatenate((zeros, sig, zeros))

    return sig

def pad_time(time, pad, start, end, step_size):
    c = 299792458
    dt = step_size / c
    time = time[start:end]

    max_time = max(time)
    min_time = min(time)

    for p in range(pad): # inefficient, but who cares?
        time = np.insert(time, 0, min_time - dt*(p+1))
        time = np.append(time, max_time + dt*(p))

    return time

def fft_thz(time, sig):
    duration = max(time) - min(time)
    n = len(sig)
    sampleRate = n/duration

    fft = rfft(sig)
    freq = rfftfreq(n, 1/sampleRate)
    return(fft, freq)

gaussian = lambda t, t0, w: np.exp(-(t-t0)**2 / w**2) # standard gaussian equation (pdf)
width = lambda tau: tau*np.sqrt(np.pi) # custom width, due to our needs... I guess
cdf = lambda t: 0.5*(1 + erf(t/np.sqrt(2))) # cumulative distribution function
skewed_gaussian = lambda t, t0, w, a: 2/w * gaussian(t,t0,w) * cdf(a*(t-t0 / w)) # TODO: doesn't work with realistic numbers

def plotter(x, *vals, title='', xlabel='', ylabel='', legend=[], marker='', xline=True, vline=False, log=False, xlim=[], ylim=[]):
    for y in vals:
        if marker:
            plt.plot(x, y, marker=marker)
        else:
            plt.plot(x, y)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(color='k', linestyle=':')

    if len(xlim):
        plt.xlim(xlim)

    if len(ylim):
        plt.ylim(ylim)

    if len(legend):
        plt.legend(legend)
    
    if xline:
        plt.axhline(color='k', linestyle=':')

    if vline:
        plt.axvline(color='k', linestyle=':')

    if log:
        plt.yscale('log')

    plt.show()
