import numpy as np
from scipy.integrate import cumulative_trapezoid as cum_trap
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from fft_lib import *

# IMPORT RAW DATA
path = './experimental-data/'
filenames = \
['datastruct_uncompressed - 23-11-14.csv',
'datastruct_uncompressed - prelim 1 - 23-11-14.csv',
'datastruct_uncompressed - 23-06-23-unpurged.csv',
'datastruct_uncompressed - 1299 chop - 23-06-24.csv',
'datastruct_uncompressed - 22-07-11.csv',
'datastruct_uncompressed - 22-07-11.csv',
'datastruct_compressed - 23-05-02.csv']

fullpath = path + filenames[3]

(sig, pos, std) = import_thz(fullpath)
time = pos2time(pos)

normalised_sig = normalise(sig)
zeroed_time = normalise_time(time, sig)

pad = 500
step_size = pos[1]-pos[0]
start = 20
end = 740
print(len(sig))

padded_sig = np.array(pad_sig(normalised_sig, pad, start, end))
padded_time = np.array(pad_time(zeroed_time, pad, start, end, step_size))

# TEST PLOTS
# plotter(orig_pos, orig_sig, title='Raw data', xlabel='Position (mm)')
# plotter(orig_time, orig_sig, title='Raw time data', xlabel='Time (s)')
# plotter(zeroed_time, orig_sig, title='Zeroed time', xlabel='Time (s)')
# plotter(padded_time, padded_sig, title='Padded data', xlabel='Time (s)')

# APPLY GAUSSIAN
# # UNCOMPRESSED 23-06-23 unpurged start = 0 end = 316
# t0 = 0e-12
# tau = 0.5e-12
# factor = 5
# COMPRESSED
# t0 = 0.378e-12
# tau = 0.3105e-12
# factor = 10
# UNCOMPRESSED 1299 CHOP start = 20 end = 740
t0 = 0.36e-12
tau = 0.355e-12
factor = 7.7

gauss = normalise(gaussian(padded_time, t0, width(tau)))
gauss = gauss*factor

sig_gauss = padded_sig*(1 + gauss)

cum_sum = normalise(cum_trap(sig_gauss, padded_time))

# def optimal_cum(params): # this should be as small as possible
#     t0, tau, factor = params
#     gauss = gaussian(padded_time, t0, width(tau))*factor
#     sig_gauss = padded_sig*(1 + gauss)
#     cum_sum = cum_trap(sig_gauss, padded_time)
#     return cum_sum[-1]
#
# initial_guess = [t0, tau, factor]
# result = minimize(optimal_cum, initial_guess, tol=1e-20, options={'maxiter': 1000})
#
# if result.success:
#     print(result.x)
#     print(result.fun)
# else:
#     raise ValueError(result.message)
#
# optimal = optimal_cum(initial_guess)
print(f"Cumulative sum: {cum_sum[-1]}")

# PLOT COMPARISON OF SIGNALS (TIME-DOMAIN)
plotter(padded_time[0:-1], padded_sig[0:-1], sig_gauss[0:-1], cum_sum, title='Comparing padded sig, sig_gauss, and cum sum', legend=['Padded sig', 'Sig gauss', 'Cum sum'])

# FFT
(orig_fft, freq) = fft_thz(padded_time, padded_sig)
(fft, freq) = fft_thz(padded_time, sig_gauss)
orig_fft = normalise(np.abs(orig_fft))
fft = normalise(np.abs(fft))

xlim = [-1, 10] # in THz
ylim = [1e-5, 10]

plotter(freq*10**-12, orig_fft, fft, title='Comparing FFTs', xlabel="Frequency (THz)", legend=['Original', 'Modified'], xlim=xlim, ylim=ylim, log=True)
