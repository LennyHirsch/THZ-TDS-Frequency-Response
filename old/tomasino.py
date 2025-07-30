## Simulation of THz generation and detection, as done by Tomasino et al.
## Equation 10 is neglected (Bethe's diffraction), as our THz source has a relatively large source area compared
## to that in the paper (3.5 mm vs 30 um).

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
# from intersect import intersection
from tomasino_fns import *

wvl_probe = 1030e-9 # probe wavelength: 1030 nm
freq_thz = 3e12 # THz frequency: 3 THz

# SETTING SIMULATION PARAMETERS
res = int(1e3)
z = np.linspace(0,100e-9,res) # NOTE: This variable is a bit of a mystery... Not entirely sure why this range works.
# z = np.linspace(0,100e-9,res) # NOTE: This variable is a bit of a mystery... Not entirely sure why this range works.
freq = np.linspace(0.01e11,1e13,res)

t_pump = 245e-15
# t_pump = 160e-15
t_probe_delta = 1e-21
t_probe_short = 55e-15
t_probe_long = 245e-15
t_probe_test = 170e-15

# CALCULATING N_THZ: THIS IS DONE FROM A FITTING EQUATION OF EXPERIMENTAL DATA
# x = np.linspace(min(parsons['wl']), max(parsons['wl']), res)
x = np.linspace(wvl(min(freq)), wvl(max(freq)), res)
n_thz = power(x) # this gives us values of n_thz for all points simulated. R-square of the fit to the experimental data is 0.9987

# CHECK COHERENCE LENGTH; CRYSTAL MIGHT BE TOO LONG...
# print(L_coh(freq_thz, n_thz[1000]))

# CALCULATE ALL THE THINGS
(Ef_delta, f_delta, fp_delta, foc_delta, ovr_delta, E_delta) = transfer_function(freq, wvl_probe, t_probe_delta, t_pump, n_thz, z, 3.5e-3)
(Ef_short, f_short, fp_short, foc_short, ovr_short, E_short) = transfer_function(freq, wvl_probe, t_probe_short, t_pump, n_thz, z, 3.5e-3)
(Ef_long, f_long, fp_long, foc_long, ovr_long, E_long) = transfer_function(freq, wvl_probe, t_probe_long, t_pump, n_thz, z, 3.5e-3)
(Ef_test, f_test, fp_test, foc_test, ovr_test, E_test) = transfer_function(freq, wvl_probe, t_probe_test, t_pump, n_thz, z, 3.5e-3)

# CALCULATE TIME DOMAIN
nfft = 8 # smooths out FFT
sampleSpacing = freq[2] - freq[1]

# CALCULATE POWER SPECTRUM
E_delta_pow = [e**2 for e in np.abs(E_delta)]
E_short_pow = [e**2 for e in np.abs(E_short)]
E_long_pow = [e**2 for e in np.abs(E_long)]
E_test_pow = [e**2 for e in np.abs(E_test)]

# E_delta_pow = [e**2 for e in np.real(np.abs(E_delta))]
# E_short_pow = [e**2 for e in np.real(np.abs(E_short))]
# E_long_pow = [e**2 for e in np.real(np.abs(E_long))]
# E_test_pow = [e**2 for e in np.real(np.abs(E_test))]

td_delta = fft.fft(np.real(E_delta), res*nfft)
td_short = fft.fft(np.real(E_short), res*nfft)
td_long = fft.fft(np.real(E_long), res*nfft)
td_test = fft.fft(np.real(E_test), res*nfft)
xt = fft.fftfreq(res*nfft, sampleSpacing)

# NORMALISING TO 1
td_delta = normalise(td_delta)
td_short = normalise(td_short)
td_long = normalise(td_long)
td_test = normalise(td_test)

E_delta_pow = normalise(E_delta_pow)
E_short_pow = normalise(E_short_pow)
E_long_pow = normalise(E_long_pow)
E_test_pow = normalise(E_test_pow)

# PLOTTING E-FIELD
plt.subplot(1,3,1)
plt.plot(freq, np.real(Ef_delta))
plt.title("E-field (Frequency domain)")
# plt.yscale('log')
plt.xlabel("Freq")
plt.ylabel("E field ^2")
plt.grid()

# PLOTTING DETECTED E-FIELD
plt.subplot(1,3,2)
# plt.plot(freq, E_delta_pow)
# plt.plot(freq, E_short_pow)
plt.plot(freq, E_long_pow)
# plt.plot(freq, E_test_pow)
plt.legend(["Delta", "Short", "Long", "Test"])
plt.yscale('log')
plt.title("Detected E-field (Frequency domain)")
plt.xlabel("Freq")
plt.ylabel("E field ^2")
start = 3e-4
end = 1.4
# plt.ylim([start,end])
plt.grid()

# PLOTTING TIME-DOMAIN
plt.subplot(1,3,3)
# plt.plot(xt, np.real(td_delta))
plt.plot(xt, np.real(td_short))
plt.plot(xt, np.real(td_long))
# plt.plot(xt, np.real(td_test))
plt.legend(["Delta", "Short", "Long", "Test"])
plt.title("THz pulse (time domain)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
start = 0.9e-11
end = 1.8e-11
plt.xlim([start,end])
plt.grid()

plt.show()
#
# linex = np.linspace(1.08e-11,1.15e-11,res)
# liney = np.linspace(0,0,res)
#
# xs, ys = intersection(linex, liney, xt, np.real(td_short))
# xl, yl = intersection(linex, liney, xt, np.real(td_long))
# plt.plot(np.linspace(0,1.8e-11,res),np.linspace(0,0,res))
# plt.plot(xt, np.real(td_short))
# plt.plot(xt, np.real(td_long))
# plt.plot(xs,ys,"*k")
# plt.plot(xl,yl,"xk")
# print("Duration short: " + str(xs[2]-xs[0]))
# print("Duration long: " + str(xl[2]-xl[0]))
# plt.show()
#
