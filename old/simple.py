import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
from tomasino_fns import *

wvl_probe = 1030e-9 # probe wavelength: 1030 nm
freq_thz = 3e12 # THz frequency: 3 THz

# SETTING SIMULATION PARAMETERS
res = int(1e3)
z = np.linspace(0,100e-9,res) # NOTE: This variable is a bit of a mystery... Not entirely sure why this range works.
freq = np.linspace(0.01e11,1e13,res)

t_pump = 160e-15
t_probe = 245e-15

# CALCULATING N_THZ: THIS IS DONE FROM A FITTING EQUATION OF EXPERIMENTAL DATA
x = np.linspace(wvl(min(freq)), wvl(max(freq)), res)
n_thz = power(x) # this gives us values of n_thz for all points simulated. R-square of the fit to the experimental data is 0.9987

# CALCULATE ALL THE THINGS
(Ef, f, fp, foc, ovr, E_det) = transfer_function(freq, wvl_probe, t_probe, t_pump, n_thz, z, 3.5e-3)

# CALCULATE TIME DOMAIN
nfft = 8 # smooths out FFT
sampleSpacing = freq[2] - freq[1]

# CALCULATE POWER SPECTRUM
E_pow = [e**2 for e in np.abs(E_det)]

td = fft.fft(np.real(E_det), res*nfft)
xt = fft.fftfreq(res*nfft, sampleSpacing)

# NORMALISING TO 1
td = normalise(td)
E_pow = normalise(E_pow)

# PLOTTING E-FIELD
plt.subplot(1,3,1)
plt.plot(freq, np.real(Ef))
plt.title("E-field (Frequency domain)")
plt.xlabel("Freq")
plt.ylabel("E field ^2")
plt.grid()

# PLOTTING DETECTED E-FIELD
plt.subplot(1,3,2)
plt.plot(freq, E_pow)
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
plt.plot(xt, np.real(td))
plt.title("THz pulse (time domain)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
start = 0.9e-11
end = 1.8e-11
plt.xlim([start,end])
plt.grid()

plt.show()
