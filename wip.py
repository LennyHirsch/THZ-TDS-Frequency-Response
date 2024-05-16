import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
from tomasino_fns_wip import *

wvl_probe = 1030e-9 # probe wavelength: 1030 nm
freq_thz = 3e12 # THz frequency: 3 THz

# SETTING SIMULATION PARAMETERS
res = int(1e4)
z = np.linspace(0,1e-3,res, dtype=np.float128)
freq = np.linspace(9e11,1e13,res, dtype=np.float128)

t_pump = 245e-15
t_pp = t_pump / 2*np.sqrt(np.log(2))
print(f"t_pp: {t_pp}")
t_probe = 245e-15
print(f"t_probe: {t_probe}")

# CALCULATING N_THZ: THIS IS DONE FROM A FITTING EQUATION OF EXPERIMENTAL DATA
x = np.linspace(wvl(min(freq)), wvl(max(freq)), res)
# n_thz = power(x) # this gives us values of n_thz for all points simulated. R-square of the fit to the experimental data is 0.9987
n_thz = np.sqrt([eps_thz(omega(f)) for f in freq])

# CALCULATE ALL THE THINGS
(Ef, f, fp, foc, ovr, E_det, dk) = transfer_function(freq, wvl_probe, t_probe, t_pp, n_thz, z, 3.5e-3)

# CALCULATE TIME DOMAIN
nfft = 8 # smooths out FFT
sampleSpacing = freq[2] - freq[1]

# CALCULATE POWER SPECTRUM
E_pow = [e**2 for e in np.abs(E_det)]

td = fft.fft(np.abs(Ef), res*nfft)
xt = fft.fftfreq(res*nfft, sampleSpacing)

# NORMALISING TO 1
td = normalise(td)
E_det = normalise(E_det)
E_pow = normalise(E_pow)

# E_det = [float(e) for e in np.real(E_det)]
# Ef = [float(e) for e in np.real(Ef)]
# td = [float(t) for t in np.real(td)]

# PLOTTING E-FIELD
# plt.subplot(1,3,1)
# plt.plot(freq, np.abs(np.real(Ef)))
# plt.yscale('log')
# plt.title("E-field (Frequency domain)")
# plt.xlabel("Freq")
# plt.ylabel("E field ^2")
# plt.grid()

# PLOTTING DETECTED E-FIELD
# plt.subplot(1,2,1)
plt.plot(freq, np.abs(E_pow))
plt.yscale('log')
plt.title("Detected E-field (Frequency domain)")
plt.xlabel("Freq")
plt.ylabel("E field ^2")
starty = 0.8e-4
endy = 10
# plt.ylim([starty,endy])
startx = 0
endx = 1e13
plt.xlim([startx,endx])
# plt.grid()

# PLOTTING TIME-DOMAIN
# plt.subplot(1,2,2)
# plt.plot(xt, np.real(td))
#
# plt.title("THz pulse (time domain)")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# start = 0.8e-11
# end = 1.6e-11
# # plt.xlim([start,end])
# # plt.axis('off')
# # plt.gca().set_position([0,0,1,1])
# # plt.grid()
#
dir = "./figures/gallot-deltak-freqrespxEfield/"
filename = "pump-" + str(int(t_pump*10**15)) + "fs_probe-" + str(int( t_probe*10**15 )) + "fs"
fullname = f"{dir}{filename}.svg"
# print(fullname)
# plt.savefig(fullname, format="svg")
# plt.savefig(fullname)
plt.show()

