## Simulation of THz generation and detection, as done by Tomasino et al.
## Equation 10 is neglected (Bethe's diffraction), as our THz source has a relatively large source area compared
## to that in the paper (3.5 mm vs 30 um).

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
from tomasino_fns import *

wvl_probe = 1030e-9 # probe wavelength: 1030 nm
freq_thz = 3e12 # THz frequency: 3 THz

# INITIALISE ARRAYS AND CONSTANTS
Freq_response_delta = []
Freq_response_long = []
Freq_response_short = []
E_field = []
E_det_delta = []
E_det_long = []
E_det_short = []
fabry_perot = []
focal = []

# SETTING SIMULATION PARAMETERS
res = int(1e4)
z = np.linspace(0,100e-9,res) # NOTE: This variable is a bit of a mystery... Not entirely sure why this range works.
freq = np.linspace(0.01e12,1e13,res)

t_pump = 245e-15
t_probe = 55e-15
t_probe_delta = 1e-21
t_probe_short = 70e-15
t_probe_long = 245e-15

# CALCULATING N_THZ: THIS IS DONE FROM A FITTING EQUATION OF EXPERIMENTAL DATA
x = np.linspace(min(parsons['wl']), max(parsons['wl']), res)
n_thz = power(x) # this gives us values of n_thz for all points simulated. R-square of the fit to the experimental data is 0.9987

# CHECK COHERENCE LENGTH; CRYSTAL MIGHT BE TOO LONG...
# print(L_coh(freq_thz, n_thz[1000]))

for i, f in enumerate(freq):
    delta_k = k(wvl_probe) + k(wvl(f)) - k(wvl_probe - wvl(f))
    FR_delta = freq_response(Aopt(omega(f), t_probe_delta), X2, deltaPhi(L_det, delta_k))
    FR_long = freq_response(Aopt(omega(f), t_probe_long), X2, deltaPhi(L_det, delta_k))
    FR_short = freq_response(Aopt(omega(f), t_probe_short), X2, deltaPhi(L_det, delta_k))
    EF = E(n_thz[i], omega(f), z[i], t_pump)
    tfoc = T_foc(f)

    fabry_perot.append(T_fp(omega(f), n_thz[i]))
    focal.append(tfoc)
    

    Freq_response_delta.append(FR_delta)
    Freq_response_long.append(FR_long)
    Freq_response_short.append(FR_short)

    E_field.append(EF**2)

# E_det_delta = [f*e*fp for f,e in zip(Freq_response_delta, E_field)]
# E_det_long = [f*e*fp for f,e in zip(Freq_response_long, E_field)]
# E_det_short = [f*e*fp for f,e in zip(Freq_response_short, E_field)]
E_det_delta = [f*e*fp*foc for f,e,fp,foc in zip(Freq_response_delta, E_field, fabry_perot, focal)]
E_det_long = [f*e*fp*foc for f,e,fp,foc in zip(Freq_response_long, E_field, fabry_perot, focal)]
E_det_short = [f*e*fp*foc for f,e,fp,foc in zip(Freq_response_short, E_field, fabry_perot, focal)]

# CALCULATE TIME DOMAIN
nfft = 4 # smooths out FFT
sampleSpacing = freq[2] - freq[1]

td_delta = fft.fft(np.real(E_det_delta), res*nfft)
td_long = fft.fft(np.real(E_det_long), res*nfft)
td_short = fft.fft(np.real(E_det_short), res*nfft)
xt = fft.fftfreq(res*nfft, sampleSpacing)

# NORMALISING TO 1
td_delta = normalise(td_delta)
td_long = normalise(td_long)
td_short = normalise(td_short)
E_det_delta = normalise(E_det_delta)
E_det_long = normalise(E_det_long)
E_det_short = normalise(E_det_short)


# PLOTTING E-FIELD
plt.subplot(1,3,1)
plt.plot(freq, np.real(E_field))
plt.title("E-field (Frequency domain)")
plt.xlabel("Freq")
plt.ylabel("E field ^2")

# PLOTTING DETECTED E-FIELD
plt.subplot(1,3,2)
plt.plot(freq, np.abs(np.real(E_det_delta)))
plt.plot(freq, np.abs(np.real(E_det_long)))
plt.plot(freq, np.abs(np.real(E_det_short)))
plt.yscale('log')
plt.legend(['Delta','Long','Short'])
plt.title("Detected E-field (Frequency domain)")
plt.xlabel("Freq")
plt.ylabel("E field ^2")

# PLOTTING TIME-DOMAIN
xlim = 0.5e-11
plt.subplot(1,3,3)
plt.plot(xt, np.real(td_delta))
plt.plot(xt, np.real(td_long))
plt.plot(xt, np.real(td_short))
plt.legend(['Long','Short'])
# plt.xlim([0,xlim])
plt.title("THz pulse (time domain)")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.show()
