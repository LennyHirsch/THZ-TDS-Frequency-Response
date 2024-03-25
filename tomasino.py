import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import pandas as pd

c = 299792458 # speed of light in vacuum: m/s
L_det = 300e-6 # detection crystal thickness: 300 um
wvl_probe = 1030e-9 # probe wavelength: 1030 nm
freq_thz = 3e12 # THz frequency: 3 THz
X2 = 200e-12
# X2 = 0.97e-12 # chi(2) coefficient: pm/V. NOTE: This should be the electro-optic coefficient (I think?)
# In Tomasino et al. they say X2 is the second-order susceptibility for the DFG case (the same as the OR case). Is this the r_41? Or d_41? I... don't know.

# E-field constants
E0 = 1e3 # normalise electric field for now
n_g = 3.4216 # NOTE: from refractiveindex.info (Parsons and Coleman)

# HELPER FUNCTIONS
k = lambda wvl: 2*np.pi / wvl
omega = lambda freq: 2*np.pi*freq
wvl = lambda freq: c / freq

# FREQUENCY RESPONSE FUNCTIONS
def Aopt(omega, pulseDuration):
    return((np.sqrt(np.pi)/pulseDuration)*np.exp(-(omega**2 * pulseDuration**2)/4))

def deltaPhi(Ldet, delta_k):
    return((np.exp(-1j * delta_k * Ldet) - 1)/(1j * delta_k))

def freq_response(Aopt, X2, deltaPhi):
    return(Aopt*X2*deltaPhi)

# E-FIELD FUNCTIONS
def E1(n_thz, omega):
    return( (E0**2 * X2 * t_pump * np.sqrt(np.pi))/(2*(n_thz**2 - n_g**2)) * np.exp(- (t_pump**2 * omega**2) / 4) )

def E2(n_thz, omega, z):
    return( 0.5*(1 - n_g/n_thz)*np.exp(1j * n_thz * omega * z / c) )

def E3(n_thz, omega, z):
    return( 0.5*(1 + n_g/n_thz)*np.exp(-1j * n_thz * omega * z / c) )

def E4(omega, z):
    return( np.exp(-1j * n_g * omega * z / c) )

def E(n_thz, omega, z):
    return( E1(n_thz, omega)*(E2(n_thz, omega, z) + E3(n_thz, omega, z) - E4(omega, z)) )

#N_THZ CALCULATION
power = lambda x: 1346*(x**-2.373) + 3.34
parsons = pd.read_csv('C:/Users/2090496H/OneDrive - University of Glasgow/Documents/Python/THz-TDS-Frequency-Response/Parsons.csv')

# INITIALISE ARRAYS AND CONSTANTS
Freq_response = []
E_field = []
E_det = []

res = int(1e5)
z = np.linspace(0,100e-9,res) # NOTE: This variable is a bit of a mystery...
freq = np.linspace(0.1e12,5e12,res)

t_pump = 245e-15
t_probe = 245e-15

# CALCULATING N_THZ: THIS IS DONE FROM A FITTING EQUATION OF EXPERIMENTAL DATA
x = np.linspace(min(parsons['wl']), max(parsons['wl']), res)
n_thz = power(x) # this gives us values of n_thz for all points simulated. R-square of the fit to the experimental data is 0.9987

for i, f in enumerate(freq):
    delta_k = k(wvl_probe) + k(wvl(f)) - k(wvl_probe - wvl(f))
    FR = freq_response(Aopt(omega(f), t_probe), X2, deltaPhi(L_det, delta_k))
    EF = E(n_thz[i], omega(f), z[i])

    Freq_response.append(FR)
    E_field.append(EF**2)
    E_det.append(FR*(EF**2))

# CALCULATE TIME DOMAIN
nfft = 8 # smooths out FFT
sampleSpacing = freq[2] - freq[1]

td = fft.fft(np.real(E_det), res*nfft)
xt = fft.fftfreq(res*nfft, sampleSpacing)

# PLOTTING E-FIELD
plt.subplot(1,3,1)
plt.plot(freq, np.real(E_field))
plt.title("E-field (Frequency domain)")
plt.xlabel("Freq")
plt.ylabel("E field ^2")

# PLOTTING DETECTED E-FIELD
plt.subplot(1,3,2)
plt.plot(freq, np.abs(np.real(E_det)))
plt.title("Detected E-field (Frequency domain)")
plt.xlabel("Freq")
plt.ylabel("E field ^2")

# PLOTTING TIME-DOMAIN
xlim = 0.5e-11
plt.subplot(1,3,3)
plt.plot(xt, np.real(td))
plt.xlim([0,xlim])
plt.title("THz pulse (time domain)")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.show()