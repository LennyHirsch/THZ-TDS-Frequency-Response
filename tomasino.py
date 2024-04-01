import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import pandas as pd

c = 299792458 # speed of light in vacuum: m/s
L_gen = 1e-3 # generation crystal thickness: 1mm
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

# functions for calculating Fabry-Perot transfer function

def eps_thz(omega):
    return( eps_inf + ((11.1 - eps_inf) * phonon_res**2)/(phonon_res**2 - omega**2 + 1j*phonon_mode*omega) )

def T12(omega):
    return( (2*np.sqrt(eps_thz(omega)))/(np.sqrt(eps_thz(omega)) + 1) )

def R12(omega):
    return( ((np.sqrt(eps_thz(omega)) - 1)/(np.sqrt(eps_thz(omega)) + 1)) )

def T_fp(omega, n_thz):
    top = T12(omega)**2 * np.exp(-1j * omega * n_thz * L_gen/c)
    bottom = 1 - ( R12(omega)**2 * np.exp(-2j * omega * n_thz * L_gen/c) )
    return(top/bottom)

# functions for calculating focusing transfer function
def z_B(omega):
    return( (omega**2 * ( 3.5e-3 )**3) / (4 * np.sqrt(2) * c**2) )

def z_diff(freq):
    if k(wvl(freq))*3.5e-3 <= 1:
        return z_B(omega(freq))
    else:
        return k(wvl(freq)) * (3.5e-3)**2 / 2

def r_in(freq):
    term1 = 50.4e-3 / 2*np.sqrt(2)
    term2 = 3.5e-3 * np.sqrt(1 + freq/z_diff(omega(freq))**2)

    if term1 < term2:
        return term1
    else:
        return term2

def r_foc(freq):
    return( 2*c*freq / omega(freq)*r_in(omega(freq)) )

def T_foc(freq):
    return r_foc(freq)/r_in(freq)

#N_THZ CALCULATION
power = lambda x: 1346*(x**-2.373) + 3.34 # coefficients from Matlab power fit of experimental data (Parsons)
parsons = pd.read_csv('C:/Users/Lenny/Documents/Python/Tomasino_THz/Parsons.csv')

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

res = int(1e4)
z = np.linspace(0,100e-9,res) # NOTE: This variable is a bit of a mystery... Not entirely sure why this range works.
freq = np.linspace(0.01e12,1e13,res)

eps_inf = 5
phonon_res = 11e12
phonon_mode = 1e12

t_pump = 245e-15
t_probe_delta = 1e-21
t_probe_short = 70e-15
t_probe_long = 245e-15

# CALCULATING N_THZ: THIS IS DONE FROM A FITTING EQUATION OF EXPERIMENTAL DATA
x = np.linspace(min(parsons['wl']), max(parsons['wl']), res)
n_thz = power(x) # this gives us values of n_thz for all points simulated. R-square of the fit to the experimental data is 0.9987

for i, f in enumerate(freq):
    delta_k = k(wvl_probe) + k(wvl(f)) - k(wvl_probe - wvl(f))
    FR_delta = freq_response(Aopt(omega(f), t_probe_delta), X2, deltaPhi(L_det, delta_k))
    FR_long = freq_response(Aopt(omega(f), t_probe_long), X2, deltaPhi(L_det, delta_k))
    FR_short = freq_response(Aopt(omega(f), t_probe_short), X2, deltaPhi(L_det, delta_k))
    EF = E(n_thz[i], omega(f), z[i])
    tfoc = T_foc(f)

    fabry_perot.append(T_fp(omega(f), n_thz[i]))
    focal.append(tfoc)
    

    Freq_response_delta.append(FR_delta)
    Freq_response_long.append(FR_long)
    Freq_response_short.append(FR_short)

    E_field.append(EF**2)
    # E_det_delta.append(FR_delta*(EF**2))
    # E_det_long.append(FR_long*(EF**2))
    # E_det_short.append(FR_short*(EF**2))

# max_delta = max(Freq_response_delta)
# max_long = max(Freq_response_long)
# max_short = max(Freq_response_short)

# Freq_response_delta = Freq_response_delta / max_delta
# Freq_response_long = Freq_response_long / max_long
# Freq_response_short = Freq_response_short / max_short

# E_det_delta = [f*e*fp for f,e in zip(Freq_response_delta, E_field)]
# E_det_long = [f*e*fp for f,e in zip(Freq_response_long, E_field)]
# E_det_short = [f*e*fp for f,e in zip(Freq_response_short, E_field)]
E_det_delta = [f*e*fp*foc for f,e,fp,foc in zip(Freq_response_delta, E_field, fabry_perot, focal)]
E_det_long = [f*e*fp*foc for f,e,fp,foc in zip(Freq_response_long, E_field, fabry_perot, focal)]
E_det_short = [f*e*fp*foc for f,e,fp,foc in zip(Freq_response_short, E_field, fabry_perot, focal)]

# CALCULATE TIME DOMAIN
nfft = 8 # smooths out FFT
sampleSpacing = freq[2] - freq[1]

td_delta = fft.fft(np.real(E_det_delta), res*nfft)
td_long = fft.fft(np.real(E_det_long), res*nfft)
td_short = fft.fft(np.real(E_det_short), res*nfft)
xt = fft.fftfreq(res*nfft, sampleSpacing)

max_delta = max(td_delta)
max_long = max(td_long)
max_short = max(td_short)

td_delta = td_delta / max_delta
td_long = td_long / max_long
td_short = td_short / max_short

max_delta = max(E_det_delta)
max_long = max(E_det_long)
max_short = max(E_det_short)

E_det_delta = E_det_delta / max_delta
E_det_long = E_det_long / max_long
E_det_short = E_det_short / max_short

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

# plt.plot(freq, np.abs(np.real(E_det_delta)))
# plt.plot(freq, np.abs(np.real(E_det_long)))
# plt.plot(freq, np.abs(np.real(E_det_short)))
# plt.yscale('log')
# plt.ylim([1e-3,5e0])
# plt.legend(['Delta','Long','Short'])
# plt.title("Detected E-field (Frequency domain)")
# plt.xlabel("Freq")
# plt.ylabel("E field ^2")
# plt.show()
