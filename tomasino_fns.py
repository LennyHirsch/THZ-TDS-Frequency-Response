# Function definitions for tomasino.py
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import pandas as pd

c = 299792458 # speed of light in vacuum: m/s
L_gen = 1e-3 # generation crystal thickness: 1mm
L_det = 300e-6 # detection crystal thickness: 300 um
thz_waist = 3.5e-3
# X2 = 200e-12
X2 = 0.97e-12 # chi(2) coefficient: pm/V. NOTE: This should be the electro-optic coefficient (I think?)
# In Tomasino et al. they say X2 is the second-order susceptibility for the DFG case (the same as the OR case). Is this the r_41? Or d_41? I... don't know.

# E-field constants
E0 = 1e3 # normalise electric field for now
n_g = 3.4216 # NOTE: from refractiveindex.info (Parsons and Coleman)

# NOTE: these probably need fixing/changing frequently for accurate usage
eps_0 = 11.1
eps_inf = 9.11
phonon_res = 11e12
phonon_mode = 2e12

# HELPER FUNCTIONS
k = lambda wvl: 2*np.pi / wvl
omega = lambda freq: 2*np.pi*freq
wvl = lambda freq: c / freq

# E-FIELD FUNCTIONS
def E1(n_thz, omega, t_pump):
    return( ( (E0**2 * X2 * t_pump * np.sqrt(np.pi)) * np.exp(- (t_pump**2 * omega**2) / 4) ) / (2*(n_thz**2 - n_g**2)))

def E2(n_thz, omega, z):
    return( 0.5*(1 - n_g/n_thz)*np.exp(1j * n_thz * omega * z / c) )

def E3(n_thz, omega, z):
    return( 0.5*(1 + n_g/n_thz)*np.exp(-1j * n_thz * omega * z / c) )

def E4(omega, z):
    return( np.exp(-1j * n_g * omega * z / c) )

def E(n_thz, omega, z, t_pump):
    return( E1(n_thz, omega, t_pump)*(E2(n_thz, omega, z) + E3(n_thz, omega, z) - E4(omega, z)) )

# FREQUENCY RESPONSE FUNCTIONS
def Aopt(omega, pulseDuration):
    return((np.sqrt(np.pi)/pulseDuration)*np.exp(-(omega**2 * pulseDuration**2)/4))

def deltaPhi(Ldet, delta_k):
    return((np.exp(-1j * delta_k * Ldet) - 1)/(1j * delta_k))

def freq_response(Aopt, X2, deltaPhi):
    return(Aopt*X2*deltaPhi)

# FABRY-PEROT TRANSFER FUNCTIONS
def eps_thz(omega):
    return(eps_inf + ( ((eps_0 - eps_inf) * phonon_res**2)/(phonon_res**2 - omega**2 + 1j*phonon_mode*omega) ))

def T12(omega):
    return( (2*np.sqrt(eps_thz(omega)))/(np.sqrt(eps_thz(omega)) + 1) )

def R12(omega):
    return( ((np.sqrt(eps_thz(omega)) - 1)/(np.sqrt(eps_thz(omega)) + 1)) )

def T_fp(omega, n_thz):
    top = T12(omega)**2 * np.exp(-1j * omega * n_thz * L_gen/c)
    bottom = 1 - ( R12(omega)**2 * np.exp(-2j * omega * n_thz * L_gen/c) )
    return(top/bottom)

# FOCUSING TRANSFER FUNCTIONS
def z_B(omega):
    return( (omega**2 * ( thz_waist )**3) / (4 * np.sqrt(2) * c**2) )

def z_diff(freq):
    if k(wvl(freq))*thz_waist <= 1:
        return z_B(omega(freq))
    else:
        return ( k(wvl(freq)) * thz_waist**2 / 2 )

def r_in(freq):
    term1 = 50.4e-3 / 2*np.sqrt(2)
    term2 = thz_waist * np.sqrt(1 + ( freq**2 )/( z_diff(omega(freq))**2 ))

    if term1 < term2:
        return term1
    else:
        return term2

def r_foc(freq):
    return( 2*c*freq / omega(freq)*r_in(omega(freq)) )

def T_foc(freq):
    return r_foc(freq)/r_in(freq)

# OVERLAP TRANSFER FUNCTION
def T_overlap(freq, probe_waist):
    top = np.pi * probe_waist**2 * r_foc(freq)**2
    bottom = probe_waist**2 + 2*r_foc(freq)**2
    return np.sqrt( top/bottom )

# FUNCTION TO CHECK COHERENCE LENGTH
def L_coh(freq, n_thz):
    return wvl(freq) / 2*np.abs(n_g - n_thz)

# NORMALISATION
def normalise(array):
    max_val = max(array)
    return (array / max_val)

# DO EVERYTHING
def transfer_function(freq, wvl_probe, t_probe, t_pump, n_thz, z, probe_waist):
    E_field = []
    freq_resp = []
    trans_fp = []
    trans_foc = []
    trans_overlap = []
    E_det = []

    for i, f in enumerate(freq):
        delta_k = k(wvl_probe) + k(wvl(f)) - k(wvl_probe - wvl(f))
        
        freq_resp.append( freq_response(Aopt(omega(f), t_probe), X2, deltaPhi(L_det, delta_k)) )
        trans_fp.append(T_fp(omega(f), n_thz[i]))
        trans_foc.append(T_foc(f))
        trans_overlap.append(T_overlap(f, probe_waist))

        E_field.append(E(n_thz[i], omega(f), z[i], t_pump)**2)

    E_det = [f*fp*foc*ovr*E for f, fp, foc, ovr, E in zip(freq_resp, trans_fp, trans_foc, trans_overlap, E_field)]
    # E_det = [f*e for f, e in zip(freq_resp, E_field)]

    return (E_field, freq_resp, trans_fp, trans_foc, trans_overlap, E_det)

#N_THZ CALCULATION
# power = lambda x: 1346*(x**-2.373) + 3.34 # coefficients from Matlab power fit of experimental data (Parsons)
power = lambda x: (7.732e-12)*(x**-2.373) + 3.34 # coefficients from Matlab power fit of experimental data (Parsons) EDIT: corrected version; previous was using um, this is using m as wavelength unit
# parsons = pd.read_csv('~/Documents/PhD/THz-TDS-Frequency-Response/Parsons.csv')
