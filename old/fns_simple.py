import numpy as np
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

power = lambda x: (7.732e-12)*(x**-2.373) + 3.34

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

# HELPER FUNCTIONS
k = lambda freq: 2*np.pi*freq / c
omega = lambda freq: 2*np.pi*freq
wvl = lambda freq: c / freq
frq = lambda wave: c / wave

# FINAL TRANSFER FUNCTION
def transfer_function(freq, wvl_probe, t_probe, t_pump, n_thz, z):
    E_field = []
    freq_resp = []
    dk = []
    E_det = []

    for i, f in enumerate(freq):
        delta_k = k(frq(wvl_probe)) + k(f) - k(frq(wvl_probe) - f)
        dk.append(delta_k)
        freq_resp.append( freq_response(Aopt(omega(f), t_probe), X2, deltaPhi(L_det, delta_k)) )
        E_field.append(E(n_thz[i], omega(f), z[i], t_pump)**2)

    E_det = [f*e for f, e in zip(freq_resp, E_field)]

    return (E_field, E_det, freq_resp, dk)

# NORMALISATION
def normalise(array):
    max_val = max(array)
    normalised = [a/max_val for a in array]
    return (normalised)
