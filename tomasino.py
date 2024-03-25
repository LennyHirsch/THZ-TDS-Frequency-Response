"""This is *meant* to calculate the frequency response of EOS to a probe with given duration and a given crystal thickness. Not sure if it works though."""
import numpy as np

t_probe = 246e-15 #fs
t_probe_compressed = 55e-15
f_thz = 3e12 #THz
Omega = f_thz * 2*np.pi
L_det = 300e-6 #um
c = 299792458 #m/s
lambda_probe = 1030e-9 #nm
lambda_thz = c/f_thz
k_probe = 2*np.pi/lambda_probe
k_thz = 2*np.pi/lambda_thz
k_diff = np.abs(2*np.pi/(lambda_probe - lambda_thz))
delta_k = k_probe + k_thz - k_diff
chi = 220e-12 #pm/V

delta_Phi = (np.exp(-1j * delta_k * L_det) - 1) / 1j*delta_k

def A_opt(duration):
    return(np.sqrt(np.pi)/duration * np.exp(-Omega**2 * duration**2 / 4))

def freq(duration):
    return(A_opt(duration)*chi*delta_Phi)

F_optimal = freq(1e-15)
F_norm = freq(t_probe)
F_compr = freq(t_probe_compressed)

F_optimal = "{:.2E}".format(np.abs(np.real(F_optimal)))
F_norm = "{:.2E}".format(np.abs(np.real(F_norm)))
F_compr = "{:.2E}".format(np.abs(np.real(F_compr)))

print(f"Optimal F = {F_optimal}\nNormal F = {F_norm}\nCompressed F = {F_compr}")
