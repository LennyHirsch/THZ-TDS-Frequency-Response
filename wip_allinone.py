import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
from tomasino_fns_wip import wvl, omega, eps_thz, transfer_function, normalise
import csv

wvl_probe = 1030e-9  # probe wavelength: 1030 nm
freq_thz = 3e12  # THz frequency: 3 THz

# SETTING SIMULATION PARAMETERS
res = int(1e2)
z = np.linspace(0, 1e-3, res, dtype=np.float128)
freq = np.linspace(0.01e11, 1.03e13, res, dtype=np.float128)


def all_in_one(probes):
    # contain frequencies and signals for all input probes
    freqs = []
    sigs = []
    counter = 0

    for p in probes:

        # CALCULATE ALL THE THINGS
        (Ef, f, fp, foc, ovr, E_det, dk) = transfer_function(
            freq, wvl_probe, p, t_pp, n_thz, z, 3.5e-3
        )

        # NORMALISING TO 1
        E_det = normalise(E_det)

        E_pow = [e**2 for e in np.abs(E_det)]
        E_pow = normalise(E_pow)
        plt.plot(freq, E_pow, label=(p * (2 * np.sqrt(np.log(2))) * 1e15).astype("str"))

        freqs.append(freq)
        sigs.append(E_pow)
        counter += 1

    plt.legend()
    plt.yscale("log")
    plt.title("Detected E-field (Frequency domain)")
    plt.xlabel("Freq")
    plt.ylabel("E field ^2")
    plt.xticks(np.arange(0, 1e13, 1e12))
    starty = 5e-4
    endy = 2
    plt.ylim([starty, endy])
    startx = 0
    endx = 1e13
    plt.xlim([startx, endx])
    plt.show()

    return (freqs, sigs)


tpb = lambda t: t / (2 * np.sqrt(np.log(2)))

t_pump = 150e-15
t_pp = t_pump / (2 * np.sqrt(np.log(2)))

# CALCULATING N_THZ: THIS IS DONE FROM A FITTING EQUATION OF EXPERIMENTAL DATA
x = np.linspace(wvl(min(freq)), wvl(max(freq)), res)
n_thz = np.sqrt([eps_thz(omega(f)) for f in freq])

probes = [tpb(245e-15), tpb(55e-15)]

(freqs, sigs) = all_in_one(probes)

plt.plot(freqs[0], sigs[0])
plt.plot(freqs[1], sigs[1])

with open("simulation_data.csv", "w", newline="") as file:
    writer = csv.writer(file, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(freqs[0])
    writer.writerow(sigs[0])
    writer.writerow(sigs[1])
