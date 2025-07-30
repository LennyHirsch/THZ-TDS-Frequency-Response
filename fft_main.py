import numpy as np
import csv
from scipy.integrate import cumulative_trapezoid as cum_trap
from matplotlib import pyplot as plt
from fft_lib import (
    import_thz,
    pos2time,
    normalise,
    normalise_time,
    pad_sig,
    pad_time,
    fft_thz,
    plotter,
    gaussian,
    width,
)

# IMPORT RAW DATA
# path = "./traces/cooled/"
path = "./traces/"
filenames = [
    "datastruct_compressed - 23-05-02.csv",
    "config4.csv",
    # "28W-400uW-cooled.csv",
    # "28W-400uW-uncooled.csv",
    # "22W-400uW-cooled.csv",
    # "22W-400uW-uncooled.csv",
]


def mod_fft(filepath, pad, start, end, t0, tau, factor):
    (sig, pos, std) = import_thz(filepath)
    time = pos2time(pos)

    normalised_sig = normalise(sig)
    zeroed_time = normalise_time(time, sig)

    step_size = pos[1] - pos[0]

    padded_sig = np.array(pad_sig(normalised_sig, pad, start, end))
    padded_time = np.array(pad_time(zeroed_time, pad, start, end, step_size))

    gauss = normalise(gaussian(padded_time, t0, width(tau)))
    gauss = gauss * factor

    sig_gauss = padded_sig * (1 + gauss)

    cum_sum = cum_trap(sig_gauss, padded_time)
    print(f"Cumulative sum: {cum_sum[-1]}")

    (fft, freq) = fft_thz(padded_time, sig_gauss)
    fft = normalise(np.abs(fft) ** 2)
    plt.plot(freq * 10**-12, fft)

    return (freq * 10**-12, fft)


# -----------------------i---pad--s---e------t0-------tau-----f----
# compressed = mod_fft(path + filenames[0], 500, 20, 740, 0.398e-12, 0.365e-12, 10)

compressed = mod_fft(path + filenames[0], 500, 0, -1, 0.398e-12, 0.365e-12, 0)
# COOLED
# uncompressed = mod_fft(path + filenames[1], 500, 0, -1, -0.02e-12, 0.5e-12, 5)  # COOLED

# mod_fft(path + filenames[1], 500, 0, -1, -0.48e-12, 0.45e-12, 5)  # UNCOOLED
plt.yscale("log")

xlim = [0, 10]  # in THz
ylim = [5e-4, 2]
plt.title("Comparing compressed and uncompressed experimental spectra")
plt.xlabel("Frequency (THz)")
plt.ylabel("Power spectrum (a.u.)")
plt.legend(["Cooled", "Uncooled"])
plt.xticks(np.arange(0, 10, 1))
# plt.xlim(xlim)
# plt.axhline(0.5, linestyle=":", color="k")
# plt.ylim(ylim)
plt.show()

with open("experimental_data_fig9.csv", "w", newline="") as file:
    writer = csv.writer(file, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(compressed[0])
    writer.writerow(compressed[1])
    writer.writerow(uncompressed[0])
    writer.writerow(uncompressed[1])
