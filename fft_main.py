import numpy as np
from scipy.integrate import cumulative_trapezoid as cum_trap
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
path = "./experimental-data/"
filenames = [
    "datastruct_uncompressed - 23-11-14.csv",
    "datastruct_uncompressed - prelim 1 - 23-11-14.csv",
    "datastruct_uncompressed - 23-06-23-unpurged.csv",
    "datastruct_uncompressed - 1299 chop - 23-06-24.csv",
    "datastruct_uncompressed - 22-07-11.csv",
    "datastruct_uncompressed - 22-07-11.csv",
    "datastruct_compressed - 23-05-02.csv",
]

fullpath = path + filenames[2]

(sig, pos, std) = import_thz(fullpath)
time = pos2time(pos)

normalised_sig = normalise(sig)
zeroed_time = normalise_time(time, sig)

pad = 500
step_size = pos[1] - pos[0]

# SET PARAMETERS FOR GAUSSIAN
# COMPRESSED
# start = 20
# end = 740
# t0 = 0.398e-12
# tau = 0.365e-12
# factor = 10
# UNCOMPRESSED 1299 CHOP
# start = 20
# end = 740
# t0 = 0.398e-12
# tau = 0.365e-12
# factor = 10
# UNCOMPRESSED 23-06-23 unpurged start = 0 end = 316
# start = 0
# end = 316
# t0 = -0.02e-12
# tau = 0.5e-12
# factor = 5
# UNCOMPRESSED 23-06-23 UNPURGED start = 0 end = 276
start = 0
end = 276
t0 = 0.05e-12
tau = 0.26e-12
factor = 9

# PAD TIME AND SIGNAL
padded_sig = np.array(pad_sig(normalised_sig, pad, start, end))
padded_time = np.array(pad_time(zeroed_time, pad, start, end, step_size))

# DEBUG PLOTS
# plotter(zeroed_time, orig_sig, title='Zeroed time', xlabel='Time (s)')
# plotter(padded_time, padded_sig, title='Padded data', xlabel='Time (s)')

# APPLY GAUSSIAN

gauss = normalise(gaussian(padded_time, t0, width(tau)))
gauss = gauss * factor

sig_gauss = padded_sig * (1 + gauss)

cum_sum = normalise(cum_trap(sig_gauss, padded_time))

print(f"Cumulative sum: {cum_sum[-1]}")
print(".")  # just here to keep result in terminal after execution

# PLOT COMPARISON OF SIGNALS (TIME-DOMAIN)
plotter(
    padded_time[0:-1],
    padded_sig[0:-1],
    sig_gauss[0:-1],
    cum_sum,
    title="Comparing padded sig, sig_gauss, and cum sum",
    legend=["Padded sig", "Sig gauss", "Cum sum"],
)

# FFT
(orig_fft, freq) = fft_thz(padded_time, padded_sig)
(fft, freq) = fft_thz(padded_time, sig_gauss)
orig_fft = normalise(np.abs(orig_fft) ** 2)
fft = normalise(np.abs(fft) ** 2)

xlim = [-1, 10]  # in THz
ylim = [9e-5, 2]

plotter(
    freq * 10**-12,
    orig_fft,
    fft,
    title="Comparing FFTs",
    xlabel="Frequency (THz)",
    legend=["Original", "Modified"],
    xlim=xlim,
    ylim=ylim,
    log=True,
)
