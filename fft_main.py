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
path = "./traces/"
filenames = [
    "datastruct_uncompressed - 23-11-14.csv",  # 0
    "datastruct_uncompressed - prelim 1 - 23-11-14.csv",  # 1
    "datastruct_uncompressed - 23-06-23-unpurged.csv",  # 2
    "datastruct_uncompressed - 1299 chop - 23-06-24.csv",  # 3
    "datastruct_uncompressed - 22-07-11.csv",  # 4
    "datastruct_uncompressed - 22-07-11.csv",  # 5
    "datastruct_compressed - 23-05-02.csv",  # 6
    "28W-pump-400uW-probe-2khz-chop-5-hum-100k-rep-cooled.csv",  # 7
    "28W-pump-400uW-probe-2khz-chop-5-hum-100k-rep-uncooled.csv",  # 8
    "22-07-13-high_res_scan.csv",  # 9
    "23-06-23-trace-6k-sampling-100ms-tc-15W-pump-400uW-probe-2kHz chop.csv",  # 10
    "23-06-24-uncompressed-allegedly.csv",  # 11
    "23-11-24-uncooled-400uW-3s-sampling-1mm-trace-apparently-uncomp.csv",  # 12
    "24-02-03-16_8W-100uW-100ms_polling-6k_sampling-compressed_probe.csv",  # 13
]


fullpath = path + filenames[10]

(sig, pos, std) = import_thz(fullpath)
time = pos2time(pos)

normalised_sig = normalise(sig)
zeroed_time = normalise_time(time, sig)

pad = 500
step_size = pos[1] - pos[0]

# SET PARAMETERS FOR GAUSSIAN
# start = 0
# end = -1
# t0 = -0e-12
# tau = 0.6e-12
# factor = 3.7
start = 0
end = 297
t0 = 0e-12
tau = 0.2e-12
factor = 1

# plotter(range(0, len(sig)), normalised_sig)

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

cum_sum = cum_trap(sig_gauss, padded_time)

print(f"Cumulative sum: {cum_sum[-1]}")
print(".")  # just here to keep result in terminal after execution

# PLOT COMPARISON OF SIGNALS (TIME-DOMAIN)
plotter(
    padded_time[0:-1],
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
ylim = [5e-6, 2]

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
