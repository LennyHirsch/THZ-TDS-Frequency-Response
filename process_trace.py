import pandas as pd
from scipy.io import loadmat
from matplotlib import pyplot as plt

df = pd.read_csv(
    "./traces/24-02-03-16_8W-100uW-100ms_polling-6k_sampling-compressed_probe.csv"
)


df.columns = ["sig", "pos", "std"]

df["pos"] = df["pos"].values[::-1]

c = 299792458

df["time"] = df["pos"] / c
df["snr"] = df["sig"] / df["std"]

plt.plot(df["time"], df["sig"])
plt.plot(df["time"], df["std"] * 100)
plt.show()

print(f"Max SNR: {df['snr'].abs().max()}")
print(f"Mean SNR: {df['snr'].abs().mean()}")
