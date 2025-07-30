import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("./traces/config4.csv")
df.columns = ["sig", "pos", "std"]

df["pos"] = df["pos"].values[::-1]

c = 299792458

df["time"] = df["pos"] / c

df["snr"] = df["sig"] / df["std"]

plt.plot(df["time"], df["sig"] * 100000)
plt.plot(df["time"], df["snr"])
plt.show()

print(f"Max SNR: {df['snr'].abs().max()}")
print(f"Mean SNR: {df['snr'].abs().mean()}")
