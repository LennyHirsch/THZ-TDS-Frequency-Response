from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv(
    "./compressed-spectra/1030nm-1mmBBO.txt",
    delimiter="\t",
    header=None,
    names=["Frequency", "Amplitude"],
)
print(df.head())

plt.plot(df["Frequency"], df["Amplitude"])
plt.yscale("log")
plt.show()
