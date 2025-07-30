import pandas as pd
from scipy.io import loadmat
from matplotlib import pyplot as plt

data = loadmat("./early traces/dataTrace1.mat")

df = pd.DataFrame(
    {
        "pos": data["dataStruct"]["pos"][0][0][0],
        "sig": data["dataStruct"]["x_mean"][0][0][0],
        "std": data["dataStruct"]["x_std"][0][0][0],
    }
)

print(df.head())

plt.plot(df["pos"], df["sig"])
plt.plot(df["pos"], df["std"])

plt.show()
