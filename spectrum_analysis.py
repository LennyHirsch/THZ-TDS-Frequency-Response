import pandas as pd
from matplotlib import pyplot as plt

dir = "compressed-spectra/"
filename = "1030nm_500umBBO_156umFocalSpot.txt"
spec = pd.read_csv(dir + filename, delimiter="\t", header=None)
spec.columns = ["wvl", "ampl"]

# normalise to 0 -> 1
spec["ampl"] = (spec["ampl"] - spec["ampl"].min()) / (
    spec["ampl"].max() - spec["ampl"].min()
)

plt.plot(spec["wvl"], spec["ampl"])
plt.axhline(0.5, linestyle=":", color="k")
plt.show()
