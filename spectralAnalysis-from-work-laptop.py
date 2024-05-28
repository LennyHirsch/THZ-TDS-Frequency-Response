import csv
from os import listdir
from os.path import isfile, join
from tkinter.filedialog import askopenfilename, askdirectory, asksaveasfile
from matplotlib import pyplot as plt

dirname = askdirectory(initialdir='C:/Users/2090496H/OneDrive - University of Glasgow/Documents/MATLAB/ARC THz setup/2um to 1um conversion/23-02-11 - efficiency readings of compressed probe and FFTs of compressed probe/data')
filelist = [f for f in listdir(dirname) if isfile(join(dirname, f))]
#import all datasets and store in one array
datasets = []
for filename in filelist:

    data = [[],[]] #freq(0), ampl(1)

    with open(dirname + '/' + filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in reader:
            for i, item in enumerate(row):
                data[i].append(row[i])

    #convert strings to floats
    for i, item in enumerate(data[0]):
        data[0][i] = float(data[0][i])
        data[1][i] = float(data[1][i])

    maxAmpl = max(data[1])

    for i, item in enumerate(data[1]):
        data[1][i] = item/maxAmpl

    datasets.append(data)
    print(filename + ' imported.')

for set in datasets:
    plt.plot(set[0],set[1])

# plt.plot(datasets[0][0], datasets[0],[1])
plt.legend(labels=filelist)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalised counts')
plt.title('Comparison of OPA pump spectra and 1030nm second harmonic with 0.5mm BBO')
plt.show()

print('done')