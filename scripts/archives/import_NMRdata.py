import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def extract_date_from_acqus(path):

    date = None

    with open(path, 'r') as file:
        for line in file:
            if line.startswith('$$'):
                match = re.search(
                    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \+\d{4})', line)
                if match:
                    date_str = match.group(1)
                    date = datetime.strptime(
                        date_str, '%Y-%m-%d %H:%M:%S.%f %z')
                    break
    return date


#! Variables
# Jana's data Exp variables
exp_name = "2023-08-11_jana_insituLiLiTEGDME-02_galv"
subDir = f"example_data/Example-TEGDME/{exp_name}"
Npath = "pdata/1/ascii-spec.txt"

# Save Data
data_name = "LiLiTEGDMEinsitu_02"
data_subDir = f"example_data/{data_name}"

os.makedirs(data_subDir, exist_ok=True)

nos_experiments = len([d for d in os.listdir(
    subDir) if os.path.isdir(os.path.join(subDir, d))])
start_at = 1
exclude_exp = []
exp_folder = None

if exclude_exp:
    exp_folder = [exp for exp in range(
        start_at, nos_experiments + 1) if exp not in exclude_exp]
else:
    exp_folder = list(range(start_at, nos_experiments + 1))


#! Lot of variables, need to discuss if rly useful (except ppm1 and ppm2)
# Maybe not useful on datalab, and can be stored in "Description"
ppm1 = 220
ppm2 = 310
nucleus = "7Li"
compound = "Li metal"
compound2 = "metal"
cellsetup = "Li-Li"
electrolyte = "1M LiTFSI TEGDME"
cellorientation = "vertical"
separator = "celgard, glass fiber, celgard"
wires = "copper mesh anf Aluminium mesh"

# Generate a vector with the path of all the NMR exp
p2 = [f"{subDir}/{exp}/{Npath}" for exp in exp_folder]

p3 = [f"{subDir}/{exp}/acqus" for exp in exp_folder]

#! Add an input for manual tunning ?
d1 = []
t1 = []

for path in p3:
    date_time = extract_date_from_acqus(path)
    d1.append(date_time.timestamp() / 3600)

for date_time in d1:
    t1.append(date_time - d1[0])

tNMR = t1

# Read first file
fdata2 = pd.read_csv(p2[0], header=None, skiprows=1)

# Get last col of first file (ppm)
M2 = fdata2.iloc[:, 3].values

# Create a matrix of the size of ppm x folders nb
M_Li = pd.DataFrame(index=range(len(M2)), columns=range(len(p2) + 1))

# Add ppm values in col 0
M_Li.iloc[:, 0] = M2

# Add value of each folder in the matrix
for m in range(0, len(p2)):
    data2 = pd.read_csv(p2[m], header=None, skiprows=1)
    M_Li.iloc[:, m + 1] = data2.iloc[:, 1]

# Exclude value outside ppm1 and ppm2
M_Li = M_Li[(M_Li.iloc[:, 0] > ppm1) & (M_Li.iloc[:, 0] < ppm2)]

# Save before renaming col
data_Li = M_Li
file_name = "_NMRforstackplot.json"
file_path = f"{data_subDir}/{data_name}{file_name}"
data_Li.to_json(file_path)

# Rename col
new_column_names = ['ppm'] + list(range(0, M_Li.shape[1] - 1))
M_Li.columns = new_column_names

# Save after renaming col
file_name = "_NMRdata.json"
file_path = f"{data_subDir}/{data_name}{file_name}"
M_Li.to_json(file_path)

ppm = M_Li.iloc[:, 0].values

env = []

for m in range(1, M_Li.shape[1]):
    y = M_Li.iloc[:, m].values
    env.append(abs(np.trapz(y, x=ppm)))

norm_intensity = []

for x in env:
    norm_intensity.append(x/max(env))

insituLi_data = pd.DataFrame({
    'time': tNMR,
    'intensity': env,
    'norm_intensity': norm_intensity,
})

file_name = "dfenv_"
file_path = f"{data_subDir}/{file_name}{data_name}.json"
insituLi_data.to_json(file_path)

plt.figure(figsize=(6, 10))
plt.scatter(tNMR, norm_intensity)
plt.grid(True, linestyle='--', color='lightgray')
plt.xlabel("Time (hrs)")
plt.ylabel("Normalized Intensity")

printname = os.path.join(data_subDir, "normintensity.png")
plt.savefig(printname)

# She save in txt, cvs tsv ... needed ?
