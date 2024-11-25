import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import nmrglue as ng


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


folder_path = "/Users/Ben/Desktop/datalab-app-plugin-nmr-insitu/example_data/Example-TEGDME/2023-08-11_jana_insituLiLiTEGDME-02_galv"
save_folder = "/Users/Ben/Desktop/datalab-app-plugin-nmr-insitu/example_data/Test_1r/result.json"
os.makedirs(folder_path, exist_ok=True)

nos_experiments = len([d for d in os.listdir(
    folder_path) if os.path.isdir(os.path.join(folder_path, d))])

start_at = 1
exclude_exp = []
exp_folder = None

if exclude_exp:
    exp_folder = [exp for exp in range(
        start_at, nos_experiments + 1) if exp not in exclude_exp]
else:
    exp_folder = list(range(start_at, nos_experiments + 1))


ppm1 = 220
ppm2 = 310

p2 = [f"{folder_path}/{exp}" for exp in exp_folder]
p3 = [f"{folder_path}/{exp}/pdata/1" for exp in exp_folder]
exp_DATE = []

for index, p in enumerate(p3[:]):
    p_dic, p_data = ng.fileio.bruker.read_pdata(str(p))
    test_dic, test_data = ng.fileio.bruker.read(str(p))
    udic = ng.bruker.guess_udic(p_dic, p_data)
    uc = ng.fileiobase.uc_from_udic(udic)
    ppm_scale = uc.ppm_scale()

df_list = pd.DataFrame(
    columns=['ppm'] + [f'{i+1}' for i in range(nos_experiments)])
df_list['ppm'] = ppm_scale

for index, p in enumerate(p3):
    p_dic, p_data = ng.fileio.bruker.read_pdata(str(p))
    df_list[f'{index+1}'] = p_data
    exp_DATE.append(p_dic['acqus']['DATE'])

tNMR = [(val - exp_DATE[0])/3600 for val in exp_DATE]

ppm = df_list.iloc[:, 0].values
env = []

for m in range(1, df_list.shape[1]):
    y = df_list.iloc[:, m].values
    env.append(abs(np.trapz(y, x=ppm)))

norm_intensity = []

for x in env:
    norm_intensity.append(x/max(env))

final_data = pd.DataFrame({
    'time': tNMR,
    'intensity': env,
    'norm_intensity': norm_intensity,
})

final_data.to_json(save_folder)

plt.figure(figsize=(6, 10))
plt.scatter(tNMR, norm_intensity)
plt.xlabel("Time (hrs)")
plt.ylabel("Normalized Intensity")
plt.grid(True, linestyle='--', color='lightgray')
printname = os.path.join(
    "/Users/Ben/Desktop/datalab-app-plugin-nmr-insitu/example_data/Test_1r/", "normintensity.png")
plt.savefig(printname)
