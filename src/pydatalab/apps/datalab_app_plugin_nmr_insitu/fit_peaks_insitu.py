import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import lmfit
from lmfit.models import PseudoVoigtModel
import numpy as np
from scipy.signal import find_peaks

data_name = "LiLiTEGDMEinsitu_02"
data_subDir = f"example_data/{data_name}"
file_name = "_NMRdata.json"
file_path = f"{data_subDir}/{data_name}{file_name}"
M_Li = pd.read_json(file_path)

ppm = M_Li['ppm']

env_peak1 = []
env_peak2 = []

file_name = "dfenv_"
file_path = f"{data_subDir}/{file_name}{data_name}.json"
data_df = pd.read_json(file_path)

tNMR = data_df['time']
env = data_df['intensity']
norm_intensity = data_df['norm_intensity']

for x in range(1, M_Li.shape[1]):
    intensity = M_Li.iloc[:, x]

    model1 = PseudoVoigtModel(prefix='peak1_')
    model2 = PseudoVoigtModel(prefix='peak2_')

    model = model1 + model2

    params = model.make_params()
    params['peak1_amplitude'].set(value=8.976e5, min=1e5, max=6e6)
    params['peak1_center'].set(value=248.0, min=244.0, max=252.5)
    params['peak1_sigma'].set(value=5, min=0.5, max=6.5)
    params['peak1_fraction'].set(value=0.3, min=0.2, max=1)

    params['peak2_amplitude'].set(value=0, min=0, max=5e7)
    params['peak2_center'].set(value=272.0, min=257.0, max=277)
    params['peak2_sigma'].set(value=5, min=0.5, max=6.5)
    params['peak2_fraction'].set(value=0.3, min=0.2, max=1)

    result = model.fit(intensity, x=ppm, params=params)

    peak1_params = {name: param for name, param in result.params.items()
                    if name.startswith('peak1_')}
    peak2_params = {name: param for name, param in result.params.items()
                    if name.startswith('peak2_')}

    peak1_intensity = model1.eval(params=peak1_params, x=ppm)
    peak2_intensity = model2.eval(params=peak2_params, x=ppm)

    env_peak1.append(abs(np.trapz(peak1_intensity, x=ppm)))
    env_peak2.append(abs(np.trapz(peak2_intensity, x=ppm)))

norm_intensity_peak1 = [x/max(env_peak1) for x in env_peak1]
norm_intensity_peak2 = [x/max(env_peak2) for x in env_peak2]


def data_fitted(tNMR, peak_intensity, norm_intensity):
    result = pd.DataFrame({
        'time': tNMR,
        'intensity': peak_intensity,
        'norm_intensity': norm_intensity,
    })
    return result


df_peakfit1 = data_fitted(tNMR, env_peak1, norm_intensity_peak1)
df_peakfit2 = data_fitted(tNMR, env_peak2, norm_intensity_peak2)

df_all = {
    "data_df": {
        "time": data_df["time"].tolist(),
        "intensity": data_df["intensity"].tolist(),
        "norm_intensity": data_df["norm_intensity"].tolist()
    },
    "df_peakfit1": {
        "time": df_peakfit1["time"].tolist(),
        "intensity": df_peakfit1["intensity"].tolist(),
        "norm_intensity": df_peakfit1["norm_intensity"].tolist()
    },
    "df_peakfit2": {
        "time": df_peakfit2["time"].tolist(),
        "intensity": df_peakfit2["intensity"].tolist(),
        "norm_intensity": df_peakfit2["norm_intensity"].tolist()
    }
}

file_name = "_df_all.json"
file_path = f"{data_subDir}/{data_name}{file_name}"
with open(file_path, 'w') as f:
    json.dump(df_all, f, indent=4)

plt.figure(figsize=(6, 10))
plt.plot(tNMR, norm_intensity, label='Total intensity', color='blue')
plt.plot(tNMR, norm_intensity_peak1, label='Peak 1', color='red')
plt.plot(tNMR, norm_intensity_peak2, label='Peak 2', color='green')
plt.xlabel('tNMR (Hrs)')
plt.ylabel('Normalize intensity')
plt.legend()
printname = os.path.join(data_subDir, "df_all.png")
plt.savefig(printname)
