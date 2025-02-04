import os
import re
import zipfile
import tempfile
import warnings


from datalab_api import DatalabClient
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from lmfit.models import PseudoVoigtModel
from navani import echem as ec

import numpy as np
import pandas as pd


def check_nmr_dimension(nmr_folder_path: str) -> str:
    """
    Check if NMR data is 1D or pseudo 2D based on:
    - Single experiment folder: must have both acqus and acqu2s (pseudo2D)
    - Multiple experiment folders: must have only acqus, no acqu2s (1D)

    Args:
        nmr_folder_path (str): Path to the NMR data folder containing numbered experiment folders

    Returns:
        str: 'pseudo2D' for single experiment with acqu2s, '1D' for multiple experiments

    Raises:
        FileNotFoundError: If required files are missing
        RuntimeError: If there's an error accessing the files or invalid configuration
    """
    try:
        exp_folders = [d for d in os.listdir(nmr_folder_path)
                       if os.path.isdir(os.path.join(nmr_folder_path, d)) and d.isdigit()]

        if not exp_folders:
            raise FileNotFoundError("No experiment folders found in NMR data")

        if len(exp_folders) == 1:
            exp_path = os.path.join(nmr_folder_path, exp_folders[0])
            acqus_path = os.path.join(exp_path, "acqus")
            acqu2s_path = os.path.join(exp_path, "acqu2s")

            if not os.path.exists(acqus_path):
                raise FileNotFoundError(
                    f"acqus file not found in experiment {exp_folders[0]}")

            if not os.path.exists(acqu2s_path):
                raise FileNotFoundError(
                    f"acqu2s file not found in experiment {exp_folders[0]} - required for pseudo2D")

            return "pseudo2D"

        else:
            for exp_folder in exp_folders:
                exp_path = os.path.join(nmr_folder_path, exp_folder)
                acqus_path = os.path.join(exp_path, "acqus")
                acqu2s_path = os.path.join(exp_path, "acqu2s")

                if not os.path.exists(acqus_path):
                    raise FileNotFoundError(
                        f"acqus file not found in experiment {exp_folder}")

                if os.path.exists(acqu2s_path):
                    raise RuntimeError(
                        f"acqu2s file found in experiment {exp_folder} - not allowed for 1D experiments")

            return "1D"

    except Exception as e:
        raise RuntimeError(f"Error checking NMR dimension: {str(e)}")


def extract_date_from_acqus(path: str) -> Optional[datetime]:
    """Extract date from acqus file."""
    try:
        with open(path, 'r') as file:
            for line in file:
                if line.startswith('$$'):
                    match = re.search(
                        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \+\d{4})', line)
                    if match:
                        date_str = match.group(1)
                        return datetime.strptime(
                            date_str, '%Y-%m-%d %H:%M:%S.%f %z')
    except Exception as e:
        print(f"Warning: Could not extract date from {path}: {e}")
    return None


def setup_paths(nmr_folder_path: str, start_at: int, exclude_exp: Optional[List[int]]) -> Tuple[List[str], List[str]]:
    """Setup experiment paths and create output directory."""

    nos_experiments = len([d for d in os.listdir(
        nmr_folder_path) if os.path.isdir(os.path.join(nmr_folder_path, d))])

    exp_folder = [exp for exp in range(
        start_at, nos_experiments + 1) if exp not in (exclude_exp or [])]

    spec_paths = [
        f"{nmr_folder_path}/{exp}/pdata/1/ascii-spec.txt" for exp in exp_folder]
    acqu_paths = [
        f"{nmr_folder_path}/{exp}/acqus" for exp in exp_folder]

    return spec_paths, acqu_paths


def process_time_data(acqu_paths: List[str]) -> List[float]:
    """Process time data from acqus files."""
    timestamps = []
    for path in acqu_paths:
        date_time = extract_date_from_acqus(path)
        if date_time:
            timestamps.append(date_time.timestamp() / 3600)
        else:
            raise ValueError(f"Could not extract date from {path}")

    time_points = [t - timestamps[0] for t in timestamps]
    return time_points


def process_spectral_data(spec_paths: List[str], time_points: List[float], ppm1: float, ppm2: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process spectral data from ascii-spec files with PPM range filtering."""
    first_data = pd.read_csv(spec_paths[0], header=None, skiprows=1)
    ppm_values = first_data.iloc[:, 3].values

    nmr_data = pd.DataFrame(index=range(len(ppm_values)),
                            columns=['ppm'] + [str(i) for i in range(1, len(spec_paths) + 1)])
    nmr_data['ppm'] = ppm_values

    for i, path in enumerate(spec_paths):
        data = pd.read_csv(path, header=None, skiprows=1)
        nmr_data[str(i + 1)] = data.iloc[:, 1]

    nmr_data = nmr_data[(nmr_data['ppm'] >= ppm1) & (nmr_data['ppm'] <= ppm2)]

    intensities = []
    for m in range(1, nmr_data.shape[1]):
        y = nmr_data.iloc[:, m].values
        intensities.append(abs(np.trapz(y, x=nmr_data['ppm'])))

    norm_intensities = [x/max(intensities) for x in intensities]

    df = pd.DataFrame({
        'time': time_points,
        'intensity': intensities,
        'norm_intensity': norm_intensities,
    })

    return nmr_data, df


def process_echem_data(tmpdir: str, folder_name: str, echem_folder_name: str) -> Optional[pd.DataFrame]:
    """Process electrochemical data if available."""
    echem_folder_path = os.path.join(
        tmpdir, folder_name, echem_folder_name, 'echem')

    if not os.path.exists(echem_folder_path):
        warnings.warn(
            f"Echem folder not found: {echem_folder_name}. Continuing without echem data.")
        return None

    try:
        gcpl_full_paths = []
        for filename in os.listdir(echem_folder_path):
            if "GCPL" in filename and filename.endswith(".mpr"):
                full_path = os.path.join(
                    echem_folder_path, filename)
                gcpl_full_paths.append(full_path)

        all_echem_df = []
        for path in gcpl_full_paths:
            raw_df = ec.echem_file_loader(path)
            all_echem_df.append(raw_df)

        merged_df = pd.concat(all_echem_df, axis=0)
        return merged_df.sort_index()
    except Exception as e:
        warnings.warn(
            f"Error processing echem data: {str(e)}. Continuing without echem data.")
        return None


def prepare_for_bokeh(nmr_data: pd.DataFrame, df: pd.DataFrame, echem_df: Optional[pd.DataFrame]) -> Dict:
    """Prepare data for Bokeh visualization, with optional echem data."""
    result = {
        "metadata": {
            "ppm_range": {
                "start": nmr_data['ppm'].min(),
                "end": nmr_data['ppm'].max()
            },
            "time_range": {
                "start": df['time'].min(),
                "end": df['time'].max()
            }
        },
        "nmr_spectra": {
            "ppm": nmr_data["ppm"].tolist(),
            "spectra": [
                {
                    "time": df["time"][i],
                    "intensity": nmr_data[str(i+1)].tolist()
                }
                for i in range(len(df))
            ]
        }
    }

    if echem_df is not None:
        result["echem"] = {
            "Voltage": echem_df["Voltage"].tolist(),
            "time": (echem_df["time/s"] / 3600).tolist()
        }

    return result


def process_data(
    api_url: str,
    item_id: str,
    folder_name: str,
    nmr_folder_name: str,
    echem_folder_name: str,
    ppm1: float,
    ppm2: float,
    start_at: int = 1,
    exclude_exp: Optional[List[int]] = None,
) -> Dict:
    """
    Process NMR spectroscopy data from multiple experiments.

    Args:
        api_url (str): URL of the Datalab API
        folder_name (str): Base folder
        nmr_folder_name (str): Folder containing NMR experiments,
        echem_folder_name (str): Folder containing Echem data,
        ppm1 (float): Lower PPM range limit
        ppm2 (float): Upper PPM range limit
        start_at (int, optional): Starting experiment number. Defaults to 1
        exclude_exp (List[int], optional): List of experiment numbers to exclude

    Returns:
        pandas.DataFrame: A dataframe with insitu NMR data: time, intensities and normalised intensities
    """

    if not all([folder_name, nmr_folder_name]):
        raise ValueError("Folder names for NMR data are required")

    client = DatalabClient(api_url)

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.chdir(tmpdir)
            try:
                client.get_item_files(item_id=item_id)
            except Exception as e:
                raise RuntimeError(f"API error: {e}")

            zip_path = os.path.join(tmpdir, folder_name)

            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"ZIP file not found: {folder_name}")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            folder_name = os.path.splitext(folder_name)[0]
            nmr_folder_name = os.path.splitext(nmr_folder_name)[0]
            nmr_folder_path = os.path.join(
                tmpdir, folder_name, nmr_folder_name)

            echem_folder_path = os.path.join(
                tmpdir, folder_name, echem_folder_name)

            if not os.path.exists(nmr_folder_path):
                raise FileNotFoundError(
                    f"NMR folder not found: {nmr_folder_name}")

            if not os.path.exists(echem_folder_path):
                warnings.warn(
                    f"Echem folder not found: {echem_folder_name}")

            # Process data
            nmr_dimension = check_nmr_dimension(nmr_folder_path)

            if nmr_dimension == '1D':
                spec_paths, acqu_paths = setup_paths(
                    nmr_folder_path, start_at, exclude_exp)
                time_points = process_time_data(acqu_paths)
                nmr_data, df = process_spectral_data(
                    spec_paths, time_points, ppm1, ppm2)
                merged_df = process_echem_data(
                    tmpdir, folder_name, echem_folder_name) if echem_folder_name else None
                result = prepare_for_bokeh(nmr_data, df, merged_df)
            elif nmr_dimension == 'pseudo2D':
                print(nmr_dimension)
            else:
                raise ValueError(
                    f"Unknown NMR dimension type: {nmr_dimension}")

            return result

        except Exception as e:
            raise RuntimeError(f"Error processing NMR data: {str(e)}")


#! Will need to be handle by UI at some point if we want to keep fitting
FITTING_CONFIG = {
    'peak1': {
        'amplitude': {'value': 8.976e5, 'min': 1e4, 'max': 6e7},
        'center': {'value': 248.0, 'min': 244.0, 'max': 252.5},
        'sigma': {'value': 5, 'min': 0.5, 'max': 6.5},
        'fraction': {'value': 0.3, 'min': 0.2, 'max': 1}
    },
    'peak2': {
        'amplitude': {'value': 12.394e5, 'min': 0, 'max': 5e7},
        'center': {'value': 266.0, 'min': 256.0, 'max': 276},
        'sigma': {'value': 5, 'min': 0.5, 'max': 6.5},
        'fraction': {'value': 0.3, 'min': 0.2, 'max': 1}
    }
}


def fitting_data(nmr_data: pd.DataFrame, df: pd.DataFrame, config: dict = FITTING_CONFIG) -> Dict:
    """
    Perform fitting using pseudo-Voigt Model on insitu NMR data.

    Args:
        nmr_data (pd.DataFrame): Raw NMR spectral data from previous processing
        df (pd.DataFrame): Time, Intensities and Normalised intensities data from previous processing

    Returns:
        Dict: Fitting results and processed data
    """
    try:

        ppm = np.array(nmr_data['ppm'], dtype=float)
        tNMR = np.array(df['time'], dtype=float)
        env = np.array(df['intensity'], dtype=float)
        env_peak1 = []
        env_peak2 = []

        for x in range(1, nmr_data.shape[1]):
            intensity = np.array(nmr_data.iloc[:, x], dtype=float)

            model1 = PseudoVoigtModel(prefix='peak1_')
            model2 = PseudoVoigtModel(prefix='peak2_')

            model = model1 + model2

            params = model.make_params()
            for param, settings in config['peak1'].items():
                params[f'peak1_{param}'].set(**settings)
            for param, settings in config['peak2'].items():
                params[f'peak2_{param}'].set(**settings)

            result = model.fit(intensity, x=ppm, params=params)

            peak1_params = {name: param for name, param in result.params.items()
                            if name.startswith('peak1_')}
            peak2_params = {name: param for name, param in result.params.items()
                            if name.startswith('peak2_')}

            peak1_intensity = model1.eval(
                params=peak1_params, x=ppm)
            peak2_intensity = model2.eval(
                params=peak2_params, x=ppm)

            env_peak1.append(abs(np.trapz(peak1_intensity, x=ppm)))
            env_peak2.append(abs(np.trapz(peak2_intensity, x=ppm)))

        norm_intensity_peak1 = [x/max(env) for x in env_peak1]
        norm_intensity_peak2 = [x/max(env) for x in env_peak2]

        def data_fitted(tNMR, peak_intensity, norm_intensity):
            result = pd.DataFrame({
                'time': tNMR,
                'intensity': peak_intensity,
                'norm_intensity': norm_intensity,
            })
            return result

        df_peakfit1 = data_fitted(tNMR, env_peak1, norm_intensity_peak1)
        df_peakfit2 = data_fitted(tNMR, env_peak2, norm_intensity_peak2)

        df_fit = {
            "data_df": {
                "time": df["time"].tolist(),
                "intensity": df["intensity"].tolist(),
                "norm_intensity": df["norm_intensity"].tolist()
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

        return df_fit

    except Exception as e:
        raise RuntimeError(f"Error fitting NMR data: {str(e)}")
