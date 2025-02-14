import os
import re
import warnings

from typing import List, Optional, Dict, Tuple
from navani import echem as ec
from datetime import datetime

import numpy as np
import pandas as pd
import nmrglue as ng


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
                        f"acqu2s file found in experiment {exp_folder}")

            return "1D"

    except Exception as e:
        raise RuntimeError(f"Error checking NMR dimension: {str(e)}")


def extract_td_parameters(acqus_path: str) -> Tuple[Optional[int], Optional[str]]:
    """Extract TD and TD_INDIRECT parameters from acqus file."""
    try:
        with open(acqus_path, 'r') as file:
            content = file.read()
            td_match = re.search(r'##\$TD=\s*(\d+)', content)
            td_indirect_match = re.search(
                r'##\$TD_INDIRECT=(.*?)(?=##|\Z)', content, re.DOTALL)

            td_value = int(td_match.group(1)) if td_match else None
            td_indirect = td_indirect_match.group(
                1).strip() if td_indirect_match else None

            return td_value, td_indirect

    except Exception as e:
        raise RuntimeError(f"Error extracting TD parameters: {str(e)}")


def extract_date_from_acqus(path: str) -> Optional[datetime]:
    """Extract date from acqus file."""
    try:
        with open(path, 'r') as file:
            for line in file:
                if line.startswith('$$'):
                    match = re.search(
                        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \+\d{4})', line)
                    if match:
                        return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S.%f %z')

    except Exception as e:
        raise ValueError(f"Warning: Could not extract date from {path}: {e}")
    return None


def setup_paths(nmr_folder_path: str, start_at: int, exclude_exp: Optional[List[int]]) -> Tuple[List[str], List[str]]:
    """Setup experiment paths and create output directory."""
    exp_folders = [d for d in os.listdir(nmr_folder_path)
                   if os.path.isdir(os.path.join(nmr_folder_path, d)) and d.isdigit()]

    exp_folder = [exp for exp in range(start_at, len(exp_folders) + 1)
                  if exp not in (exclude_exp or [])]

    spec_paths = [
        f"{nmr_folder_path}/{exp}/pdata/1/ascii-spec.txt" for exp in exp_folder]
    acqu_paths = [f"{nmr_folder_path}/{exp}/acqus" for exp in exp_folder]

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

    column_names = ['ppm'] + \
        [f'{i+1}' for i in range(len(spec_paths))]
    nmr_data = pd.DataFrame(index=range(len(ppm_values)), columns=column_names)
    nmr_data['ppm'] = ppm_values

    num_experiments = len(spec_paths)

    for i, path in enumerate(spec_paths):
        data = pd.read_csv(path, header=None, skiprows=1)
        nmr_data[f'{i+1}'] = data.iloc[:, 1]

    nmr_data = nmr_data[(nmr_data['ppm'] >= ppm1) & (nmr_data['ppm'] <= ppm2)]

    intensities = calculate_intensities(nmr_data)

    df = pd.DataFrame({
        'time': time_points,
        'intensity': intensities,
        'norm_intensity': intensities / np.max(intensities),
    })

    return nmr_data, df, num_experiments


def process_pseudo2d_spectral_data(exp_dir: str, ppm1: float, ppm2: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process pseudo-2D spectral data from Bruker files."""

    p_dic, p_data = ng.fileio.bruker.read_pdata(
        os.path.join(exp_dir, "pdata", "1"))

    udic = ng.bruker.guess_udic(p_dic, p_data)
    uc = ng.fileiobase.uc_from_udic(udic)
    ppm_scale = uc.ppm_scale()

    num_experiments = int(p_data.shape[0])

    column_names = ['ppm'] + \
        [f'{i+1}' for i in range(num_experiments)]
    nmr_data = pd.DataFrame(columns=column_names)
    nmr_data['ppm'] = ppm_scale

    for i in range(num_experiments):
        nmr_data[f'{i+1}'] = p_data[i]

    nmr_data = nmr_data[(nmr_data['ppm'] >= ppm1) & (nmr_data['ppm'] <= ppm2)]

    intensities = calculate_intensities(nmr_data)

    time_points = np.arange(num_experiments, dtype=float)
    df = pd.DataFrame({
        'time': time_points,
        'intensity': intensities,
        'norm_intensity': intensities / np.max(intensities),
    })

    return nmr_data, df, num_experiments


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


def prepare_for_bokeh(nmr_data: pd.DataFrame, df: pd.DataFrame, echem_df: Optional[pd.DataFrame], num_experiments: int) -> Dict:
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
            },
            "num_experiments": num_experiments,
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


def calculate_intensities(data: pd.DataFrame, ppm_col: str = 'ppm') -> np.ndarray:
    """Calculate intensities for spectral data using vectorized operations."""
    cols = [col for col in data.columns if col != ppm_col]
    ppm_values = data[ppm_col].values

    intensities = np.array([
        abs(np.trapz(data[col].values, x=ppm_values))
        for col in cols
    ])

    return intensities


def _process_data(
    base_folder: str,
    nmr_folder_path: str,
    echem_folder_name: str,
    ppm1: float,
    ppm2: float,
    start_at: int,
    exclude_exp: Optional[List[int]]
) -> Dict:
    """
    Common processing logic for both local and Datalab data.

    Args:
        base_folder: Path to the base folder containing all data
        nmr_folder_path: Path to the NMR data folder
        echem_folder_name: Name of the electrochemistry folder
        ppm1: Lower PPM range limit
        ppm2: Upper PPM range limit
        start_at: Starting experiment number
        exclude_exp: List of experiment numbers to exclude

    Returns:
        Dictionary containing processed NMR and electrochemical data
    """
    try:
        nmr_dimension = check_nmr_dimension(nmr_folder_path)

        if nmr_dimension == '1D':
            spec_paths, acqu_paths = setup_paths(
                nmr_folder_path, start_at, exclude_exp)
            time_points = process_time_data(acqu_paths)
            nmr_data, df, num_experiments = process_spectral_data(
                spec_paths, time_points, ppm1, ppm2)

        elif nmr_dimension == 'pseudo2D':
            exp_folders = [d for d in os.listdir(nmr_folder_path)
                           if os.path.isdir(os.path.join(nmr_folder_path, d)) and d.isdigit()]
            if not exp_folders:
                raise FileNotFoundError(
                    "No experiment folders found in NMR data")

            exp_folder = os.path.join(nmr_folder_path, exp_folders[0])
            nmr_data, df, num_experiments = process_pseudo2d_spectral_data(
                exp_folder, ppm1, ppm2)

        else:
            raise ValueError(f"Unknown NMR dimension type: {nmr_dimension}")

        if echem_folder_name:
            echem_path = os.path.join(base_folder, echem_folder_name, 'echem')
            merged_df = process_echem_data(base_folder, os.path.basename(
                base_folder), echem_folder_name) if os.path.exists(echem_path) else None
        else:
            merged_df = None

        return prepare_for_bokeh(nmr_data, df, merged_df, num_experiments)

    except Exception as e:
        raise RuntimeError(f"Error in common processing: {str(e)}")
