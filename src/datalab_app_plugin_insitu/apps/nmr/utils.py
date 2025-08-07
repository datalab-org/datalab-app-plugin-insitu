import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nmrglue as ng
import numpy as np
import pandas as pd
from navani import echem as ec

from datalab_app_plugin_insitu.utils import (
    _find_folder_path,
    should_skip_path,
)


def check_nmr_dimension(nmr_folder_path: Path) -> str:
    """
    Check if NMR data is 1D or pseudo 2D based on:
    - Single experiment folder: must have both acqus and acqu2s (pseudo2D)
    - Multiple experiment folders: must have only acqus, no acqu2s (1D)

    Args:
        nmr_folder_path: Path to the NMR data folder containing numbered experiment folders

    Returns:
        str: 'pseudo2D' for single experiment with acqu2s, '1D' for multiple experiments

    Raises:
        FileNotFoundError: If required files are missing
        RuntimeError: If there's an error accessing the files or invalid configuration
    """
    try:
        if not nmr_folder_path.exists() or not nmr_folder_path.is_dir():
            raise FileNotFoundError(f"NMR folder path is not a valid directory: {nmr_folder_path}")

        exp_folders = [
            d
            for d in Path(nmr_folder_path).iterdir()
            if d.is_dir() and d.name.isdigit() and not should_skip_path(d)
        ]

        if not exp_folders:
            raise FileNotFoundError("No experiment folders found in NMR data")

        if len(exp_folders) == 1:
            exp_path = Path(nmr_folder_path) / exp_folders[0]
            acqus_path = exp_path / "acqus"
            acqu2s_path = exp_path / "acqu2s"

            if not acqus_path.exists():
                raise FileNotFoundError(f"acqus file not found in experiment {exp_folders[0]}")

            if not acqu2s_path.exists():
                raise FileNotFoundError(
                    f"acqu2s file not found in experiment {exp_folders[0]} - required for pseudo2D"
                )

            return "pseudo2D"

        else:
            for exp_folder in exp_folders:
                exp_path = Path(nmr_folder_path) / exp_folder
                acqus_path = exp_path / "acqus"
                acqu2s_path = exp_path / "acqu2s"

                if not acqus_path.exists():
                    raise FileNotFoundError(f"acqus file not found in experiment {exp_folder}")

                if acqu2s_path.exists():
                    raise RuntimeError(f"acqu2s file found in experiment {exp_folder}")

            return "1D"

    except Exception as e:
        raise RuntimeError(f"Error checking NMR dimension: {str(e)}")


def extract_td_parameters(acqus_path: str) -> Tuple[Optional[int], Optional[str]]:
    """Extract TD and TD_INDIRECT parameters from acqus file."""
    try:
        with open(acqus_path) as file:
            content = file.read()
            td_match = re.search(r"##\$TD=\s*(\d+)", content)
            td_indirect_match = re.search(r"##\$TD_INDIRECT=(.*?)(?=##|\Z)", content, re.DOTALL)

            td_value = int(td_match.group(1)) if td_match else None
            td_indirect = td_indirect_match.group(1).strip() if td_indirect_match else None

            return td_value, td_indirect

    except Exception as e:
        raise RuntimeError(f"Error extracting TD parameters: {str(e)}")


def extract_date_from_acqus(path: str) -> Optional[datetime]:
    """Extract date from acqus file."""
    try:
        with open(path) as file:
            for line in file:
                if line.startswith("$$"):
                    match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \+\d{4})", line)
                    if match:
                        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f %z")

    except Exception as e:
        raise ValueError(f"Warning: Could not extract date from {path}: {e}")
    return None


def setup_bruker_paths(
    nmr_folder_path: Path,
    start_at: int,
    end_at: Optional[int] = None,
    step: Optional[int] = None,
    exclude_exp: Optional[List[int]] = None,
) -> Tuple[List[str], List[str], List[int]]:
    """Setup experiment paths within the Bruker project and create output directory."""
    exp_folders = [d for d in Path(nmr_folder_path).iterdir() if d.is_dir() and d.name.isdigit()]

    max_exp = len(exp_folders)

    if start_at < 1:
        raise ValueError(f"start_exp must be >= 1, got {start_at}")
    if start_at > max_exp:
        raise ValueError(f"start_exp ({start_at}) exceeds available experiments ({max_exp})")

    end_at = end_at if end_at is not None else max_exp
    end_at = min(end_at, max_exp)

    if step is None:
        # aim for a default of ~50 experiments
        step = max(1, (end_at - start_at + 1) // 50)

    if step < 1:
        raise ValueError(f"step_exp must be >= 1, got {step}")

    if step > (end_at - start_at + 1):
        step = end_at - start_at + 1

    if end_at < start_at:
        raise ValueError(f"end_exp ({end_at}) must be >= start_exp ({start_at})")

    exp_numbers = [
        exp for exp in range(start_at, end_at + 1, step) if exp not in (exclude_exp or [])
    ]

    if not exp_numbers:
        raise ValueError(
            f"No experiments selected (start: {start_at}, end: {end_at}, step: {step}, exclude: {exclude_exp})"
        )

    spec_paths = [
        str(Path(nmr_folder_path) / str(exp) / "pdata" / "1" / "ascii-spec.txt")
        for exp in exp_numbers
    ]
    acqu_paths = [str(Path(nmr_folder_path) / str(exp) / "acqus") for exp in exp_numbers]

    return spec_paths, acqu_paths, exp_numbers


def process_time_data(acqu_paths: List[str], keep_absolute_time: bool = False) -> List[float]:
    """Process time data from acqus files."""
    timestamps = []
    for path in acqu_paths:
        date_time = extract_date_from_acqus(path)
        if date_time:
            timestamps.append(date_time.timestamp() / 3600)
        else:
            raise ValueError(f"Could not extract date from {path}")

    if keep_absolute_time:
        return timestamps

    time_points = [t - timestamps[0] for t in timestamps]
    return time_points


def process_spectral_data(
    spec_paths: List[str], time_points: List[float], exp_numbers: List[int]
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Process spectral data from ascii-spec files"""

    first_data = pd.read_csv(spec_paths[0], header=None, skiprows=1)
    ppm_values = first_data.iloc[:, 3].values

    column_names = ["ppm"] + [f"{i + 1}" for i in range(len(spec_paths))]
    nmr_data = pd.DataFrame(index=range(len(ppm_values)), columns=column_names)
    nmr_data["ppm"] = ppm_values

    num_experiments = len(spec_paths)

    for i, path in enumerate(spec_paths):
        data = pd.read_csv(path, header=None, skiprows=1)
        nmr_data[f"{i + 1}"] = data.iloc[:, 1]

    numeric_cols = [f"{i + 1}" for i in range(num_experiments)]
    global_max_intensity = nmr_data[numeric_cols].values.max()
    nmr_data[numeric_cols] = (nmr_data[numeric_cols] / global_max_intensity).round(6)

    intensities = calculate_intensities(nmr_data)

    df = pd.DataFrame(
        {
            "time": time_points,
            "intensity": intensities,
            "norm_intensity": intensities / np.max(intensities),
            "original_exp_number": [int(_) for _ in exp_numbers],
        }
    )

    return nmr_data, df, num_experiments


def process_pseudo2d_spectral_data(exp_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Process pseudo-2D spectral data from Bruker files."""

    p_dic, p_data = ng.fileio.bruker.read_pdata(Path(exp_dir) / "pdata" / "1")

    udic = ng.bruker.guess_udic(p_dic, p_data)
    uc = ng.fileiobase.uc_from_udic(udic)
    ppm_scale = uc.ppm_scale()

    num_experiments = int(p_data.shape[0])

    column_names = ["ppm"] + [f"{i + 1}" for i in range(num_experiments)]
    nmr_data = pd.DataFrame(columns=column_names)
    nmr_data["ppm"] = ppm_scale

    for i in range(num_experiments):
        nmr_data[f"{i + 1}"] = p_data[i]

    numeric_cols = [f"{i + 1}" for i in range(num_experiments)]
    global_max_intensity = nmr_data[numeric_cols].values.max()
    nmr_data[numeric_cols] = (nmr_data[numeric_cols] / global_max_intensity).round(6)

    intensities = calculate_intensities(nmr_data)

    time_points = np.arange(num_experiments, dtype=float)
    df = pd.DataFrame(
        {
            "time": time_points,
            "intensity": intensities,
            "norm_intensity": intensities / np.max(intensities),
        }
    )

    return nmr_data, df, num_experiments


def process_echem_data(
    base_folder: Path, echem_folder_path: Union[Path, str]
) -> Optional[pd.DataFrame]:
    """Process electrochemical data using .mpr file(s), and combine them if there is "GCPL" in their filename."""

    if not echem_folder_path:
        return None

    try:
        if isinstance(echem_folder_path, str):
            echem_folder_path = Path(echem_folder_path)

        echem_subfolder = echem_folder_path / "eChem"
        if echem_subfolder.exists():
            echem_folder_path = echem_subfolder

        if not echem_folder_path.exists():
            raise FileNotFoundError(f"Echem folder not found at {echem_folder_path}")

        mpr_files = [f for f in echem_folder_path.iterdir() if f.suffix.upper() == ".MPR"]

        if not mpr_files:
            raise FileNotFoundError(f"No MPR files found in {echem_folder_path}")

        if len(mpr_files) == 1:
            file_to_process = mpr_files[0]
        else:
            gcpl_files = [f for f in mpr_files if "GCPL" in f.name.upper()]

            if gcpl_files:
                files_to_process = sorted(gcpl_files)
            else:
                raise ValueError(
                    "Multiple MPR files found but none contain 'GCPL' in the filename. Cannot determine which file to use."
                )

        try:
            if len(mpr_files) == 1:
                echem_data = [ec.echem_file_loader(str(file_to_process))]
            else:
                echem_data = [
                    ec.echem_file_loader(str(file_path)) for file_path in files_to_process
                ]

            combined_data = pd.concat(echem_data, axis=0)
            return combined_data.sort_index()

        except Exception as e:
            raise RuntimeError(f"Error processing MPR files: {str(e)}")

    except Exception as e:
        raise RuntimeError(f"Error in electrochemical data processing: {str(e)}")


def prepare_for_bokeh(
    sampled_nmr_data: pd.DataFrame,
    df: pd.DataFrame,
    echem_df: Optional[pd.DataFrame],
    num_experiments: int,
) -> Dict:
    """Prepare data for Bokeh visualization, with optional echem data."""

    ppm_values = sampled_nmr_data["ppm"].values

    all_intensities = []
    for i in range(len(df)):
        spectrum_intensities = sampled_nmr_data[str(i + 1)].values
        all_intensities.extend(spectrum_intensities)

    global_max_intensity = float(max(all_intensities))

    result = {
        "metadata": {
            "time_range": {"start": float(df["time"].min()), "end": float(df["time"].max())},
            "num_experiments": num_experiments,
            "global_max_intensity": global_max_intensity,
        },
        "nmr_spectra": {
            "ppm": ppm_values.tolist(),
            "spectra": [
                {
                    "time": round(float(df["time"][i]), 4),
                    "intensity": sampled_nmr_data[str(i + 1)].tolist(),
                    "experiment_number": int(df["original_exp_number"][i])
                    if "original_exp_number" in df.columns
                    else i + 1,
                    "display_index": i + 1,
                }
                for i in range(len(df))
            ],
        },
        "integrated_data": {
            "intensity": df["intensity"].tolist(),
            "norm_intensity": df["norm_intensity"].tolist(),
            "time": df["time"].tolist(),
        },
    }

    if echem_df is not None:
        result["echem"] = {
            "Voltage": echem_df["Voltage"].tolist(),
            "time": (echem_df["time/s"] / 3600).tolist(),
        }

    return result


def calculate_intensities(data: pd.DataFrame, ppm_col: str = "ppm") -> np.ndarray:
    """Calculate intensities for spectral data using vectorized operations."""
    cols = [col for col in data.columns if col != ppm_col]
    ppm_values = data[ppm_col].values

    intensities = np.array([abs(np.trapz(data[col].values, x=ppm_values)) for col in cols])

    return intensities


def _process_data(
    base_folder: Path,
    nmr_folder_path: Path,
    echem_folder_name: str,
    start_at: int,
    end_at: Optional[int] = None,
    step: Optional[int] = None,
    exclude_exp: Optional[List[int]] = None,
) -> Dict:
    """
    Common processing logic for both local and Datalab data.

    Args:
        base_folder: Path to the base folder containing all data
        nmr_folder_path: Path to the NMR data folder
        echem_folder_name: Name of the electrochemistry folder
        start_at: Starting experiment number
        step: Desired step size for experiments, will default to keeping
            ~50 experiments
        exclude_exp: List of experiment numbers to exclude

    Returns:
        Dictionary containing processed NMR and electrochemical data
    """
    try:
        nmr_dimension = check_nmr_dimension(nmr_folder_path)

        if nmr_dimension == "1D":
            spec_paths, acqu_paths, exp_numbers = setup_bruker_paths(
                nmr_folder_path, start_at, end_at, step, exclude_exp
            )

            all_acqu_paths = [
                str(Path(nmr_folder_path) / str(i) / "acqus")
                for i in range(1, len(list(nmr_folder_path.iterdir())) + 1)
            ]
            all_timestamps = []
            for path in all_acqu_paths:
                if Path(path).exists():
                    date_time = extract_date_from_acqus(path)
                    if date_time:
                        all_timestamps.append(date_time.timestamp() / 3600)

            if all_timestamps:
                base_time = min(all_timestamps)
                selected_timestamps = []
                for i, exp_num in enumerate(exp_numbers):
                    if exp_num - 1 < len(all_timestamps):
                        selected_timestamps.append(all_timestamps[exp_num - 1])
                    else:
                        date_time = extract_date_from_acqus(acqu_paths[i])
                        if date_time:
                            selected_timestamps.append(date_time.timestamp() / 3600)

                time_points = [t - base_time for t in selected_timestamps]

            else:
                time_points = process_time_data(acqu_paths, keep_absolute_time=False)

            nmr_data, df, num_experiments = process_spectral_data(
                spec_paths, time_points, exp_numbers
            )

        elif nmr_dimension == "pseudo2D":
            exp_folders = [
                d for d in Path(nmr_folder_path).iterdir() if d.is_dir() and d.name.isdigit()
            ]
            if not exp_folders:
                raise FileNotFoundError("No experiment folders found in NMR data")

            exp_folder = str(Path(nmr_folder_path) / exp_folders[0])
            nmr_data, df, num_experiments = process_pseudo2d_spectral_data(exp_folder)

        else:
            raise ValueError(f"Unknown NMR dimension type: {nmr_dimension}")

        echem_folder_path = None
        if echem_folder_name:
            echem_folder_path = _find_folder_path(base_folder, echem_folder_name)
            if not echem_folder_path:
                raise FileNotFoundError(f"Echem folder not found: {echem_folder_name}")

        merged_df = process_echem_data(base_folder, echem_folder_path or echem_folder_name)

        return prepare_for_bokeh(nmr_data, df, merged_df, num_experiments)

    except Exception as e:
        raise RuntimeError(f"Error in common processing: {str(e)}")
