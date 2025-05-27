import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import nmrglue as ng
import numpy as np
import pandas as pd
from navani import echem as ec


def should_skip_path(path: Union[str, Path]) -> bool:
    """
    Check if a path should be skipped (macOS system files, hidden files, etc.)

    Args:
        path: Path or string to check

    Returns:
        bool: True if the path should be skipped, False otherwise
    """
    path_str = str(path)
    return "__MACOSX" in path_str or Path(path_str).name.startswith(".")


def _find_folder_path(base_path: Path, target_folder_name: str) -> Optional[Path]:
    """
    Find a folder path inside a zip regardless of whether the zip has an extra level of nesting.

    Args:
        base_path: Base directory to start the search
        target_folder_name: Name of the folder to find

    Returns:
        Optional[Path]: Path to the found folder, or None if not found
    """
    target_folder_name = Path(target_folder_name).stem

    direct_path = base_path / target_folder_name
    if direct_path.exists() and direct_path.is_dir():
        return direct_path

    for item in base_path.iterdir():
        if should_skip_path(item):
            continue

        if item.is_dir():
            nested_path = item / target_folder_name
            if nested_path.exists() and nested_path.is_dir():
                return nested_path

    for root, dirs, _ in os.walk(str(base_path)):
        dirs[:] = [d for d in dirs if not should_skip_path(Path(d))]

        if target_folder_name in dirs:
            return Path(root) / target_folder_name

    return None


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


def setup_paths(
    nmr_folder_path: Path, start_at: int, exclude_exp: Optional[List[int]]
) -> Tuple[List[str], List[str]]:
    """Setup experiment paths and create output directory."""
    exp_folders = [d for d in Path(nmr_folder_path).iterdir() if d.is_dir() and d.name.isdigit()]

    exp_folder = [
        exp for exp in range(start_at, len(exp_folders) + 1) if exp not in (exclude_exp or [])
    ]

    spec_paths = [
        str(Path(nmr_folder_path) / str(exp) / "pdata" / "1" / "ascii-spec.txt")
        for exp in exp_folder
    ]
    acqu_paths = [str(Path(nmr_folder_path) / str(exp) / "acqus") for exp in exp_folder]

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


def process_spectral_data(
    spec_paths: List[str], time_points: List[float]
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

    intensities = calculate_intensities(nmr_data)

    df = pd.DataFrame(
        {
            "time": time_points,
            "intensity": intensities,
            "norm_intensity": intensities / np.max(intensities),
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
    nmr_data: pd.DataFrame, df: pd.DataFrame, echem_df: Optional[pd.DataFrame], num_experiments: int
) -> Dict:
    """Prepare data for Bokeh visualization, with optional echem data."""

    if nmr_data is None or df is None:
        raise ValueError("Required NMR data or integrated data is None")

    result = {
        "metadata": {
            "time_range": {"start": df["time"].min(), "end": df["time"].max()},
            "num_experiments": num_experiments,
        },
        "nmr_spectra": {
            "ppm": nmr_data["ppm"].tolist(),
            "spectra": [
                {
                    "time": float(df["time"][i]),
                    "intensity": nmr_data[str(i + 1)].tolist(),
                    "experiment_number": i + 1,
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
    exclude_exp: Optional[List[int]],
) -> Dict:
    """
    Common processing logic for both local and Datalab data.

    Args:
        base_folder: Path to the base folder containing all data
        nmr_folder_path: Path to the NMR data folder
        echem_folder_name: Name of the electrochemistry folder
        start_at: Starting experiment number
        exclude_exp: List of experiment numbers to exclude

    Returns:
        Dictionary containing processed NMR and electrochemical data
    """
    try:
        nmr_dimension = check_nmr_dimension(nmr_folder_path)

        if nmr_dimension == "1D":
            spec_paths, acqu_paths = setup_paths(nmr_folder_path, start_at, exclude_exp)
            time_points = process_time_data(acqu_paths)
            nmr_data, df, num_experiments = process_spectral_data(spec_paths, time_points)

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

        result = prepare_for_bokeh(nmr_data, df, merged_df, num_experiments)

        if result is None:
            raise RuntimeError("prepare_for_bokeh returned None instead of expected data")

        return result

    except Exception as e:
        raise RuntimeError(f"Error in common processing: {str(e)}")


def get_cache_path(key: str) -> str:
    """
    Get the path to the cache file for a given key.
    If key is a composite key (e.g. "file_id_subfolder1_subfolder2"),
    it extracts the file_id and uses it to determine the cache location.
    Uses the parent directory of the original file location for cache storage.
    """

    file_id = key.split("_")[0] if "_" in key else key

    try:
        from pydatalab.file_utils import get_file_info_by_id

        file_info = get_file_info_by_id(file_id)
        if file_info and "location" in file_info:
            file_location = Path(file_info["location"])
            cache_dir = file_location.parent / ".datalab_cache"
        else:
            cache_dir = Path(os.path.expanduser("~")) / ".datalab_cache"
    except Exception:
        cache_dir = Path(os.path.expanduser("~")) / ".datalab_cache"

    os.makedirs(cache_dir, exist_ok=True)
    return str(cache_dir / key)


def save_data_to_cache(file_id: str, data: Dict) -> None:
    """Save processed data to HDF5 cache file."""

    cache_path = f"{get_cache_path(file_id)}_data.h5"

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        with h5py.File(cache_path, "w") as f:
            if "metadata" in data:
                f.attrs["metadata"] = json.dumps(data["metadata"])

            if "nmr_spectra" in data:
                nmr_group = f.create_group("nmr_spectra")
                nmr_group.create_dataset("ppm", data=data["nmr_spectra"]["ppm"])

                spectra_group = nmr_group.create_group("spectra")
                for i, spectrum in enumerate(data["nmr_spectra"]["spectra"]):
                    spectrum_group = spectra_group.create_group(str(i))
                    spectrum_group.attrs["time"] = spectrum["time"]
                    spectrum_group.attrs["experiment_number"] = spectrum.get(
                        "experiment_number", i + 1
                    )
                    spectrum_group.create_dataset("intensity", data=spectrum["intensity"])

            if "integrated_data" in data:
                integrated_group = f.create_group("integrated_data")
                for key, values in data["integrated_data"].items():
                    integrated_group.create_dataset(key, data=values)

            if "echem" in data and data["echem"]:
                echem_group = f.create_group("echem")
                for key, values in data["echem"].items():
                    echem_group.create_dataset(key, data=values)

    except Exception as e:
        raise RuntimeError(f"Error saving to cache: {str(e)}")


def load_data_from_cache(file_id: str) -> Optional[Dict]:
    """Load processed data from HDF5 cache file."""

    cache_path = f"{get_cache_path(file_id)}_data.h5"

    if not os.path.exists(cache_path):
        return None

    try:
        with h5py.File(cache_path, "r") as f:
            data = {}

            if "metadata" in f.attrs:
                data["metadata"] = json.loads(f.attrs["metadata"])

            if "nmr_spectra" in f:
                nmr_group = f["nmr_spectra"]
                data["nmr_spectra"] = {"ppm": nmr_group["ppm"][:].tolist(), "spectra": []}

                for i in range(len(nmr_group["spectra"])):
                    spectrum_group = nmr_group["spectra"][str(i)]
                    data["nmr_spectra"]["spectra"].append(
                        {
                            "time": float(spectrum_group.attrs["time"]),
                            "experiment_number": int(spectrum_group.attrs["experiment_number"]),
                            "intensity": spectrum_group["intensity"][:].tolist(),
                        }
                    )

            if "integrated_data" in f:
                integrated_group = f["integrated_data"]
                data["integrated_data"] = {}
                for key in integrated_group.keys():
                    data["integrated_data"][key] = integrated_group[key][:].tolist()

            if "echem" in f:
                echem_group = f["echem"]
                data["echem"] = {}
                for key in echem_group.keys():
                    data["echem"][key] = echem_group[key][:].tolist()

            return data
    except Exception as e:
        raise RuntimeError(f"Error loading from cache: {str(e)}")


def save_plot_to_cache(cache_key: str, plot_data: Dict) -> None:
    """Save plot data to a separate JSON cache file."""
    plot_cache_path = f"{get_cache_path(cache_key)}_plot.json"

    try:
        os.makedirs(os.path.dirname(plot_cache_path), exist_ok=True)
        with open(plot_cache_path, "w") as f:
            json.dump(plot_data, f)
    except Exception as e:
        raise RuntimeError(f"Error saving plot to cache: {str(e)}")


def load_plot_from_cache(cache_key: str) -> Optional[Dict]:
    """Load plot data from the JSON cache file."""
    plot_cache_path = f"{get_cache_path(cache_key)}_plot.json"

    if not os.path.exists(plot_cache_path):
        return None

    try:
        with open(plot_cache_path) as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading plot from cache: {str(e)}")
