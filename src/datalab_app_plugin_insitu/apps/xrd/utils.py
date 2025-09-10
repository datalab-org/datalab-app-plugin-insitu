import re
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydatalab.logger import LOGGER
from scipy.interpolate import interp1d

from datalab_app_plugin_insitu.echem_utils import process_echem_data
from datalab_app_plugin_insitu.utils import _find_folder_path


def process_local_xrd_data(
    file_path: str | Path,
    xrd_folder_name: Path,
    log_folder_name: Path,
    start_exp: int = 1,
    exclude_exp: Union[list, None] = None,
    time_series_source: str = "log",
    echem_folder_name: Optional[Path] = None,
):
    """
    Process local XRD data from a zip file.

    Parameters:
        file_path: Path to the zip file containing XRD data.
        xrd_folder_name: Path to the folder containing XRD data.
        log_folder_name: Path to the folder containing log data.
        start_exp: Starting experiment number.
        exclude_exp: List of experiments to exclude.
        time_series_source: Source of time series data, either 'log' or 'echem' to select whether temperature or echem is the time series data shown.
        echem_folder_name: Optional path to the folder containing echem data. Only used if time_series_source is 'echem'.

    Returns:
        dict: Processed XRD data and metadata.
    """

    # Check if the folder exists
    if not all([xrd_folder_name, log_folder_name]):
        raise ValueError("Both XRD and log folders must be specified.")

    if time_series_source == "echem":
        if not echem_folder_name:
            raise ValueError(
                "Echem folder name must be specified when using echem as time series source."
            )

    if isinstance(file_path, str):
        file_path = Path(file_path)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            if file_path.suffix == ".zip":
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    members = [
                        m for m in zip_ref.namelist() if not ("__MACOSX" in m or m.startswith("."))
                    ]
                    for member in members:
                        try:
                            zip_ref.extract(member, tmpdir)
                        except Exception as e:
                            raise RuntimeError(f"Failed to extract {member}: {str(e)}")
                base_path = Path(tmpdir)
            else:
                base_path = Path(file_path)

            # Find the relative paths to the XRD and reference folders
            xrd_path = _find_folder_path(base_path, xrd_folder_name)
            log_path = _find_folder_path(base_path, log_folder_name)

            # Check if the folder exists
            if not all([xrd_path, log_path]):
                raise ValueError("XRD folder and log folder must be specified.")

            assert isinstance(xrd_path, Path), "xrd_path must be a Path object"
            assert isinstance(log_path, Path), "log_path must be a Path object"

            # If echem mode perform the same checks for folder existing etc.
            if time_series_source == "echem":
                if not echem_folder_name:
                    raise ValueError(
                        "Echem folder name must be specified when using echem as time series source."
                    )
                echem_path = _find_folder_path(base_path, echem_folder_name)
                assert isinstance(echem_path, Path), "echem_path must be a Path object"
                if not echem_path.exists():
                    raise FileNotFoundError(f"Echem folder not found: {echem_path}")

            # Load the XRD data
            xrd_data = process_xrd_data(
                xrd_folder=xrd_path,
                start_at=start_exp,
                exclude_exp=exclude_exp,
                glob_str="*summed*",
            )

            # Load the 1D data
            if log_path.exists():
                log_files = list(log_path.glob("*.csv"))
                if len(log_files) > 1:
                    raise ValueError(
                        f"Log folder should contain exactly one CSV file: {log_path}. Found {len(log_files)} files. Files found: {log_files}"
                    )
                    # TODO handle multiple files
                elif len(log_files) == 0:
                    raise ValueError(f"Log folder should contain at least one CSV file: {log_path}")
                else:
                    log_file = log_files[0]

            else:
                raise FileNotFoundError(f"No log files found with extension .csv in {log_path}")

            try:
                if time_series_source == "echem":
                    log_data = load_echem_log_file(log_file)
                elif time_series_source == "log":
                    log_data = load_temperature_log_file(log_file)
                else:
                    raise ValueError(f"Unknown time_series_source: {time_series_source}")

            except Exception as e:
                raise RuntimeError(f"Failed to load log file: {str(e)}")

            # TODO alternate pathway for echem data
            # Check that the log scans are the same as the xrd scans
            log_scan_numbers = set(log_data["scan_number"].astype(int))
            xrd_scan_numbers = set(xrd_data["file_num_index"][:, 0].astype(int))
            if not log_scan_numbers.issubset(xrd_scan_numbers):
                raise ValueError(
                    "Log file scan numbers do not match XRD data scan numbers. "
                    "Ensure the log file contains all XRD scans."
                )

            if time_series_source == "log":
                # Add the log data to the xrd_data dictionary
                log_data = log_data.rename(columns={"Temp": "x", "scan_number": "y"})

                time_series_data_dict = {
                    "x": log_data["x"].values,
                    "y": log_data["y"].values,
                    "metadata": {
                        "min_y": log_data["y"].min(),
                        "max_y": log_data["y"].max(),
                    },
                }
                xrd_data["Time_series_data"] = time_series_data_dict

                # Create explicit link between file num, temperature and experiment number
                index_df = pd.DataFrame.from_dict(
                    {
                        "file_num": xrd_data["Time_series_data"]["y"],
                        "exp_num": np.arange(1, xrd_data["metadata"]["num_experiments"] + 1),
                        "Temperature": xrd_data["Time_series_data"]["x"],
                    }
                )
                index_df.index.name = "index"

                xrd_data["index_df"] = index_df

                return xrd_data

            elif time_series_source == "echem":
                if not isinstance(echem_path, Path):
                    raise ValueError("Echem path must be a Path object.")
                echem_data = process_echem_data(echem_path)
                xrd_data["Time_series_data"] = echem_data
                xrd_data["log data"] = log_data

                df_echem = echem_data["data"]
                df_echem["Timestamp"] = pd.to_datetime(df_echem["Timestamp"])
                df_echem = df_echem.rename(columns={"Timestamp": "echem_timestamp"})
                time_deltas = df_echem.echem_timestamp - df_echem.echem_timestamp.iloc[0]
                df_echem["elapsed_time_hours"] = [
                    delta.total_seconds() / 3600 for delta in time_deltas
                ]
                df_echem["elapsed_time_seconds"] = [delta.total_seconds() for delta in time_deltas]
                log_data.rename(columns={"start_time": "xrd_timestamp"}, inplace=True)
                log_data["xrd_timestamp"] = pd.to_datetime(log_data["xrd_timestamp"])
                df_merged = pd.merge_asof(
                    log_data,
                    df_echem,
                    left_on="xrd_timestamp",
                    right_on="echem_timestamp",
                    direction="nearest",
                )

                df_merged["exp_num"] = np.arange(1, len(df_merged) + 1)
                df_merged = df_merged.rename(
                    columns={
                        "scan_number": "file_num",
                        "Voltage": "voltage",
                        "elapsed_time_seconds": "time",
                    }
                )
                xrd_data["index_df"] = df_merged

                return xrd_data

    except Exception as e:
        raise RuntimeError(f"Failed to process XRD data: {str(e)}")


def process_xrd_data(
    xrd_folder: Path,
    start_at: int = 1,
    exclude_exp: Optional[List[int]] = None,
    glob_str: str = "*summed*",
) -> Dict:
    """
    Process XRD data from a specified folder.

    Args:
        xrd_folder: Path to the folder containing XRD data.
        log_file: Path to the log file for storing processing results.
        start_at: Starting experiment number.
        exclude_exp: List of experiments to exclude.

    Returns:
        Dict: Processed XRD data and metadata.
    """
    from pydatalab.apps.xrd.blocks import XRDBlock

    if not xrd_folder.exists():
        raise FileNotFoundError(f"XRD folder does not exist: {xrd_folder}")

    file_list = list(xrd_folder.glob(glob_str))
    # Process the first XRD pattern file
    if not file_list:
        raise FileNotFoundError(f"No XRD files found in {xrd_folder} with pattern {glob_str}")

    first_file = XRDBlock.load_pattern(file_list[0])
    two_theta = first_file[0]["2θ (°)"].values

    # Initialize a DataFrame to store all patterns
    all_patterns = pd.DataFrame(index=file_list, columns=two_theta)

    for file in file_list:
        try:
            pattern = XRDBlock.load_pattern(file)
            if pattern is not None:
                # Some files seem to be missing one or two two theta values - this will raise a warning when this happens but deal with the missing data in a reasonable fashion by interpolating and putting zeros for when the range is out of bounds
                intensity_values = pattern[0]["intensity"].values
                new_two_theta_values = pattern[0]["2θ (°)"].values
                if set(new_two_theta_values) != set(two_theta):
                    missing_values = set(two_theta) - set(new_two_theta_values)
                    LOGGER.warning(
                        f"Inconsistent 2θ values found in file {file}: {missing_values}."
                    )
                interpolater = interp1d(
                    new_two_theta_values, intensity_values, bounds_error=False, fill_value="zero"
                )
                all_patterns.loc[file, two_theta] = interpolater(two_theta)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    # Sort the dataframe based on the scan_number extracted from the filename of the format: 1058063-mythen_summed.dat
    # TODO make this more robust to different file naming conventions
    def extract_number(filename, pattern=r"(?<!\d)(\d{6,8})(?!\d)"):
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
        return None

    filename_pattern = r"(?<!\d)(\d{6,8})(?!\d)"
    all_patterns.index = all_patterns.index.map(lambda x: extract_number(x.name, filename_pattern))
    all_patterns.sort_index(inplace=True)

    file_num_index = all_patterns.index.values.reshape(-1, 1)

    metadata = {
        "num_experiments": len(all_patterns.index),
        "y_range": {
            "max_y": max(all_patterns.index),
            "min_y": min(all_patterns.index),
        },
    }
    return {
        "2D_data": all_patterns,
        "Two theta": two_theta,
        "metadata": metadata,
        "file_num_index": file_num_index,
    }


def load_temperature_log_file(log_file: Path) -> pd.DataFrame:
    """
    Load temperature log file and return as a DataFrame. This currently assumes the Temperature is recorded in Celsius.

    Args:
        log_file (Path): Path to the temperature log file, must contain scan_number and Temp as column headers.

    Returns:
        pd.DataFrame: DataFrame containing the temperature log data.
    """
    if not log_file.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_file}")

    log_df = pd.read_csv(log_file)
    if "scan_number" not in log_df.columns:
        raise ValueError("Log file must contain a 'scan_number' column.")

    if "Temp" not in log_df.columns:
        raise ValueError("Log file must contain a 'Temp' column.")

    return log_df


def load_echem_log_file(log_file: Path) -> pd.DataFrame:
    """
    Load electrochemical log file and return as a DataFrame.

    Args:
        log_file (Path): Path to the electrochemical log file, must contain scan_number, start_time and end_time as column headers.

    Returns:
        pd.DataFrame: DataFrame containing the electrochemical log data.
    """
    if not log_file.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_file}")

    log_df = pd.read_csv(log_file)
    if "scan_number" not in log_df.columns:
        raise ValueError("Log file must contain a 'scan_number' column.")

    if "start_time" not in log_df.columns or "end_time" not in log_df.columns:
        raise ValueError("Log file must contain 'start_time' and 'end_time' columns.")

    return log_df
