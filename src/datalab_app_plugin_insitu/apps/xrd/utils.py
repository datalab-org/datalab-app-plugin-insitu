import re
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from datalab_app_plugin_insitu.echem_utils import process_echem_data
from datalab_app_plugin_insitu.utils import (
    _find_folder_path,
    flexible_data_reader,
    should_skip_path,
)


def process_local_xrd_data(
    file_path: str | Path,
    xrd_folder_name: Path,
    log_folder_name: Path,
    start_exp: int = 1,
    exclude_exp: Union[list, None] = None,
    time_series_source: str = "log",
    echem_folder_name: Optional[Path] = None,
    glob_str: Optional[str] = None,
    scan_number_pattern: Optional[str] = None,
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
        glob_str: Optional glob pattern to match XRD files (e.g., "*summed*"). If None, all files in the folder are used.
        scan_number_pattern: Optional regular expression used to extract the scan number from
            each XRD filename. If None, the scan number is detected automatically.

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

            if not isinstance(xrd_path, Path):
                raise RuntimeError("xrd_path must be a Path object")

            if not isinstance(log_path, Path):
                raise RuntimeError("log_path must be a Path object")

            # If echem mode perform the same checks for folder existing etc.
            if time_series_source == "echem":
                if not echem_folder_name:
                    raise ValueError(
                        "Echem folder name must be specified when using echem as time series source."
                    )
                echem_path = _find_folder_path(base_path, echem_folder_name)

                if not isinstance(echem_path, Path):
                    raise RuntimeError("echem_path must be a Path object")

                if not echem_path.exists():
                    raise FileNotFoundError(f"Echem folder not found: {echem_path}")

            # Load the XRD data
            xrd_data = process_xrd_data(
                xrd_folder=xrd_path,
                start_at=start_exp,
                exclude_exp=exclude_exp,
                glob_str=glob_str,
                scan_number_pattern=scan_number_pattern,
            )

            # Load the 1D data
            if log_path.exists():
                # Look for both CSV and text files
                csv_files = list(log_path.glob("*.csv"))
                txt_files = list(log_path.glob("*.txt"))
                log_files = csv_files + txt_files

                if len(log_files) > 1:
                    raise ValueError(
                        f"Log folder should contain exactly one data file: {log_path}. Found {len(log_files)} files. Files found: {log_files}"
                    )
                    # TODO handle multiple files
                elif len(log_files) == 0:
                    raise ValueError(
                        f"Log folder should contain at least one CSV or TXT file: {log_path}"
                    )
                else:
                    log_file = log_files[0]

            else:
                raise FileNotFoundError(
                    f"No log files found with extension .csv or .txt in {log_path}"
                )

            try:
                if time_series_source == "echem":
                    log_data = load_echem_log_file(log_file)
                elif time_series_source == "log":
                    log_data = load_temperature_log_file(log_file)
                else:
                    raise ValueError(f"Unknown time_series_source: {time_series_source}")

            except Exception as e:
                raise RuntimeError(f"Failed to load log file: {str(e)}")

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

                df_echem = xrd_data["Time_series_data"]["data"]

                log_data.rename(columns={"start_time": "xrd_timestamp"}, inplace=True)
                log_data["xrd_timestamp"] = pd.to_datetime(log_data["xrd_timestamp"])

                # Note: Timestamp column is already standardized in process_echem_data()
                # Note: elapsed_time_seconds and elapsed_time_hours are already calculated in process_echem_data()
                if "Timestamp" not in df_echem.columns:
                    warnings.warn(
                        "No absolute timestamp found in echem data, aligning start of echem to start of XRD."
                    )
                    echem_start_time = log_data["xrd_timestamp"].min()
                    df_echem["Timestamp"] = echem_start_time + pd.to_timedelta(
                        df_echem["time/s"], unit="s"
                    )

                df_echem["Timestamp"] = pd.to_datetime(df_echem["Timestamp"])
                df_echem = df_echem.rename(columns={"Timestamp": "echem_timestamp"})

                df_merged = pd.merge_asof(
                    log_data,
                    df_echem,
                    left_on="xrd_timestamp",
                    right_on="echem_timestamp",
                    direction="nearest",
                )

                # Adding scan_number to the echem data to be used for the legend later.
                # Note: Timestamp column is already standardized to "Timestamp" in process_echem_data()
                echem_merged = pd.merge_asof(
                    xrd_data["Time_series_data"]["data"],
                    xrd_data["log data"][["xrd_timestamp", "scan_number"]],
                    left_on="Timestamp",
                    right_on="xrd_timestamp",
                    direction="nearest",
                )

                xrd_data["Time_series_data"]["scan_number"] = echem_merged["scan_number"].values
                df_merged["exp_num"] = np.arange(1, len(df_merged) + 1)
                df_merged = df_merged.rename(
                    columns={
                        "scan_number": "file_num",
                        "Voltage": "voltage",
                        "elapsed_time_seconds": "time",
                    }
                )

                # Remove 'index' column if it exists in the data to avoid conflicts with pandas reset_index()
                if "index" in df_merged.columns:
                    df_merged = df_merged.drop(columns=["index"])

                xrd_data["index_df"] = df_merged

                # Create a mapping from file_num to exp_num
                file_num_to_exp_num = dict(
                    zip(xrd_data["index_df"]["file_num"], xrd_data["index_df"]["exp_num"])
                )

                # Map scan_number to exp_num using the dictionary
                echem_merged["exp_num"] = echem_merged["scan_number"].map(file_num_to_exp_num)
                xrd_data["Time_series_data"]["exp_num"] = echem_merged["exp_num"].values

                xrd_data["Time_series_data"]["data"] = echem_merged

                return xrd_data

    except Exception as e:
        raise RuntimeError(f"Failed to process XRD data: {str(e)}")


def process_xrd_data(
    xrd_folder: Path,
    start_at: int = 1,
    exclude_exp: Optional[List[int]] = None,
    glob_str: Optional[str] = None,
    scan_number_pattern: Optional[str] = None,
) -> Dict:
    """
    Process XRD data from a specified folder.

    Args:
        xrd_folder: Path to the folder containing XRD data.
        start_at: Starting experiment number.
        exclude_exp: List of experiments to exclude.
        glob_str: Optional glob pattern to match files (e.g., "*summed*", "*.xy").
                 If None, all files in the folder are processed.
        scan_number_pattern: Optional regular expression used to extract the scan number
            from each filename (see `extract_scan_numbers`). If None, the scan number is
            detected automatically.

    Returns:
        Dict: Processed XRD data and metadata.
    """
    from pydatalab.apps.xrd.blocks import XRDBlock

    if not xrd_folder.exists():
        raise FileNotFoundError(f"XRD folder does not exist: {xrd_folder}")

    # Get list of files based on glob_str
    if glob_str is None:
        # Select all files in the folder (excluding directories and system files)
        file_list = [f for f in xrd_folder.iterdir() if f.is_file() and not should_skip_path(f)]
    else:
        file_list = [f for f in xrd_folder.glob(glob_str) if not should_skip_path(f)]

    # Process the first XRD pattern file
    if not file_list:
        pattern_msg = f"with pattern '{glob_str}'" if glob_str else ""
        raise FileNotFoundError(f"No XRD files found in {xrd_folder} {pattern_msg}".strip())

    first_file = XRDBlock.load_pattern(file_list[0])
    two_theta = first_file[0]["2θ (°)"].values

    # Initialize a DataFrame to store all patterns
    all_patterns = pd.DataFrame(index=file_list, columns=two_theta)

    for file in file_list:
        pattern = XRDBlock.load_pattern(file)
        if pattern is not None:
            # Some files seem to be missing one or two two theta values - this will raise a warning when this happens but deal with the missing data in a reasonable fashion by interpolating and putting zeros for when the range is out of bounds
            intensity_values = pattern[0]["intensity"].values
            new_two_theta_values = pattern[0]["2θ (°)"].values
            if set(new_two_theta_values) != set(two_theta):
                warnings.warn(
                    f"Inconsistent 2θ values found in file {file}; interpolated missing values."
                )
                interpolator = interp1d(
                    new_two_theta_values, intensity_values, bounds_error=False, fill_value=0
                )
                all_patterns.loc[file, two_theta] = interpolator(two_theta)
            else:
                all_patterns.loc[file, two_theta] = intensity_values

    # Sort the dataframe based on the scan number extracted from each filename
    all_patterns.index = pd.Index(
        extract_scan_numbers([f.name for f in all_patterns.index], pattern=scan_number_pattern)
    )
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


def extract_scan_numbers(filenames: List[str], pattern: Optional[str] = None) -> List[int]:
    """Extract the scan number from each filename in a set of in situ XRD scans.

    If a `pattern` is provided, it is used as a regular expression searched against each
    filename; the first capturing group (or the whole match, if there are no groups) must
    contain the scan number.

    Otherwise, the scan number is detected automatically from the runs of digits in each
    filename (with any extension removed), so that any zero-padding (or none) is supported,
    e.g., `scan_1.xy`, `0001-scan.dat`, `1058063-mythen_summed.dat` or
    `1355639_summed_mythen3.xye`:

    - If every filename has the same number of digit runs, the run (by position) whose
      values are unique across all files is used, ignoring runs that are constant
      (e.g., the `3` in `mythen3`) or repeated (e.g., a date).
      If several positions are unique, the one with the most digits is used, preferring
      the last in the filename in case of a tie.
    - Otherwise, the longest run of digits in each filename is used, preferring the last
      in the filename in case of a tie.

    Parameters:
        filenames: The filenames (not full paths) of the scans.
        pattern: Optional regular expression to extract the scan number.

    Returns:
        The scan number of each file, in the same order as `filenames`.

    Raises:
        ValueError: If a scan number cannot be found for every file, or if the
            extracted scan numbers are not unique.

    """
    if not filenames:
        return []

    if pattern is not None:
        regex = re.compile(pattern)
        scan_numbers = []
        for filename in filenames:
            match = regex.search(filename)
            if not match:
                raise ValueError(
                    f"Could not find a scan number matching pattern {pattern!r} in filename {filename!r}"
                )
            try:
                scan_numbers.append(int(match.group(1) if regex.groups else match.group(0)))
            except ValueError:
                raise ValueError(
                    f"Scan number pattern {pattern!r} matched non-integer text in filename {filename!r}"
                )

    else:
        digit_runs = [re.findall(r"\d+", Path(filename).stem) for filename in filenames]
        missing = [f for f, runs in zip(filenames, digit_runs) if not runs]
        if missing:
            raise ValueError(
                f"Could not find a scan number in the following filename(s): {missing[:5]}"
                + (f" (and {len(missing) - 5} more)" if len(missing) > 5 else "")
            )

        def _longest_last(runs: List[str]) -> str:
            # max returns the first of equal-length runs, so search the reversed list
            return max(reversed(runs), key=len)

        position = None
        num_runs = {len(runs) for runs in digit_runs}
        if len(num_runs) == 1:
            n = num_runs.pop()
            unique_positions = [
                i for i in range(n) if len({int(runs[i]) for runs in digit_runs}) == len(filenames)
            ]
            if unique_positions:
                position = max(
                    reversed(unique_positions),
                    key=lambda i: max(len(runs[i]) for runs in digit_runs),
                )

        if position is not None:
            scan_numbers = [int(runs[position]) for runs in digit_runs]
        else:
            scan_numbers = [int(_longest_last(runs)) for runs in digit_runs]

    if len(set(scan_numbers)) != len(scan_numbers):
        seen: Dict[int, str] = {}
        duplicates = []
        for filename, number in zip(filenames, scan_numbers):
            if number in seen:
                duplicates.append((seen[number], filename, number))
            else:
                seen[number] = filename
        raise ValueError(
            "Scan numbers extracted from the XRD filenames are not unique, e.g., "
            + ", ".join(f"{a!r} and {b!r} (scan {n})" for a, b, n in duplicates[:3])
            + ". Use a more specific glob pattern to select the files, or provide a scan number pattern."
        )

    return scan_numbers


def load_temperature_log_file(log_file: Path) -> pd.DataFrame:
    """
    Load temperature log file and return as a DataFrame. This currently assumes the Temperature is recorded in Celsius.

    Supports CSV, TXT, and Excel files with automatic delimiter detection.

    Args:
        log_file (Path): Path to the temperature log file, must contain scan_number and Temp as column headers.

    Returns:
        pd.DataFrame: DataFrame containing the temperature log data.

    Raises:
        FileNotFoundError: If the log file does not exist.
        ValueError: If the file cannot be parsed or required columns are missing.
    """
    log_df = flexible_data_reader(log_file, required_columns=["scan_number", "Temp"])

    return log_df


def load_echem_log_file(log_file: Path) -> pd.DataFrame:
    """
    Load electrochemical log file and return as a DataFrame.

    Supports CSV, TXT, and Excel files with automatic delimiter detection.

    Args:
        log_file (Path): Path to the electrochemical log file, must contain scan_number, start_time and end_time as column headers.

    Returns:
        pd.DataFrame: DataFrame containing the electrochemical log data.

    Raises:
        FileNotFoundError: If the log file does not exist.
        ValueError: If the file cannot be parsed or required columns are missing.
    """
    log_df = flexible_data_reader(
        log_file, required_columns=["scan_number", "start_time", "end_time"]
    )

    return log_df
