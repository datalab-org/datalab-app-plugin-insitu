import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from datalab_app_plugin_insitu.echem_utils import process_echem_data
from datalab_app_plugin_insitu.utils import _find_folder_path


def find_scan_time(filename: Path) -> float:
    """
    Note: Not currently used as the scan time is passed as a parameter to the block due to no way to calculate the rest time from the files.
    Finds the scan time from the UV-Vis .Raw8.txt file. Doesn't take into account the time between scans, just the time it takes to scan the sample.
    This is calculated as the integration time multiplied by the number of scans, divided by 1000 to convert milliseconds to seconds.
    This is useful for plotting the UV-Vis data against time
    Args:
        filename (Path): Path to the .Raw8.txt file
    Returns:
        float: Scan time in seconds
    """
    # Initialize variables
    integration_time = None
    num_scans = None
    with open(filename) as file:
        lines = file.readlines()
        while num_scans is None or integration_time is None:
            for line in lines[:20]:
                if "Integration time [ms]:" in line:
                    integration_time = float(line.split(":")[1].strip())
                if "Averaging Nr. [scans]:" in line:
                    num_scans = int(line.split(":")[1].strip())
            if integration_time is None or num_scans is None:
                raise ValueError(
                    f"Could not find integration time or number of scans in file: {filename}"
                )
        scan_time = integration_time * num_scans / 1000.0  # Convert ms to seconds

    return scan_time


def parse_uvvis_txt(filename: Path) -> pd.DataFrame:
    """
    Parses a UV-Vis .txt file into a pandas DataFrame
    Args:
        filename (Path): Path to the .txt file
    Returns:
        pd.DataFrame: DataFrame containing the UV-Vis data with columns for wavelength and absorbance
    """
    # Read the file, skipping the first 7 rows and using the first row as header
    data = pd.read_csv(filename, sep=r";", skiprows=7, header=None)

    # @be-smith: I need to look into what dark counts and reference counts are - I never used them just the sample counts from two different runs
    data.columns = ["Wavelength", "Sample counts", "Dark counts", "Reference counts"]
    return data


def find_absorbance(data_df, reference_df):
    """
    Calculates the absorbance from the sample and reference dataframes
    Args:
        data_df (pd.DataFrame): DataFrame containing the sample data
        reference_df (pd.DataFrame): DataFrame containing the reference data
    Returns:
        pd.DataFrame: DataFrame containing the absorbance data
    """
    # Calculate absorbance using Beer-Lambert Law
    absorbance = -np.log10(data_df["Sample counts"] / reference_df["Sample counts"])
    # Create a new DataFrame with the wavelength and absorbance
    absorbance_data = pd.DataFrame({"Wavelength": data_df["Wavelength"], "Absorbance": absorbance})
    return absorbance_data


def process_local_uvvis_data(
    folder_name: Path,
    uvvis_folder: Path,
    reference_folder: Path,
    echem_folder: Path,
    start_at: int = 0,
    sample_file_extension: str = ".Raw8.txt",
    reference_file_extension: str = ".Raw8.TXT",
    exclude_exp: Optional[List[int]] = None,
    scan_time: Optional[float] = None,
) -> Dict:
    """
    Processes UV-Vis and Echem data from a local folder or a zip file. This function basicallly wraps the process_uvvis_data and process_echem_data functions to handle both types of data.
    It checks if the folder is a zip file and extracts it to a temporary directory if so.
    It then finds the relative paths to the UV-Vis and reference folders, processes the UV-Vis data, and processes the Echem data.
    Finally, it combines the UV-Vis data and Echem data into a single dictionary and returns it.

    Args:
        folder_name: Path to the folder or zip file containing the data
        uvvis_folder: Folder name containing the UV-Vis data files
        reference_folder: Folder name containing the reference data file
        echem_folder: Folder name containing the Echem data files
        start_at: Index to start processing from
        sample_file_extension: File extension for sample files
        reference_file_extension: File extension for reference files
        exclude_exp: List of indices to exclude from processing
        scan_time: Time taken for the scan in seconds

    Returns:
        Dictionary containing the following keys, the processed UV-Vis data [2D data] and Echem data [Time_series_data], along with wavelength, metadata, time of scan, and file number index.

    Raises:
        ValueError: If the UV-Vis or reference folders are not specified or do not exist
        FileNotFoundError: If the UV-Vis or reference folders are not found in the provided path
        RuntimeError: If there is an error extracting the zip file or finding the folder
    """
    # Check if the folder exists
    if not all([uvvis_folder, reference_folder]):
        raise ValueError("Both UV-Vis and reference folders must be specified.")
    # Wrap everything in a temporary directory if the folder is a zip file
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            if folder_name.suffix == ".zip":
                with zipfile.ZipFile(folder_name, "r") as zip_ref:
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
                base_path = Path(folder_name)

            # Find the relative paths to the UV-Vis and reference folders from the temporary directory
            uvvis_path = _find_folder_path(base_path, uvvis_folder)
            reference_path = _find_folder_path(base_path, reference_folder)
            echem_path = _find_folder_path(base_path, echem_folder)
            # Check there is one file in the reference folder with the right extension - if so parse it for the reference scan
            if reference_path is None or not reference_path.is_dir():
                raise FileNotFoundError(f"Reference folder not found: {reference_folder}")
            if uvvis_path is None or not uvvis_path.is_dir():
                raise FileNotFoundError(f"UV-Vis folder not found: {uvvis_folder}")
            if echem_path is None or not echem_path.is_dir():
                raise FileNotFoundError(f"Echem folder not found: {echem_folder}")

            # Process the UV-Vis data
            uvvis_data = process_uvvis_data(
                uvvis_path,
                reference_path,
                start_at,
                sample_file_extension,
                reference_file_extension,
                exclude_exp,
                scan_time,
            )

            # Process the Echem data
            echem_data = process_echem_data(echem_path)
            # Combine the UV-Vis and Echem data into a single dictionary
            uvvis_data["Time_series_data"] = echem_data

            return uvvis_data

    except Exception as e:
        raise RuntimeError(f"Error Unzipping and finding filepaths to data: {str(e)}")


def process_uvvis_data(
    uvvis_folder: Path,
    reference_folder: Path,
    start_at: int = 0,
    sample_file_extension: str = ".Raw8.txt",
    reference_file_extension: str = ".Raw8.TXT",
    exclude_exp: Optional[List[int]] = None,
    scan_time: Optional[float] = None,
) -> Dict:
    """
    Processes UV-Vis data from specified folders.

    Args:
        uvvis_folder: Path to the folder containing UV-Vis data files
        reference_folder: Path to the folder containing the reference data file
        echem_folder: Path to the folder containing Echem data files
        start_at: Index to start processing from
        sample_file_extension: File extension for sample files
        reference_file_extension: File extension for reference files
        exclude_exp: List of indices to exclude from processing
        scan_time: Time taken for the scan in seconds, including the time between scans. If None the time will be set to the index of the scan.

    Returns:
        Dictionary containing processed UV-Vis data, wavelength, metadata, time of scan, and file number index

    """

    reference_files = list(reference_folder.glob("*" + reference_file_extension))
    if len(reference_files) != 1:
        raise ValueError(
            f"Reference folder should contain exactly one {reference_file_extension} file: {reference_folder}"
        )

    reference_file = reference_files[0]
    reference_df = parse_uvvis_txt(reference_file)
    wavelength = reference_df["Wavelength"].values

    # Calculate absorbance for all the sample files
    # Grab all the files in the uvvis folder with the right extension
    all_files = list(uvvis_folder.glob("*" + sample_file_extension))

    # Grab file numbers for sorting - this might need to be made more flexible - currently assumes numbers are at the end of the filename before the extension
    def num_finder(x):
        filename = x.name
        return int(filename.split(".")[0].split("_")[1])

    file_num = [num_finder(x) for x in all_files]
    sort_df = pd.Series(index=file_num, data=list(all_files))
    sort_df.sort_index(inplace=True)
    # Populate X (2D array for the heatmap) with the patterns - normalising to the original scan
    X = pd.DataFrame(index=sort_df.index, columns=wavelength)

    for idx in X.index:
        file = sort_df.loc[idx]
        # Read the file
        df = parse_uvvis_txt(uvvis_folder / file)
        absorbance = find_absorbance(df, reference_df)["Absorbance"].values
        X.loc[idx, X.columns] = absorbance
        if min(absorbance) < -0.1:
            print(
                f"Warning: Negative absorbance values found in file {idx}. This may indicate an issue with the data."
            )

    if exclude_exp is not None:
        if not isinstance(exclude_exp, Iterable):
            raise ValueError("exclude_exp should be an iterable of indices to exclude.")
        X = X.drop(index=exclude_exp)

    # Remove rows before the start_at index
    if start_at > 0:
        mask = X.index >= start_at
        X = X[mask]

    # Save the index for later use - reshape so it's a column vector
    file_num_index = pd.Series(X.index.copy()).values.reshape(-1, 1)

    # Sort out timestamps - this will make the index the time the scan finishes - maybe discuss
    if scan_time is None:
        X.index = range(len(X.index))
    else:
        pass
    if scan_time is not None:
        X.index = X.index.astype(float) * scan_time

    metadata = {
        "time_range": {"min_time": min(X.index), "max_time": max(X.index)},
        "num_experiments": len(X.index),
    }
    return {
        "2D_data": X,
        "wavelength": wavelength,
        "metadata": metadata,
        "time_of_scan": X.index,
        "file_num_index": file_num_index,
    }
