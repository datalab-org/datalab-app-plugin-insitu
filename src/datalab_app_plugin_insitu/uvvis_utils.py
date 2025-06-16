import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from datalab_app_plugin_insitu.echem_utils import process_echem_data
from datalab_app_plugin_insitu.utils import _find_folder_path


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

    # I need to look into what dark counts and reference counts are - I never used them just the sample counts from two differernt runs
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
    start_at: int = 1,
    sample_file_extension: str = ".Raw8.txt",
    reference_file_extension: str = ".Raw8.TXT",
    exclude_exp: Optional[List[int]] = None,
    scan_time: Optional[float] = None,
) -> Dict:
    # Check if the folder exists
    if not all([uvvis_folder, reference_folder]):
        raise ValueError("Both UV-Vis and reference folders must be specified.")

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

            # Find the relative paths to the UV-Vis and reference folders
            uvvis_path = _find_folder_path(base_path, uvvis_folder)
            reference_path = _find_folder_path(base_path, reference_folder)
            echem_path = _find_folder_path(base_path, echem_folder)
            # Check there is one file in the reference folder with the right extension - if so parse it for the reference scan
            if reference_path is None:
                raise FileNotFoundError(f"Reference folder not found: {reference_folder}")
            if not reference_path.exists():
                raise FileNotFoundError(f"Reference folder not found: {reference_path}")
            if not reference_path.is_dir():
                raise ValueError(f"Reference folder is not a directory: {reference_path}")

            # Grab all the files in the uvvis folder with the right extension
            if uvvis_path is None:
                raise FileNotFoundError(f"UV-Vis folder not found: {uvvis_folder}")
            if not uvvis_path.exists():
                raise FileNotFoundError(f"UV-Vis folder not found: {uvvis_path}")
            if not uvvis_path.is_dir():
                raise ValueError(f"UV-Vis folder is not a directory: {uvvis_path}")

            if echem_path is None:
                raise FileNotFoundError(f"Echem folder not found: {echem_folder}")
            if not echem_path.exists():
                raise FileNotFoundError(f"Echem folder not found: {echem_path}")
            if not echem_path.is_dir():
                raise ValueError(f"Echem folder is not a directory: {echem_path}")

            uvvis_data = process_uvvis_data(
                uvvis_path,
                reference_path,
                start_at,
                sample_file_extension,
                reference_file_extension,
                exclude_exp,
                scan_time,
            )

            echem_data = process_echem_data(echem_path)
            # Combine the UV-Vis and Echem data into a single dictionary
            uvvis_data["Time_series_data"] = echem_data

            return uvvis_data

    except Exception as e:
        raise RuntimeError(f"Error Unzipping and finding filepaths to data: {str(e)}")


def process_uvvis_data(
    uvvis_folder: Path,
    reference_folder: Path,
    start_at: int = 1,
    sample_file_extension: str = ".Raw8.txt",
    reference_file_extension: str = ".Raw8.TXT",
    exclude_exp: Optional[List[int]] = None,
    scan_time: Optional[float] = None,
) -> Dict:
    """
    Processes UV-Vis and Echem data from specified folders.
    Args:
        uvvis_folder (Path): Path to the folder containing UV-Vis data files
        reference_folder (Path): Path to the folder containing the reference data file
        echem_folder (Path): Path to the folder containing Echem data files
        start_at (int): Index to start processing from
        sample_file_extension (str): File extension for sample files
        reference_file_extension (str): File extension for reference files
        exclude_exp (Optional[List[int]]): List of indices to exclude from processing
        scan_time (Optional[float]): Time taken for the scan in seconds
    Returns:
        Dict: Dictionary containing two keys, the processed UV-Vis data [2D data] and Echem data [echem data]
    """

    reference_files = list(reference_folder.glob("*" + reference_file_extension))
    if len(reference_files) != 1:
        print(f"Reference files found: {len(reference_files)}")
        print(f"Reference folder: {reference_folder}")
        raise ValueError(
            f"Reference folder should contain exactly one {reference_file_extension} file: {reference_folder}"
        )

    reference_file = reference_files[0]
    reference_df = parse_uvvis_txt(reference_file)
    wavelength = reference_df["Wavelength"].values

    # Calculate absorbance for all the sample files
    # Grab all the files in the uvvis folder with the right extension
    all_files = list(uvvis_folder.glob("*" + sample_file_extension))

    # Grab file numbers for sorting - this might need to be made more flexible - currently assumes numbers are at the end of the filename
    def num_finder(x):
        filename = x.name
        return int(filename.split(".")[0].split("_")[1])

    file_num = [num_finder(x) for x in all_files]
    sort_df = pd.Series(index=file_num, data=list(all_files))
    sort_df.sort_index(inplace=True)
    print(sort_df.head())
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

    # Remove index if it is in the exclude list
    from collections.abc import Iterable

    if exclude_exp is not None:
        if not isinstance(exclude_exp, Iterable):
            raise ValueError("exclude_exp should be an iterable of indices to exclude.")
        X = X.drop(index=exclude_exp)

    # Remove rows before the start_at index
    if start_at > 1:
        mask = X.index >= start_at
        X = X[mask]

    # Sort out timestamps - this will make the index the time the scan finishes - maybe discuss
    print(scan_time)
    print(X.index)
    if scan_time is None:
        X.index = range(len(X.index))
    else:
        pass
    if scan_time is not None:
        X.index = X.index.astype(float) * scan_time

    # Reduce data size if too large
    num_experiments = len(X.index)
    data_length = len(X.columns)
    if num_experiments > 1000:
        scan_granularity = num_experiments // 1000
    else:
        scan_granularity = 1
    if data_length > 1000:
        data_granularity = data_length // 1000
    else:
        data_granularity = 1
    print(f"Reducing data size: {num_experiments} experiments, {data_length} wavelengths")
    print(f"Scan granularity: {scan_granularity}, Data granularity: {data_granularity}")

    X = X.iloc[::scan_granularity, ::data_granularity]
    wavelength = wavelength[::data_granularity]

    print(X.index)
    metadata = {
        "time_range": {"min_time": min(X.index), "max_time": max(X.index)},
        "num_experiments": len(X.index),
    }
    return {"2D_data": X, "wavelength": wavelength, "metadata": metadata, "time_of_scan": X.index}
