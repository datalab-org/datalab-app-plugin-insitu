from pathlib import Path
from typing import Dict

from navani import echem as ec


def process_echem_data(echem_folder: Path) -> Dict:
    """
    Processes Echem data from a specified file or files. Note if multiple files are present, they will be concatenated, so do not include unrelated or duplicate files in the folder.

    Args:
        echem_folder: Path to the Echem data folder containing the echem files.

    Returns:
        Dictionary containing the processed Echem data with keys "time" and "data"
    """
    from pydatalab.apps.echem import CycleBlock

    # Makes sure accepted file extensions are unique (There was an error where .txt appeared twice in CycleBlock leading to every .txt file being added to echem files twice)
    accepted_extensions = tuple(set(CycleBlock.accepted_file_extensions))
    if echem_folder.exists():
        echem_files: list[Path] = []
        for ext in accepted_extensions:
            echem_files.extend(echem_folder.glob(f"*{ext}"))

        # Sanity check to remove any duplicate files
        echem_files = list(dict.fromkeys(echem_files))
        echem_files.sort()
        if len(echem_files) > 1:
            echem_data = ec.multi_echem_file_loader(echem_files)
        elif len(echem_files) == 0:
            raise ValueError(f"Echem folder should contain at least one file: {echem_folder}")
        elif len(echem_files) == 1:
            echem_file = echem_files[0]
            echem_data = ec.echem_file_loader(echem_file)

    else:
        raise ValueError(f"Echem folder not found: {echem_folder}")

    # Rename timestamp column to a consistent name
    if "Timestamp" in echem_data.columns:
        timestamp_col = "Timestamp"
    elif "Date_Time" in echem_data.columns:
        echem_data = echem_data.rename(columns={"Date_Time": "Timestamp"})
        timestamp_col = "Timestamp"
    else:
        timestamp_col = None

    if timestamp_col:
        time_deltas = echem_data["Timestamp"] - echem_data["Timestamp"].iloc[0]
        echem_data["elapsed_time_seconds"] = [delta.total_seconds() for delta in time_deltas]
        echem_data["elapsed_time_hours"] = [delta.total_seconds() / 3600 for delta in time_deltas]
        echem_data["Time"] = echem_data["elapsed_time_seconds"]

    min_time = echem_data["Time"].min()
    max_time = echem_data["Time"].max()

    return_dict = {
        "time": echem_data["Time"],
        "Voltage": echem_data["Voltage"],
        "metadata": {
            "min_y": min_time,
            "max_y": max_time,
        },
        "data": echem_data,
    }

    return return_dict
