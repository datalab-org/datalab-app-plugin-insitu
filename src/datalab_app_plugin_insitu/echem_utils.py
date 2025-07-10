from pathlib import Path
from typing import Dict

from navani import echem as ec


def process_echem_data(echem_folder: Path) -> Dict:
    """
    Processes Echem data from a specified file.

    Args:
        echem_file: Path to the Echem data file

    Returns:
        Dictionary containing the processed Echem data with keys "time" and "data"
    """

    if echem_folder.exists():
        echem_files = list(echem_folder.glob("*.txt"))
        if len(echem_files) > 1:
            raise ValueError(
                f"Echem folder should contain exactly one file: {echem_folder}. Found {len(echem_files)} files. Files found: {echem_files}"
            )
            # TODO handle multiple files
        elif len(echem_files) == 0:
            raise ValueError(f"Echem folder should contain at least one file: {echem_folder}")
        else:
            echem_file = echem_files[0]
            echem_data = ec.echem_file_loader(echem_file)

    else:
        raise ValueError(f"Echem folder not found: {echem_folder}")

    min_time = echem_data["Time"].min()
    max_time = echem_data["Time"].max()

    return_dict = {
        "time": echem_data["Time"],
        "Voltage": echem_data["Voltage"],
        "metadata": {
            "min_time": min_time,
            "max_time": max_time,
        },
    }

    return return_dict
