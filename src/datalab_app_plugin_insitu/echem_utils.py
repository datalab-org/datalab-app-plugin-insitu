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
    from pydatalab.apps.echem import CycleBlock

    accepted_extensions = CycleBlock.accepted_file_extensions
    if echem_folder.exists():
        echem_files: list[Path] = []
        for ext in accepted_extensions:
            echem_files.extend(echem_folder.glob(f"*{ext}"))

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
