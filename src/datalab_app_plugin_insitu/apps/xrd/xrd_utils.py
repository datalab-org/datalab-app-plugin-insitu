import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from datalab_app_plugin_insitu.echem_utils import process_echem_data
from datalab_app_plugin_insitu.utils import _find_folder_path

from pydatalab.apps.xrd.blocks import XRDBlock

def process_local_xrd_data(file_path: str | Path, xrd_folder_name: str, start_exp: int = 1, exclude_exp: list = None):
    """
    Process local XRD data from a zip file.

    Parameters:
        file_path (str | Path): Path to the zip file containing XRD data.
        xrd_folder_name (str): Name of the folder containing XRD data.
        echem_folder_name (str): Name of the folder containing electrochemical data.
        start_exp (int): Starting experiment number.
        exclude_exp (list): List of experiments to exclude.

    Returns:
        dict: Processed XRD data and metadata.
    """

    # Check if the folder exists
    if not all([xrd_folder_name]):
        raise ValueError("XRD folder must be specified.")
    

    
    # Placeholder for actual processing logic
    return XRDBlock.process_and_store_data(file_path, xrd_folder_name, start_exp, exclude_exp)

def process_xrd_data(
        xrd_folder: Path,
        log_file: Path,
        start_at: int = 1,
        exclude_exp: Optional[List[int]] = None,
) ->Dict:
    """
    Process XRD data from a specified folder.

    Args:
        xrd_folder (Path): Path to the folder containing XRD data.
        log_file (Path): Path to the log file for storing processing results.
        start_at (int): Starting experiment number.
        exclude_exp (Optional[List[int]]): List of experiments to exclude.

    Returns:
        Dict: Processed XRD data and metadata.
    """
    if not xrd_folder.exists():
        raise FileNotFoundError(f"XRD folder does not exist: {xrd_folder}")

    # Process the XRD data

    return xrd_data