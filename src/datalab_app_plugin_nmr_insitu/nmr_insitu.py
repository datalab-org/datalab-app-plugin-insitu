import os
import zipfile
import tempfile

from datalab_api import DatalabClient
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


def process_data(
    item_id: str,
    file_name: str,
    ppm1: float,
    ppm2: float,
    start_at: int = 1,
    exclude_exp: Optional[List[int]] = None,
) -> Dict:
    """
    Process NMR spectroscopy data from multiple experiments.

    Args:
        folder_path (str): Base folder containing experiments
        ppm1 (float): Lower PPM range limit
        ppm2 (float): Upper PPM range limit
        start_at (int, optional): Starting experiment number. Defaults to 1
        exclude_exp (List[int], optional): List of experiment numbers to exclude

    Returns:
        pandas.DataFrame: A dataframe with insitu NMR data: time, intensities and normalised intensities
    """

    DATALAB_API_URL = "https://demo-api.datalab-org.io"
    client = DatalabClient(DATALAB_API_URL)

    with tempfile.TemporaryDirectory() as tmpdir:
        try:

            os.chdir(tmpdir)

            client.get_item_files(item_id=item_id)

            zip_path = os.path.join(tmpdir, file_name)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            folder_name = os.path.splitext(file_name)[0]
            folder_path = os.path.join(tmpdir, folder_name)

            def extract_date_from_acqus(path: str) -> Optional[datetime]:
                """Extract date from acqus file."""
                try:
                    with open(path, 'r') as file:
                        for line in file:
                            if line.startswith('$$'):
                                match = re.search(
                                    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \+\d{4})', line)
                                if match:
                                    date_str = match.group(1)
                                    return datetime.strptime(
                                        date_str, '%Y-%m-%d %H:%M:%S.%f %z')
                except Exception as e:
                    print(f"Warning: Could not extract date from {path}: {e}")
                return None

            def setup_paths() -> Tuple[List[str], List[str]]:
                """Setup experiment paths and create output directory."""

                # Get number of experiments
                nos_experiments = len([d for d in os.listdir(
                    folder_path) if os.path.isdir(os.path.join(folder_path, d))])

                # Generate experiment list
                exp_folder = [exp for exp in range(
                    start_at, nos_experiments + 1) if exp not in (exclude_exp or [])]

                # Generate paths
                spec_paths = [
                    f"{folder_path}/{exp}/pdata/1/ascii-spec.txt" for exp in exp_folder]
                acqu_paths = [
                    f"{folder_path}/{exp}/acqus" for exp in exp_folder]

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

            def process_spectral_data(spec_paths: List[str], time_points: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
                """Process spectral data from ascii-spec files."""
                # Read first file to get PPM values
                first_data = pd.read_csv(
                    spec_paths[0], header=None, skiprows=1)
                ppm_values = first_data.iloc[:, 3].values

                # Create matrix for all data
                nmr_data = pd.DataFrame(index=range(
                    len(ppm_values)), columns=range(len(spec_paths) + 1))
                nmr_data.iloc[:, 0] = ppm_values

                # Fill matrix with intensity values
                for i, path in enumerate(spec_paths):
                    data = pd.read_csv(path, header=None, skiprows=1)
                    nmr_data.iloc[:, i + 1] = data.iloc[:, 1]

                # Filter by PPM range
                nmr_data = nmr_data[(nmr_data.iloc[:, 0] > ppm1) &
                                    (nmr_data.iloc[:, 0] < ppm2)]

                # Calculate intensities
                ppm = nmr_data.iloc[:, 0].values
                intensities = []

                for m in range(1, nmr_data.shape[1]):
                    y = nmr_data.iloc[:, m].values
                    intensities.append(abs(np.trapz(y, x=ppm)))

                norm_intensities = [x/max(intensities) for x in intensities]

                # Create time series DataFrame
                df = pd.DataFrame({
                    'time': time_points,
                    'intensity': intensities,
                    'norm_intensity': norm_intensities,
                })

                return df

            # Process data
            spec_paths, acqu_paths = setup_paths()
            time_points = process_time_data(acqu_paths)
            df = process_spectral_data(spec_paths, time_points)

            return df

        except Exception as e:
            raise RuntimeError(f"Error processing NMR data: {str(e)}")
