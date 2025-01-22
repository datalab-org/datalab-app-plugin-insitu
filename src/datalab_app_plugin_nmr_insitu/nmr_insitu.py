import os
import re
import zipfile
import tempfile

from datalab_api import DatalabClient
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from lmfit.models import PseudoVoigtModel
from numpy import exp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    DATALAB_API_URL = "http://localhost:5001/"
    client = DatalabClient(DATALAB_API_URL)

    print("#$%$#%$#%$#%$#%#$%#$%#$%$#")
    print(client.get_info())
    print("#$%$#%$#%$#%$#%#$%#$%#$%$#")

    with tempfile.TemporaryDirectory() as tmpdir:
        print("1")

        try:

            print("2")

            os.chdir(tmpdir)

            print("3")
            print(item_id)
            print(client.get_item_files(item_id=item_id))
            print("3")

            try:
                print("before")
                print(client.get_info())
                print(item_id)
                print(client.get_item_files(item_id=item_id))
                client.get_item_files(item_id=item_id)
                print(4)
            except Exception as e:
                print(f"Erreur lors de l'appel API: {e}")
            print(4)

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

                # Rename col ppm
                nmr_data = nmr_data.rename(
                    columns={nmr_data.columns[0]: 'ppm'})

                # Calculate intensities
                ppm = nmr_data['ppm']
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

                return nmr_data, df

            # Process data
            spec_paths, acqu_paths = setup_paths()
            time_points = process_time_data(acqu_paths)
            nmr_data, df = process_spectral_data(spec_paths, time_points)

            return nmr_data, df

        except Exception as e:
            raise RuntimeError(f"Error processing NMR data: {str(e)}")


def fitting_data(
        nmr_data: pd.DataFrame,
        df: pd.DataFrame,
) -> Dict:
    """
    Perform fitting using pseudo-Voigt Model on insitu NMR data.

    Args:
        nmr_data (pd.DataFrame): Raw NMR spectral data from previous processing
        df (pd.DataFrame): Time, Intensities and Normalised intensities data from previous processing

    Returns:
        Dict: Fitting results and processed data
    """
    try:

        ppm = np.array(nmr_data['ppm'], dtype=float)
        tNMR = np.array(df['time'], dtype=float)
        env = np.array(df['intensity'], dtype=float)
        env_peak1 = []
        env_peak2 = []

        for x in range(1, nmr_data.shape[1]):
            intensity = np.array(nmr_data.iloc[:, x], dtype=float)

            model1 = PseudoVoigtModel(prefix='peak1_')
            model2 = PseudoVoigtModel(prefix='peak2_')

            model = model1 + model2

            params = model.make_params()
            params['peak1_amplitude'].set(value=8.976e5, min=1e4, max=6e7)
            params['peak1_center'].set(value=248.0, min=244.0, max=252.5)
            params['peak1_sigma'].set(value=5, min=0.5, max=6.5)
            params['peak1_fraction'].set(value=0.3, min=0.2, max=1)

            params['peak2_amplitude'].set(value=12.394e5, min=0, max=5e7)
            params['peak2_center'].set(value=266.0, min=256.0, max=276)
            params['peak2_sigma'].set(value=5, min=0.5, max=6.5)
            params['peak2_fraction'].set(value=0.3, min=0.2, max=1)

            result = model.fit(intensity, x=ppm, params=params)

            peak1_params = {name: param for name, param in result.params.items()
                            if name.startswith('peak1_')}
            peak2_params = {name: param for name, param in result.params.items()
                            if name.startswith('peak2_')}

            peak1_intensity = model1.eval(
                params=peak1_params, x=ppm)
            peak2_intensity = model2.eval(
                params=peak2_params, x=ppm)

            env_peak1.append(abs(np.trapz(peak1_intensity, x=ppm)))
            env_peak2.append(abs(np.trapz(peak2_intensity, x=ppm)))

        norm_intensity_peak1 = [x/max(env) for x in env_peak1]
        norm_intensity_peak2 = [x/max(env) for x in env_peak2]

        def data_fitted(tNMR, peak_intensity, norm_intensity):
            result = pd.DataFrame({
                'time': tNMR,
                'intensity': peak_intensity,
                'norm_intensity': norm_intensity,
            })
            return result

        df_peakfit1 = data_fitted(tNMR, env_peak1, norm_intensity_peak1)
        df_peakfit2 = data_fitted(tNMR, env_peak2, norm_intensity_peak2)

        df_fit = {
            "data_df": {
                "time": df["time"].tolist(),
                "intensity": df["intensity"].tolist(),
                "norm_intensity": df["norm_intensity"].tolist()
            },
            "df_peakfit1": {
                "time": df_peakfit1["time"].tolist(),
                "intensity": df_peakfit1["intensity"].tolist(),
                "norm_intensity": df_peakfit1["norm_intensity"].tolist()
            },
            "df_peakfit2": {
                "time": df_peakfit2["time"].tolist(),
                "intensity": df_peakfit2["intensity"].tolist(),
                "norm_intensity": df_peakfit2["norm_intensity"].tolist()
            }
        }

        return df_fit

    except Exception as e:
        raise RuntimeError(f"Error fitting NMR data: {str(e)}")
