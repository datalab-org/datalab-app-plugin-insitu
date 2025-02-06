import os
import zipfile
import tempfile
import warnings

from datalab_api import DatalabClient
from typing import List, Optional, Dict
from lmfit.models import PseudoVoigtModel
from .utils import check_nmr_dimension, setup_paths, process_time_data, process_spectral_data, process_echem_data, prepare_for_bokeh, process_pseudo2d_spectral_data

import numpy as np
import pandas as pd


def process_data(
    api_url: str,
    item_id: str,
    folder_name: str,
    nmr_folder_name: str,
    echem_folder_name: str,
    ppm1: float,
    ppm2: float,
    start_at: int = 1,
    exclude_exp: Optional[List[int]] = None,
) -> Dict:
    """
    Process NMR spectroscopy data from multiple experiments.

    Args:
        api_url: URL of the Datalab API
        item_id: ID of the item to process
        folder_name: Base folder name
        nmr_folder_name: Folder containing NMR experiments
        echem_folder_name: Folder containing Echem data
        ppm1: Lower PPM range limit
        ppm2: Upper PPM range limit
        start_at: Starting experiment number (default: 1)
        exclude_exp: List of experiment numbers to exclude (default: None)

    Returns:
        Dictionary containing processed NMR and electrochemical data

    Raises:
        ValueError: If required parameters are missing
        RuntimeError: If processing fails
    """

    if not all([folder_name, nmr_folder_name]):
        raise ValueError("Folder names for NMR data are required")

    client = DatalabClient(api_url)

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.chdir(tmpdir)
            try:
                client.get_item_files(item_id=item_id)
            except Exception as e:
                raise RuntimeError(f"API error: {e}")

            zip_path = os.path.join(tmpdir, folder_name)
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"ZIP file not found: {folder_name}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            folder_name = os.path.splitext(folder_name)[0]
            nmr_folder_name = os.path.splitext(nmr_folder_name)[0]
            nmr_folder_path = os.path.join(
                tmpdir, folder_name, nmr_folder_name)

            if not os.path.exists(nmr_folder_path):
                raise FileNotFoundError(
                    f"NMR folder not found: {nmr_folder_name}")

            echem_folder_path = os.path.join(
                tmpdir, folder_name, echem_folder_name)

            if not os.path.exists(echem_folder_path):
                warnings.warn(
                    f"Echem folder not found: {echem_folder_name}")

            nmr_dimension = check_nmr_dimension(nmr_folder_path)

            if nmr_dimension == '1D':
                spec_paths, acqu_paths = setup_paths(
                    nmr_folder_path, start_at, exclude_exp)
                time_points = process_time_data(acqu_paths)
                nmr_data, df = process_spectral_data(
                    spec_paths, time_points, ppm1, ppm2)
                merged_df = process_echem_data(
                    tmpdir, folder_name, echem_folder_name) if echem_folder_name else None
                result = prepare_for_bokeh(nmr_data, df, merged_df)
            elif nmr_dimension == 'pseudo2D':

                exp_folders = [d for d in os.listdir(nmr_folder_path) if os.path.isdir(
                    os.path.join(nmr_folder_path, d)) and d.isdigit()]

                if not exp_folders:
                    raise FileNotFoundError(
                        "No experience file found in NMR data")

                exp_folder = os.path.join(nmr_folder_path, exp_folders[0])
                nmr_data, df = process_pseudo2d_spectral_data(
                    exp_folder, ppm1, ppm2)

                merged_df = process_echem_data(
                    tmpdir, folder_name, echem_folder_name) if echem_folder_name else None
                result = prepare_for_bokeh(nmr_data, df, merged_df)
            else:
                raise ValueError(
                    f"Unknown NMR dimension type: {nmr_dimension}")

            return result

        except Exception as e:
            raise RuntimeError(f"Error processing NMR data: {str(e)}")


#! Will need to be handle by UI at some point if we want to keep fitting
FITTING_CONFIG = {
    'peak1': {
        'amplitude': {'value': 8.976e5, 'min': 1e4, 'max': 6e7},
        'center': {'value': 248.0, 'min': 244.0, 'max': 252.5},
        'sigma': {'value': 5, 'min': 0.5, 'max': 6.5},
        'fraction': {'value': 0.3, 'min': 0.2, 'max': 1}
    },
    'peak2': {
        'amplitude': {'value': 12.394e5, 'min': 0, 'max': 5e7},
        'center': {'value': 266.0, 'min': 256.0, 'max': 276},
        'sigma': {'value': 5, 'min': 0.5, 'max': 6.5},
        'fraction': {'value': 0.3, 'min': 0.2, 'max': 1}
    }
}


def fitting_data(nmr_data: pd.DataFrame, df: pd.DataFrame, config: dict = FITTING_CONFIG) -> Dict:
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
            for param, settings in config['peak1'].items():
                params[f'peak1_{param}'].set(**settings)
            for param, settings in config['peak2'].items():
                params[f'peak2_{param}'].set(**settings)

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
