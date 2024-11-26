import os
import pyreadr
import pytest
import zipfile
import tempfile
import pickle

from datalab_api import DatalabClient
from datalab_app_plugin_nmr_insitu.nmr_insitu import process_data, fitting_data


DATALAB_API_URL = "https://demo-api.datalab-org.io"


@pytest.fixture(scope="session")
def test_dir():
    from pathlib import Path

    module_dir = Path(__file__).resolve().parent
    test_dir = module_dir / "test_data"
    return test_dir.resolve()


@pytest.fixture(scope="session")
def log_to_stdout():
    import logging
    import sys

    # Set Logging
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    root.addHandler(ch)


@pytest.fixture(scope="session")
def clean_dir(debug_mode):
    import os
    import shutil
    import tempfile

    old_cwd = os.getcwd()
    newpath = tempfile.mkdtemp()
    os.chdir(newpath)
    yield
    if debug_mode:
        print(f"Tests ran in {newpath}")
    else:
        os.chdir(old_cwd)
        shutil.rmtree(newpath)


@pytest.fixture
def tmp_dir():
    """Same as clean_dir but is fresh for every test"""
    import os
    import shutil
    import tempfile

    old_cwd = os.getcwd()
    newpath = tempfile.mkdtemp()
    os.chdir(newpath)
    yield
    os.chdir(old_cwd)
    shutil.rmtree(newpath)


@pytest.fixture(scope="session")
def debug_mode():
    return False


@pytest.fixture
def percentage_difference():
    """
    Calculate the percentage difference between two values.

    Args:
        val1 (float): First value
        val2 (float): Second value

    Returns:
        float: Percentage difference between val1 and val2
    """
    def calculate(val1, val2):
        if val1 == 0 or val2 == 0:
            return 0
        return abs(val1 - val2) / ((val1 + val2) / 2) * 100
    return calculate


@pytest.fixture()
def get_demo_data(tmpdir):
    """Download test data from the datalab instance."""

    client = DatalabClient(DATALAB_API_URL)

    data_dir = tmpdir.mkdir("data")

    file_name = "demo_data_nmr_insitu.zip"
    zip_path = data_dir / file_name

    os.chdir(data_dir)
    client.get_item_files("bc_nmr_insitu")

    assert zip_path.exists(), f"File {zip_path} does not exist"

    extract_dir = data_dir

    required_files = [
        "demo_data_nmr_insitu_df.rds",
        "demo_data_nmr_insitu_fit_and_dfall.rds"
    ]

    extracted_paths = {}
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_contents = zip_ref.namelist()
        for required_file in required_files:
            if required_file not in zip_contents:
                raise FileNotFoundError(
                    f"{required_file} not found in {zip_path}"
                )
            zip_ref.extract(required_file, path=extract_dir)
            extracted_paths[required_file] = os.path.join(
                extract_dir, required_file)

    df_path = extracted_paths["demo_data_nmr_insitu_df.rds"]
    fit_and_dfall_path = extracted_paths["demo_data_nmr_insitu_fit_and_dfall.rds"]

    df_data = pyreadr.read_r(df_path)
    df = df_data[None]
    df["time"] = df["time"] / 3600

    fit_data = pyreadr.read_r(fit_and_dfall_path)
    df_fit_base = fit_data[None]
    df_fit_total_df = df_fit_base[df_fit_base['peak']
                                  == 'Total intensity']
    df_fit_peak1_df = df_fit_base[df_fit_base['peak'] == 'Peak 1']
    df_fit_peak2_df = df_fit_base[df_fit_base['peak'] == 'Peak 2']

    df_fit = {
        "data_df": df_fit_total_df[['time', 'intensity', 'norm_intensity']],
        "df_peakfit1": df_fit_peak1_df[['time', 'intensity', 'norm_intensity']],
        "df_peakfit2": df_fit_peak2_df[['time', 'intensity', 'norm_intensity']]
    }

    for key, data in df_fit.items():
        data.loc[:, 'time'] = data['time'] / 3600

    nmr_data, processed_df = process_data(
        "bc_nmr_insitu", "demo_dataset_nmr_insitu.zip", 220, 310)

    processed_df_fit = fitting_data(nmr_data, df)

    yield df, df_fit, processed_df, processed_df_fit

    for path in extracted_paths.values():
        os.remove(path)
