import os
import pyreadr
import pytest

from src.datalab_app_plugin_insitu.nmr_insitu import process_datalab_data


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
def get_tests_data():
    """Download test data from the datalab instance."""

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import_data_path = os.path.join(
        base_dir,
        "example_data",
        "Example-TEGDME",
        "LiLiTEGDMEinsitu_02",
        "dfenv_LiLiTEGDMEinsitu_02.rds"
    )

    if not os.path.exists(import_data_path):
        raise FileNotFoundError(
            f"The file {import_data_path} does not exist.")

    df = pyreadr.read_r(import_data_path)
    import_data = df[None]
    import_data["time"] = import_data["time"] / 3600

    fit_peaks_path = os.path.join(
        base_dir,
        "example_data",
        "Example-TEGDME",
        "LiLiTEGDMEinsitu_02",
        "LiLiTEGDMEinsitu_02.rds"
    )
    fit_data = pyreadr.read_r(fit_peaks_path)
    df_fit_base = fit_data[None]
    df_fit_total_df = df_fit_base[df_fit_base['peak']
                                  == 'Total intensity']
    df_fit_peak1_df = df_fit_base[df_fit_base['peak'] == 'Peak 1']
    df_fit_peak2_df = df_fit_base[df_fit_base['peak'] == 'Peak 2']

    fit_peaks = {
        "data_df": df_fit_total_df[['time', 'intensity', 'norm_intensity']],
        "df_peakfit1": df_fit_peak1_df[['time', 'intensity', 'norm_intensity']],
        "df_peakfit2": df_fit_peak2_df[['time', 'intensity', 'norm_intensity']]
    }

    for _, data in fit_peaks.items():
        data.loc[:, 'time'] = data['time'] / 3600

    result = process_datalab_data(DATALAB_API_URL, "bc_insitu_block", "Example-TEGDME.zip",
                                  "2023-08-11_jana_insituLiLiTEGDME-02_galv", "LiLiTEGDMEinsitu_02", 220, 310)

    yield result, import_data, fit_peaks
