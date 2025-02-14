import os
import pyreadr
import pytest
import zipfile
import tempfile
import pickle

from datalab_api import DatalabClient
from datalab_app_plugin_nmr_insitu.nmr_insitu import process_datalab_data, fitting_data


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

    result = process_datalab_data("https://demo.datalab-org.io/", "bc_nmr_insitu", "Example-TEGDME",
                                  "2023-08-11_jana_insituLiLiTEGDME-02_galv", "LiLiTEGDMEinsitu_02", 220, 310)

    yield result
