import os
import pytest
from datalab_api import DatalabClient
import zipfile

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


@pytest.fixture()
def get_demo_data(tmpdir):
    """Download test data from the datalab instance."""

    client = DatalabClient(DATALAB_API_URL)

    data_dir_1 = tmpdir.mkdir("data_1")
    data_dir_2 = tmpdir.mkdir("data_2")

    os.chdir(data_dir_1)
    client.get_item_files("bc_nmr_insitu")
    test_path_1 = data_dir_1 / "demo_data_nmr_insitu.zip"

    os.chdir(data_dir_2)
    client.get_item_files("bc_nmr_insitu_2")
    test_path_2 = data_dir_2 / "demo_data_nmr_insitu_python.zip"

    assert test_path_1.exists(
    ), f"File {test_path_1} does not exist"
    assert test_path_2.exists(
    ), f"File {test_path_2} does not exist"

    yield test_path_1, test_path_2

    # Clean up tmp data after test
    test_path_1.remove()
    test_path_2.remove()
