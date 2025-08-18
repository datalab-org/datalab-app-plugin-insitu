from pathlib import Path

import pytest
from datalab_api.utils import bokeh_from_json


@pytest.fixture
def test_data_zip(tmp_path):
    """Zips the test data into a tmp directory."""
    import shutil

    test_data_dir = Path(__file__).parent.parent / "example_data" / "nmr"
    test_data_zip_name = "Example-TEGDME.zip"

    test_data_zip_path = tmp_path / test_data_zip_name

    # Create a zip file of the test data directory
    shutil.copy(test_data_dir / test_data_zip_name, test_data_zip_path)

    yield test_data_zip_path
    test_data_zip_path.unlink(missing_ok=True)


def test_block(test_data_zip, pytestconfig):
    from datalab_app_plugin_insitu.apps.nmr.blocks import InsituBlock

    block = InsituBlock(item_id="test-nmr-insitu")
    assert (
        block.count_experiments_in_nmr_folder(
            test_data_zip, "2023-08-11_jana_insituLiLiTEGDME-02_galv"
        )
        == 50
    )

    block.data["nmr_folder_name"] = "2023-08-11_jana_insituLiLiTEGDME-02_galv"
    block.data["echem_folder_name"] = "LiLiTEGDMEinsitu_02"
    block.generate_insitu_nmr_plot(file_path=test_data_zip, link_plots=True)

    assert block.data["bokeh_plot_data"] is not None

    if pytestconfig.getoption("show_plots"):
        bokeh_from_json(block.data["bokeh_plot_data"])
