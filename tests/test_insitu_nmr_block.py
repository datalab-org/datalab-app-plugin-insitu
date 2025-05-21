from pathlib import Path

import pytest


def test_version():
    from datalab_app_plugin_insitu import __version__

    assert __version__.startswith("0")


@pytest.fixture
def test_data_zip(tmp_path):
    """Zips the test data into a tmp directory."""
    import shutil

    test_data_dir = Path(__file__).parent.parent / "example_data" / "Example-TEGDME"
    test_data_zip_path = tmp_path / "Example-TEGDME.zip"

    # Create a zip file of the test data directory
    shutil.make_archive(str(test_data_zip_path.with_suffix("")), "zip", test_data_dir)

    yield test_data_zip_path
    test_data_zip_path.unlink(missing_ok=True)


def test_block(test_data_zip):
    from datalab_app_plugin_insitu.apps.nmr.blocks import InsituBlock

    block = InsituBlock(item_id="test-nmr-insitu")
    block.data["nmr_folder_name"] = "2023-08-11_jana_insituLiLiTEGDME-02_galv"
    block.data["echem_folder_name"] = "LiLiTEGDMEinsitu_02"
    block.generate_insitu_nmr_plot(file_path=test_data_zip, link_plots=False)
    assert block.data["bokeh_plot_data"] is not None
