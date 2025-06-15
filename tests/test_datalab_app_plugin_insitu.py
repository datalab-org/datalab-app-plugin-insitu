from pathlib import Path

import pytest

from datalab_app_plugin_insitu import __version__


def test_version():
    assert __version__.startswith("0")


def bokeh_from_json(block_data, show=True):
    """Renders bokeh plot data from JSON representation.
    Vendored from datalab_api.utils (>=0.2.14).
    """
    from bokeh.io import curdoc
    from bokeh.plotting import show as bokeh_show

    bokeh_plot_data = block_data.get("bokeh_plot_data", block_data)
    curdoc().replace_with_json(bokeh_plot_data["doc"])
    if show:
        bokeh_show(curdoc().roots[0])

    return curdoc()


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


def test_block(test_data_zip, pytestconfig):
    from datalab_app_plugin_insitu.apps.nmr.blocks import InsituBlock

    block = InsituBlock(item_id="test_item_id")
    block.data["nmr_folder_name"] = "2023-08-11_jana_insituLiLiTEGDME-02_galv"
    block.data["echem_folder_name"] = "LiLiTEGDMEinsitu_02"
    assert block.should_reprocess_data()

    block.process_and_store_data(file_path=test_data_zip)
    assert "ppm" in block.data["nmr_data"]
    assert len(block.data["nmr_data"]["ppm"]) == 8192
    assert len(block.data["nmr_data"]["spectra"]) == 600
    assert max(block.data["nmr_data"]["ppm"]) == 537.7439
    assert min(block.data["nmr_data"]["ppm"]) == -263.103409
    assert "Voltage" in block.data["echem_data"]
    assert "time" in block.data["echem_data"]
    assert len(block.data["echem_data"]["Voltage"]) == 76916
    assert len(block.data["echem_data"]["time"]) == 76916
    assert "time_range" in block.data["metadata"]
    assert "ppm1" in block.data["processing_params"]
    assert "ppm2" in block.data["processing_params"]
    assert not block.should_reprocess_data()

    block.generate_insitu_nmr_plot(file_path=test_data_zip)

    assert block.data["bokeh_plot_data"] is not None

    bokeh_from_json(block.data["bokeh_plot_data"])
