from pathlib import Path

from datalab_api.utils import bokeh_from_json


def test_block_temperature_mode(pytestconfig):
    from datalab_app_plugin_insitu.apps.xrd.blocks import XRDInsituBlock

    test_data_zip = (
        Path(__file__).parent.parent / "example_data" / "xrd" / "Example_in_situ_XRD_data.zip"
    )
    block = XRDInsituBlock(item_id="test-xrd-insitu")
    block.data["time_series_source"] = "log"
    block.data["xrd_folder_name"] = "43_up"
    block.data["time_series_folder_name"] = "log"
    block.data["glob_str"] = "*summed*"
    block.generate_insitu_xrd_plot(file_path=test_data_zip, link_plots=True)

    assert block.data["bokeh_plot_data"] is not None

    if pytestconfig.getoption("show_plots"):
        bokeh_from_json(block.data["bokeh_plot_data"])


def test_block_echem_mode(pytestconfig):
    from datalab_app_plugin_insitu.apps.xrd.blocks import XRDInsituBlock

    test_data_zip = (
        Path(__file__).parent.parent / "example_data" / "xrd" / "coincell8_LDE2_reduced.zip"
    )
    block = XRDInsituBlock(item_id="test-xrd-insitu-echem")
    block.data["time_series_source"] = "echem"
    block.data["xrd_folder_name"] = "xrd_data"
    block.data["time_series_folder_name"] = "Log"
    block.data["echem_folder_name"] = "Echem"
    block.generate_insitu_xrd_plot(file_path=test_data_zip, link_plots=True)

    assert block.data["bokeh_plot_data"] is not None

    if pytestconfig.getoption("show_plots"):
        bokeh_from_json(block.data["bokeh_plot_data"])
