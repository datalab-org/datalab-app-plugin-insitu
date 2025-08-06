from pathlib import Path

from datalab_api.utils import bokeh_from_json


def test_block(pytestconfig):
    from datalab_app_plugin_insitu.apps.xrd.blocks import XRDInsituBlock

    test_data_zip = Path(__file__).parent.parent / "example_data" / "XRD_example_insitu_data.zip"
    block = XRDInsituBlock(item_id="test-xrd-insitu")
    block.data["xrd_folder_name"] = "43_up"
    block.data["time_series_folder_name"] = "log"
    block.generate_insitu_xrd_plot(file_path=test_data_zip, link_plots=True)

    assert block.data["bokeh_plot_data"] is not None

    if pytestconfig.getoption("show_plots"):
        bokeh_from_json(block.data["bokeh_plot_data"])
