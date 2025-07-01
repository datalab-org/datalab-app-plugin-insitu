from pathlib import Path


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


def test_version():
    from datalab_app_plugin_insitu import __version__

    assert __version__.startswith("0")


def test_block(pytestconfig):
    from datalab_app_plugin_insitu.apps.uvvis.blocks import UVVisInsituBlock

    test_data_zip = Path(__file__).parent.parent / "example_data" / "Example_in_situ_folder 2.zip"
    block = UVVisInsituBlock(item_id="test-uvvis-insitu")
    block.data["uvvis_folder_name"] = "Reduced sample scans"
    block.data["echem_folder_name"] = "Echem"
    block.data["uvvis_reference_folder_name"] = "Reference scan"
    block.data["scan_time"] = 1.0  # seconds
    block.generate_insitu_uvvis_plot(file_path=test_data_zip, link_plots=True)

    assert block.data["bokeh_plot_data"] is not None

    if pytestconfig.getoption("show_plots"):
        bokeh_from_json(block.data["bokeh_plot_data"])
