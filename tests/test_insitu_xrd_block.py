from pathlib import Path

import pytest
from datalab_api.utils import bokeh_from_json

from datalab_app_plugin_insitu._version import __version__


def test_block_temperature_mode(pytestconfig):
    from datalab_app_plugin_insitu.apps.xrd.blocks import XRDInsituBlock

    test_data_zip = (
        Path(__file__).parent.parent / "example_data" / "xrd" / "Example_in_situ_XRD_data.zip"
    )
    block = XRDInsituBlock(item_id="test-xrd-insitu")
    assert block.version == __version__
    block.data["time_series_source"] = "log"
    block.data["xrd_folder_name"] = "43_up"
    block.data["time_series_folder_name"] = "log"
    block.data["glob_str"] = "*summed*"
    block.generate_insitu_xrd_plot(file_path=test_data_zip, link_plots=True)

    processed_data = block.data["processed"]
    expected_keys = (
        "2D_data",
        "Two theta",
        "metadata",
        "file_num_index",
        "Time_series_data",
        "index_df",
        "intensity_matrix",
        "spectra_intensities",
    )
    assert all(k in processed_data for k in expected_keys)

    num_experiments = 52
    start_temp = 23.95
    end_temp = 593.35
    start_scan_number = 1058063
    end_scan_number = 1058114
    two_theta_num_values_reduced = 1024
    unreduced_num_points = 22513

    assert processed_data["Two theta"][0] == pytest.approx(2.084)
    assert processed_data["Two theta"][-1] == pytest.approx(92.108)
    assert processed_data["Two theta"].shape[0] == two_theta_num_values_reduced

    assert processed_data["Two theta"][0] == pytest.approx(2.084)
    assert processed_data["Two theta"][-1] == pytest.approx(92.108)
    assert processed_data["Two theta"].shape[0] == two_theta_num_values_reduced

    assert processed_data["2D_data"].shape == (num_experiments, unreduced_num_points)

    assert (
        processed_data["Time_series_data"]["x"].shape[0]
        == processed_data["Time_series_data"]["y"].shape[0]
        == num_experiments
    )
    assert processed_data["Time_series_data"]["x"][0] == pytest.approx(start_temp)
    assert processed_data["Time_series_data"]["x"][-1] == pytest.approx(end_temp)
    assert processed_data["Time_series_data"]["y"][0] == start_scan_number
    assert processed_data["Time_series_data"]["y"][-1] == end_scan_number

    assert processed_data["index_df"].shape == (num_experiments, 3)
    assert list(processed_data["index_df"].columns) == ["file_num", "exp_num", "Temperature"]
    assert processed_data["index_df"]["file_num"].iloc[0] == start_scan_number
    assert processed_data["index_df"]["Temperature"].iloc[0] == pytest.approx(start_temp)
    assert processed_data["index_df"]["file_num"].iloc[-1] == end_scan_number
    assert processed_data["index_df"]["Temperature"].iloc[-1] == pytest.approx(end_temp)

    assert processed_data["spectra_intensities"].shape == (
        num_experiments,
        two_theta_num_values_reduced,
    )
    assert processed_data["intensity_matrix"].shape == (
        num_experiments,
        two_theta_num_values_reduced,
    )

    assert block.data["bokeh_plot_data"] is not None

    assert block.data["data_granularity"] == 22
    assert block.data["sample_granularity"] == 1

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

    processed_data = block.data["processed"]
    expected_keys = (
        "2D_data",
        "Two theta",
        "metadata",
        "file_num_index",
        "Time_series_data",
        "log data",
        "index_df",
        "intensity_matrix",
        "spectra_intensities",
    )
    assert all(k in processed_data for k in expected_keys)

    num_experiments = 5
    two_theta_num_values_reduced = 1035
    unreduced_num_points = 16553

    assert processed_data["Two theta"][0] == pytest.approx(2.08)
    assert processed_data["Two theta"].shape[0] == two_theta_num_values_reduced

    assert processed_data["Two theta"][0] == pytest.approx(2.08)
    assert processed_data["Two theta"].shape[0] == two_theta_num_values_reduced

    assert processed_data["2D_data"].shape == (num_experiments, unreduced_num_points)

    assert processed_data["spectra_intensities"].shape == (
        num_experiments,
        two_theta_num_values_reduced,
    )

    assert processed_data["index_df"].shape == (num_experiments, 24)
    assert all(
        k in list(processed_data["index_df"].columns)
        for k in ["file_num", "exp_num", "Cycle", "xrd_timestamp"]
    )
    assert processed_data["intensity_matrix"].shape == (
        num_experiments,
        two_theta_num_values_reduced,
    )

    assert block.data["bokeh_plot_data"] is not None

    assert block.data["bokeh_plot_data"] is not None

    if pytestconfig.getoption("show_plots"):
        bokeh_from_json(block.data["bokeh_plot_data"])
