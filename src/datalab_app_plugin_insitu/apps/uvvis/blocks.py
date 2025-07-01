import warnings
from pathlib import Path
from typing import List

import bokeh.embed

from datalab_app_plugin_insitu.apps.uvvis.utils import process_local_uvvis_data
from datalab_app_plugin_insitu.blocks import GenericInSituBlock
from datalab_app_plugin_insitu.plotting_uvvis import (
    create_linked_insitu_plots,
    prepare_uvvis_plot_data,
)


class UVVisInsituBlock(GenericInSituBlock):
    """This datablock processes in situ UV-Vis data from an input .zip file containing two specific directories:

    - UV-Vis Data Directory: Contains multiple UV-Vis in-situ experiment datasets.
    - Echem Data Directory: Contains echem data files in `.txt` format.

    """

    blocktype = "insitu-uvvis"
    name = "UV-Vis insitu"
    description = __doc__
    accepted_file_extensions = (".zip",)
    available_folders: List[str] = []
    uvvis_folder_name = None
    echem_folder_name = None
    folder_name = None
    plotting_label_dict = {
        "x_axis_label": "Wavelength (nm)",
        "time_series_y_axis_label": "Time (s)",
        "line_y_axis_label": "Intensity (a.u.)",
        "time_series_x_axis_label": "Voltage (V)",
        "label_source": {
            "label_template": "Exp. # {exp_num} | t = {time} s | V = {voltage} V",
            "label_field_map": {
                "exp_num": "exp_num",
                "time": "times_by_exp",
                "voltage": "voltages_by_exp",
            },
        },
    }

    defaults = {
        "start_exp": 0,
        "exclude_exp": None,
        "scan_time": None,
        "target_sample_number": 1000,
        "target_data_number": 1000,
        "data_granularity": None,
        "sample_granularity": None,
    }

    def _plot_function(self, file_path=None, link_plots=True):
        return self.generate_insitu_uvvis_plot(file_path=file_path, link_plots=link_plots)

    def process_and_store_data(self, file_path: str | Path):
        """
        Process all in situ UV-Vis and electrochemical data and store results.
        This method is a wrapper for processing both UV-Vis and electrochemical data.
        """
        scan_time = self.data.get("scan_time", self.defaults["scan_time"])
        if not scan_time:
            raise ValueError(
                "Scan time is required for processing UV-Vis data. Should include the time between scans in seconds."
            )
        file_path = Path(file_path)
        folders = self.get_available_folders(file_path)
        self.data["available_folders"] = folders

        if not self.data.get("uvvis_folder_name"):
            raise ValueError("UV-Vis folder name is required")
        uvvis_folder_name = Path(self.data.get("uvvis_folder_name"))
        if not self.data.get("uvvis_reference_folder_name"):
            raise ValueError("Reference folder name is required")
        reference_folder_name = Path(self.data.get("uvvis_reference_folder_name"))
        if not self.data.get("echem_folder_name"):
            raise ValueError("Echem folder name is required")
        echem_folder_name = Path(self.data.get("echem_folder_name"))

        start_exp = int(self.data.get("start_exp", self.defaults["start_exp"]))
        exclude_exp = self.data.get("exclude_exp", self.defaults["exclude_exp"])

        try:
            data = process_local_uvvis_data(
                folder_name=file_path,
                uvvis_folder=uvvis_folder_name,
                reference_folder=reference_folder_name,
                echem_folder=echem_folder_name,
                start_at=start_exp,
                exclude_exp=exclude_exp,
                # TODO Needs to be made more generic
                sample_file_extension=".Raw8.txt",
                reference_file_extension=".Raw8.TXT",
                scan_time=scan_time,
            )

            num_samples, data_length = data["2D_data"].shape
            try:
                from pydatalab.logger import LOGGER

                LOGGER.info(f"Loading in situ UVVis with {num_samples=} and {data_length=}")
            except ImportError:
                pass

            sample_granularity = self.data.get(
                "sample_granularity", self.defaults["sample_granularity"]
            )
            data_granularity = self.data.get("data_granularity", self.defaults["data_granularity"])
            if not sample_granularity:
                if num_samples > self.data.get("target_sample_number"):
                    sample_granularity = num_samples // self.data.get("target_sample_number")
                else:
                    sample_granularity = 1
            if not data_granularity:
                if data_length > self.data.get("target_data_number"):
                    data_granularity = data_length // self.data.get("target_data_number")
                else:
                    data_granularity = 1

            self.data["sample_granularity"] = sample_granularity
            self.data["data_granularity"] = data_granularity
            # Subsample the 2D data and wavelength data to a maximum of 1000 samples and data points
            data["2D_data"] = self.subsample_data(
                data["2D_data"],
                sample_granularity=sample_granularity,
                data_granularity=data_granularity,
                method="linear",
            )

            data["wavelength"] = self.subsample_data(
                data["wavelength"],
                data_granularity=data_granularity,
                sample_granularity=1,
                method="linear",
            )

            data["file_num_index"] = self.subsample_data(
                data["file_num_index"],
                sample_granularity=sample_granularity,
                data_granularity=1,
                method="linear",
            )

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Folder not found: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error processing data: {str(e)}")

        return data

    def generate_insitu_uvvis_plot(
        self, file_path: str | Path | None = None, link_plots: bool = False
    ):
        """Generate combined UVVis and electrochemical plots using the operando-style layout.

        This method coordinates the creation of various plot components and combines
        them into a unified visualization.

        Parameters:
            file_path: Path to the zip file containing UVVis and electrochemical data,
                rather than looking up in the database for attached files.

        """

        if not file_path:
            if "file_id" not in self.data:
                raise ValueError("No file set in the DataBlock")
            try:
                from pydatalab.file_utils import get_file_info_by_id

            except ImportError:
                raise RuntimeError(
                    "The `datalab-server[server]` extra must be installed to use this block with a database."
                )

            file_info = get_file_info_by_id(self.data["file_id"], update_if_live=True)
            file_path = Path(file_info["location"])

        if Path(file_path).suffix.lower() not in self.accepted_file_extensions:
            raise ValueError(
                f"Unsupported file extension (must be one of {self.accepted_file_extensions})"
            )

        data = self.process_and_store_data(file_path)

        if (
            self.data.get("uvvis_folder_name") is None
            or self.data.get("echem_folder_name") is None
            or self.data.get("uvvis_reference_folder_name") is None
        ):
            raise ValueError("UV-Vis and Echem folder names must be set in the DataBlock")

        plot_data = prepare_uvvis_plot_data(
            data["2D_data"],
            data["wavelength"],
            data["Time_series_data"],
            data["metadata"],
            data["file_num_index"],
        )

        gp = create_linked_insitu_plots(
            plot_data,
            data["Time_series_data"]["metadata"],
            data["metadata"]["time_range"],
            plotting_label_dict=self.plotting_label_dict,
            link_plots=link_plots,
        )

        try:
            from pydatalab.bokeh_plots import DATALAB_BOKEH_THEME
        except ImportError:
            warnings.warn("datalab-server not installed, using default bokeh theme")
            DATALAB_BOKEH_THEME = None

        self.data["bokeh_plot_data"] = bokeh.embed.json_item(gp, theme=DATALAB_BOKEH_THEME)
