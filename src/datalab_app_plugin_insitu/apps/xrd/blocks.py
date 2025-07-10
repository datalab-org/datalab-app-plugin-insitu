from pathlib import Path
from typing import List

import bokeh.embed
from pydatalab.bokeh_plots import DATALAB_BOKEH_THEME

from datalab_app_plugin_insitu.apps.xrd.xrd_utils import process_local_xrd_data
from datalab_app_plugin_insitu.blocks import GenericInSituBlock
from datalab_app_plugin_insitu.plotting_uvvis import (
    create_linked_insitu_plots,
    prepare_xrd_plot_data,
)


class XRDInsituBlock(GenericInSituBlock):
    blocktype = "insitu-xrd"
    name = "XRD insitu"
    description = """This datablock processes in situ XRD data from an input .zip file containing two specific directories:

    - **XRD Data Directory**: Contains multiple XRD in-situ experiment datasets.
    - **Time series folder directory**: Contains echem data (.txt.) or temperature data files (.csv).
    """
    accepted_file_extensions = (".zip",)
    available_folders: List[str] = []
    xrd_folder_name = None
    time_series_folder_name = None
    folder_name = None
    plotting_label_dict = {
        "x_axis_label" : "Two theta (degrees)",
        "time_series_y_axis_label" : "Experiment number",
        "line_y_axis_label" : "Intensity (a.u.)",
        "time_series_x_axis_label" : "Temp (C)",
        "label_source": {
            "label_template" : "Exp num {exp_num} | Temp = {temperature} K",
            "label_field_map" : {
                "exp_num": "exp_num",
                "temperature": "voltages_by_exp",
            }
        },
    }

    defaults = {
        "start_exp": 1,
        "exclude_exp": None,
        "metadata": None,
        "target_sample_number": 1000,
        "target_data_number": 1000,
        "data_granularity": None,
        "sample_granularity": None,
    }

    def _plot_function(self, file_path=None, link_plots=True):
        return self.generate_insitu_xrd_plot(file_path=file_path, link_plots=link_plots)

    def process_and_store_data(self, file_path: str | Path):
        """
        Process all in situ XRD and electrochemical data and store results.
        This method is a wrapper for processing both XRD and electrochemical data.
        """
        file_path = Path(file_path)
        folders = self.get_available_folders(file_path)
        self.data["available_folders"] = folders

        xrd_folder_name = Path(self.data.get("xrd_folder_name"))
        if not xrd_folder_name:
            raise ValueError("XRD folder name is required")

        time_series_folder_name = Path(self.data.get("time_series_folder_name"))
        if not time_series_folder_name:
            raise ValueError("Log or echem folder name is required")

        start_exp = int(self.data.get("start_exp", self.defaults["start_exp"]))
        exclude_exp = self.data.get("exclude_exp", self.defaults["exclude_exp"])

        try:
            data = process_local_xrd_data(
                file_path=file_path,
                xrd_folder_name=xrd_folder_name,
                log_folder_name=time_series_folder_name,
                start_exp=start_exp,
                exclude_exp=exclude_exp,
                # Needs to be made more generic
            )

            num_samples, data_length = data["2D_data"].shape
            print(f"Number of samples: {num_samples}, Data length: {data_length}")

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
            # Subsample the 2D data and wavelength data to a maximum of 1000 samples
            data["2D_data"] = self.subsample_data(
                data["2D_data"],
                sample_granularity=sample_granularity,
                data_granularity=data_granularity,
                method="linear",
            )

            data["Two theta"] = self.subsample_data(
                data["Two theta"],
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
            print(f"Subsampled 2D data shape: {data['2D_data'].shape}")
            print(f"Subsampled Two theta data shape: {data['Two theta'].shape}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Folder not found: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error processing data: {str(e)}")

        return data

    def generate_insitu_xrd_plot(
        self, file_path: str | Path | None = None, link_plots: bool = False
    ):
        """Generate combined XRD and electrochemical plots using the operando-style layout.

        This method coordinates the creation of various plot components and combines
        them into a unified visualization.

        Parameters:
            file_path: Path to the zip file containing XRD and electrochemical data,
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

        plot_data = prepare_xrd_plot_data(
            two_d_data=data["2D_data"],
            heatmap_x_values=data["Two theta"],
            time_series_data=data["Time_series_data"],
            metadata=data["metadata"],
            file_num_index=data["file_num_index"],
        )

        gp = create_linked_insitu_plots(
            plot_data,
            time_series_time_range=data["Time_series_data"]["metadata"],
            heatmap_time_range=data["metadata"]["y_range"],
            plotting_label_dict=self.plotting_label_dict,
            link_plots=link_plots,
        )
        self.data["bokeh_plot_data"] = bokeh.embed.json_item(gp, theme=DATALAB_BOKEH_THEME)
