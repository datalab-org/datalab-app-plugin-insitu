from pathlib import Path
from typing import List

import bokeh.embed
from pydatalab.bokeh_plots import DATALAB_BOKEH_THEME

from datalab_app_plugin_insitu.apps.uvvis.uvvis_utils import process_local_uvvis_data
from datalab_app_plugin_insitu.blocks import GenericInSituBlock
from datalab_app_plugin_insitu.plotting_uvvis import (
    create_linked_insitu_plots,
    prepare_uvvis_plot_data,
)


class UVVisInsituBlock(GenericInSituBlock):
    blocktype = "insitu-uvvis"
    name = "UV-Vis insitu"
    description = """This datablock processes in situ UV-Vis data from an input .zip file containing two specific directories:

    - **UV-Vis Data Directory**: Contains multiple UV-Vis in-situ experiment datasets.
    - **Echem Data Directory**: Contains echem data files in `.mpr` format.

    If multiple echem experiments are present, their filenames must include `GCPL`.

    """
    accepted_file_extensions = (".zip",)
    available_folders: List[str] = []
    uvvis_folder_name = None
    echem_folder_name = None
    folder_name = None

    defaults = {
        "start_exp": 1,
        "exclude_exp": None,
        "echem_data": None,
        "metadata": None,
        "target_sample_number": 1000,
        "target_data_number": 1000,
    }

    def _plot_function(self, file_path=None, link_plots=True):
        return self.generate_insitu_uvvis_plot(file_path=file_path, link_plots=link_plots)

    def process_and_store_data(self, file_path: str | Path):
        """
        Process all in situ UV-Vis and electrochemical data and store results.
        This method is a wrapper for processing both UV-Vis and electrochemical data.
        """
        file_path = Path(file_path)
        folders = self.get_available_folders(file_path)
        self.data["available_folders"] = folders

        uvvis_folder_name = Path(self.data.get("uvvis_folder_name"))
        if not uvvis_folder_name:
            raise ValueError("UV-Vis folder name is required")
        reference_folder_name = Path(self.data.get("uvvis_reference_folder_name"))
        if not reference_folder_name:
            raise ValueError("UV-Vis folder name is required")
        echem_folder_name = Path(self.data.get("echem_folder_name"))
        if not echem_folder_name:
            raise ValueError("Echem folder name is required")

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
                # Needs to be made more generic
                sample_file_extension=".Raw8.txt",
                reference_file_extension=".Raw8.TXT",
                scan_time=60.15,
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

        num_samples, data_length = data["2D_data"].shape
        print(f"Number of samples: {num_samples}, Data length: {data_length}")
        if num_samples > self.data.get("target_sample_number"):
            sample_granularity = num_samples // self.data.get("target_sample_number")
        else:
            sample_granularity = 1
        if data_length > self.data.get("target_data_number"):
            data_granularity = data_length // self.data.get("target_data_number")
        else:
            data_granularity = 1

        # Subsample the 2D data and wavelength data to a maximum of 1000 samples
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
        print(f"Subsampled 2D data shape: {data['2D_data'].shape}")
        print(f"Subsampled wavelength data shape: {data['wavelength'].shape}")

        plot_data = prepare_uvvis_plot_data(
            data["2D_data"],
            data["wavelength"],
            data["Time_series_data"],
            data["metadata"],
        )

        gp = create_linked_insitu_plots(
            plot_data,
            data["Time_series_data"]["metadata"],
            data["metadata"]["time_range"],
            link_plots=link_plots,
        )
        self.data["bokeh_plot_data"] = bokeh.embed.json_item(gp, theme=DATALAB_BOKEH_THEME)
