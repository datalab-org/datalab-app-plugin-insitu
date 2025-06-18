import os
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import bokeh.embed
import numpy as np
import pandas as pd
from pydatalab.blocks.base import DataBlock
from pydatalab.bokeh_plots import DATALAB_BOKEH_THEME

from datalab_app_plugin_insitu.nmr_insitu import process_local_data
from datalab_app_plugin_insitu.plotting import create_linked_insitu_plots, prepare_plot_data
from datalab_app_plugin_insitu.uvvis_utils import process_local_uvvis_data


class GenericInSituBlock(DataBlock, ABC):
    """
    Abstract base class for an in-situ data block.
    Manages data loading, processing, parameter handling, and plotting.
    """

    blocktype: str = "generic-insitu"
    name: str = "Generic Data Block"
    description: str = "A base class for in-situ data processing blocks."
    accepted_file_extensions: Tuple[str, ...] = (".zip",)
    defaults: Dict[str, Any] = {}

    @abstractmethod
    def _plot_function(self, file_path: str | Path | None = None, link_plots: bool = False):
        """Subclasses must implement this method to generate the in-situ plot."""
        pass

    def get_available_folders(self, file_path: Path) -> List[str]:
        """
        Extract and return a list of available folders from the zip file.

        This method opens the zip file identified by file_id, extracts the main folder
        and its subfolders, and returns a sorted list of subfolder names.

        Parameters:
            file_path: Path to the zip file.

        Returns:
            List[str]: Sorted list of subfolder names, or empty list if file not found or on error.
        """

        try:
            if not file_path or not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            folders = set()
            with zipfile.ZipFile(file_path, "r") as zip_folder:
                main_folder = zip_folder.namelist()[0].split("/")[0]

                for file in zip_folder.namelist():
                    if file.startswith(main_folder + "/"):
                        sub_path = file[len(main_folder) + 1 :]
                        sub_folder = sub_path.split("/")[0] if "/" in sub_path else None
                        if sub_folder:
                            folders.add(sub_folder)

            folder_list = sorted(list(folders))

            return folder_list
        except Exception as e:
            raise RuntimeError(f"Error getting folders from zip file: {str(e)}")

    @abstractmethod
    def process_and_store_data(self, file_path: str | Path):
        """Subclasses must implement this method to process and store data."""
        # Would like to replace this with a process 2d data mehod and a process time series data method
        pass

    @property
    def plot_functions(self):
        return (lambda: self._plot_function(),)

    @staticmethod
    def subsample_data(
        data: Union[pd.DataFrame, np.ndarray],
        sample_granularity: int,
        data_granularity: int,
        method: str = "linear",
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Subsample data to a specified granularity in both sample and feature dimensions.

        Parameters:
            data: pd.DataFrame or np.ndarray
                The data to be subsampled.
            sample_granularity: int
                Subsampling step along rows (samples).
            data_granularity: int
                Subsampling step along columns (features).
            method: str
                Subsampling method; currently supports only 'linear'.

        Returns:
            Subsampled data of the same type as input.
        """
        if method != "linear":
            raise NotImplementedError(f"Method '{method}' is not implemented.")

        # 1D array case
        if isinstance(data, np.ndarray) and data.ndim == 1:
            return data[::data_granularity]

        # 2D DataFrame
        elif isinstance(data, pd.DataFrame):
            return data.iloc[::sample_granularity, ::data_granularity]

        # 2D ndarray
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            return data[::sample_granularity, ::data_granularity]

        else:
            raise ValueError("Input must be a 1D or 2D numpy array or a pandas DataFrame.")



class InsituBlock(GenericInSituBlock):
    blocktype = "insitu-nmr"
    name = "NMR insitu"
    description = """This datablock processes in situ NMR data from an input .zip file containing two specific directories:

    - **NMR Data Directory**: Contains multiple Bruker in-situ NMR experiment datasets.
    - **Echem Data Directory**: Contains echem data files in `.mpr` format.

    If multiple echem experiments are present, their filenames must include `GCPL`.

    """
    accepted_file_extensions = (".zip",)
    available_folders: List[str] = []
    nmr_folder_name = ""
    echem_folder_name = ""
    folder_name = ""

    defaults = {
        "ppm1": 0.0,
        "ppm2": 0.0,
        "start_exp": 1,
        "exclude_exp": None,
    }

    def _plot_function(self, file_path=None, link_plots=False):
        return self.generate_insitu_nmr_plot(file_path=file_path, link_plots=link_plots)

    def process_and_store_data(self, file_path: str | Path):
        """
        Process insitu NMR and electrochemical data and store results.

        This method validates input parameters, extracts data from the specified folders,
        and stores the processed data in the block's data attribute.

        """
        file_path = Path(file_path)
        folders = self.get_available_folders(file_path)
        self.data["available_folders"] = folders

        nmr_folder_name = self.data.get("nmr_folder_name")
        echem_folder_name = self.data.get("echem_folder_name")

        if not all([nmr_folder_name, echem_folder_name]):
            raise ValueError("Both NMR and Echem folder names are required")

        start_exp = int(self.data.get("start_exp", self.defaults["start_exp"]))
        exclude_exp = self.data.get("exclude_exp", self.defaults["exclude_exp"])

        try:
            result = process_local_data(
                folder_name=str(file_path),
                nmr_folder_name=nmr_folder_name,
                echem_folder_name=echem_folder_name,
                start_at=start_exp,
                exclude_exp=exclude_exp,
            )

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Folder not found: {str(e)}")

        except Exception as e:
            raise RuntimeError(f"Error processing data: {str(e)}")

        nmr_data = result["nmr_spectra"]
        ppm_values = np.array(nmr_data.get("ppm", []))

        ppm1 = self.data["ppm1"] = min(ppm_values)
        ppm2 = self.data["ppm2"] = max(ppm_values)

        self.data.update(
            {
                "nmr_data": result["nmr_spectra"],
                "echem_data": result.get("echem", {}),
                "metadata": result["metadata"],
                "processing_params": {
                    "ppm1": ppm1,
                    "ppm2": ppm2,
                    "file_id": self.data.get("file_id"),
                    "start_exp": start_exp,
                    "exclude_exp": exclude_exp,
                },
            }
        )

    def should_reprocess_data(self) -> bool:
        """
        Determine if data needs to be reprocessed based on parameter changes.
        PPM changes should not trigger reprocessing.
        """
        if "processing_params" not in self.data or "nmr_data" not in self.data:
            return True

        params = self.data["processing_params"]
        current_params = {
            "file_id": self.data.get("file_id"),
            "start_exp": int(self.data.get("start_exp", self.defaults["start_exp"])),
            "exclude_exp": self.data.get("exclude_exp", self.defaults["exclude_exp"]),
        }

        for key in current_params:
            if params.get(key) != current_params[key]:
                return True

        return False

    def generate_insitu_nmr_plot(
        self, file_path: str | Path | None = None, link_plots: bool = False
    ):
        """Generate combined NMR and electrochemical plots using the operando-style layout.

        This method coordinates the creation of various plot components and combines
        them into a unified visualization.

        Parameters:
            file_path: Path to the zip file containing NMR and electrochemical data,
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

        if self.should_reprocess_data():
            self.process_and_store_data(file_path)

        else:
            self.data["processing_params"]["ppm1"] = float(
                self.data.get("ppm1", self.defaults["ppm1"])
            )
            self.data["processing_params"]["ppm2"] = float(
                self.data.get("ppm2", self.defaults["ppm2"])
            )

        plot_data = prepare_plot_data(
            self.data["nmr_data"], self.data["echem_data"], self.data["metadata"]
        )

        ppm1 = float(self.data.get("ppm1", self.defaults["ppm1"]))
        ppm2 = float(self.data.get("ppm2", self.defaults["ppm2"]))

        gp = create_linked_insitu_plots(plot_data, ppm_range=(ppm1, ppm2), link_plots=link_plots)
        self.data["bokeh_plot_data"] = bokeh.embed.json_item(gp, theme=DATALAB_BOKEH_THEME)


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
    }

    def _plot_function(self, file_path=None, link_plots=False):
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
        """Generate combined NMR and electrochemical plots using the operando-style layout.

        This method coordinates the creation of various plot components and combines
        them into a unified visualization.

        Parameters:
            file_path: Path to the zip file containing NMR and electrochemical data,
                rather than looking up in the database for attached files.

        """
        from datalab_app_plugin_insitu.plotting_uvvis import (
            create_linked_insitu_plots,
            prepare_uvvis_plot_data,
        )

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
        if num_samples > 1000:
            sample_granularity = num_samples // 1000
        else:
            sample_granularity = 1
        if data_length > 1000:
            data_granularity = data_length // 1000
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
