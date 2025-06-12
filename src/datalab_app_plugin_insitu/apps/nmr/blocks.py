import os
import zipfile
from pathlib import Path
from typing import List

import numpy as np
from pydatalab.blocks.base import DataBlock
from pydatalab.bokeh_plots import DATALAB_BOKEH_THEME
from pydatalab.file_utils import get_file_info_by_id

from datalab_app_plugin_insitu.plotting import create_linked_insitu_plots, prepare_plot_data

from .nmr_insitu import process_local_data

__all__ = ("InsituBlock",)


class InsituBlock(DataBlock):
    """This datablock processes in situ NMR data from an input .zip file containing two specific directories:

    - NMR Data Directory: Contains multiple Bruker in-situ NMR experiment datasets.
    - Echem Data Directory: Contains echem data files in `.mpr` format.

    If multiple echem experiments are present, their filenames must include `GCPL`.

    """

    blocktype = "insitu-nmr"
    name = "NMR insitu"
    description = __doc__

    accepted_file_extensions = (".zip",)
    available_folders: List[str] = []
    nmr_folder_name = ""
    echem_folder_name = ""
    folder_name = ""

    defaults = {
        "ppm1": 0.0,
        "ppm2": 0.0,
        "start_exp": 1,
        "end_exp": None,
        "step_exp": 1,
        "exclude_exp": None,
    }

    @property
    def plot_functions(self):
        return (self.generate_insitu_nmr_plot,)

    def get_available_folders(self, file_path: Path) -> List[str]:
        """
        Extract and return a list of available folders from the zip file.

        This method opens the zip file identified by file_id, extracts the main folder
        and its subfolders, and returns a sorted list of subfolder names.

        Parameters:
            file_path: Path to the zip file from which to extract folder names.

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

    def process_and_store_data(self, file_path: str | Path):
        """Process insitu NMR and electrochemical data and store results.

        This method validates input parameters, extracts data from the specified folders,
        and stores the processed data in the block's data attribute.

        """
        file_path = Path(file_path)
        folders = self.get_available_folders(file_path)
        self.data["available_folders"] = folders

        nmr_folder_name = self.data.get("nmr_folder_name")
        echem_folder_name = self.data.get("echem_folder_name")

        file_path = get_file_info_by_id(self.data["file_id"])["location"]

        if not all([nmr_folder_name, echem_folder_name]):
            raise ValueError("Both NMR and Echem folder names must be specified")

        start_exp = int(self.data.get("start_exp", self.defaults["start_exp"]))
        end_exp = self.data.get("end_exp", self.defaults["end_exp"])
        if end_exp is not None:
            end_exp = int(end_exp)
        step_exp = int(self.data.get("step_exp", self.defaults["step_exp"]))
        exclude_exp = self.data.get("exclude_exp", self.defaults["exclude_exp"])

        try:
            result = process_local_data(
                folder_name=str(file_path),
                nmr_folder_name=nmr_folder_name,
                echem_folder_name=echem_folder_name,
                start_at=start_exp,
                end_at=end_exp,
                step=step_exp,
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
                    "end_exp": end_exp,
                    "step_exp": step_exp,
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
            "end_exp": self.data.get("end_exp", self.defaults["end_exp"]),
            "step_exp": int(self.data.get("step_exp", self.defaults["step_exp"])),
            "exclude_exp": self.data.get("exclude_exp", self.defaults["exclude_exp"]),
        }

        if current_params["end_exp"] is not None:
            current_params["end_exp"] = int(current_params["end_exp"])

        for key in current_params:
            if params.get(key) != current_params[key]:
                return True

        return False

    def generate_insitu_nmr_plot(self, file_path: str | Path | None = None):
        """
        Generate combined NMR and electrochemical plots using the operando-style layout.

        This method coordinates the creation of various plot components and combines
        them into a unified visualization.

        Parameters:
            file_path: Optional path to the zip file containing NMR and echem data,
                removing the need to lookup in a database.

        """
        if not file_path:
            if "file_id" not in self.data:
                raise ValueError("No file set in the DataBlock")

            try:
                from pydatalab.file_utils import get_file_info_by_id

                file_info = get_file_info_by_id(self.data["file_id"], update_if_live=True)
                if Path(file_info["location"]).suffix.lower() not in self.accepted_file_extensions:
                    raise ValueError(
                        f"Unsupported file extension (must be one of {self.accepted_file_extensions})"
                    )

                file_path = Path(file_info["location"])
            except Exception:
                raise RuntimeError("Failed to retrieve file information. Please check the file ID.")

        try:
            needs_reprocessing = self.should_reprocess_data()
            if needs_reprocessing:
                self.process_and_store_data(file_path)
            else:
                self.data["processing_params"]["ppm1"] = float(
                    self.data.get("ppm1", self.defaults["ppm1"])
                )
                self.data["processing_params"]["ppm2"] = float(
                    self.data.get("ppm2", self.defaults["ppm2"])
                )

            if "nmr_data" not in self.data:
                raise ValueError("No NMR data available after processing")

            plot_data = prepare_plot_data(
                self.data["nmr_data"], self.data["echem_data"], self.data["metadata"]
            )

            ppm1 = float(self.data.get("ppm1", self.defaults["ppm1"]))
            ppm2 = float(self.data.get("ppm2", self.defaults["ppm2"]))

            gp = create_linked_insitu_plots(plot_data, ppm_range=(ppm1, ppm2))

            self.data["bokeh_plot_data"] = bokeh.embed.json_item(gp, theme=DATALAB_BOKEH_THEME)
