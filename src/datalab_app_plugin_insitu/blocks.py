import os
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from pydatalab.blocks.base import DataBlock

from datalab_app_plugin_insitu._version import __version__

__all__ = ("GenericInSituBlock",)


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
    version: str = __version__

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
            Sorted list of subfolder names, or empty list if file not found or on error.
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
            data: The data to be subsampled.
            sample_granularity: Subsampling step along rows (samples).
            data_granularity: Subsampling step along columns (features).
            method: Subsampling method; currently supports only 'linear' and 'max_pooling'.

        Returns:
            Subsampled data of the same type as input.
        """

        def pooling(mat, ksize, method="max_pooling", pad=True):
            """Non-overlapping pooling on 2D or 3D data.

            <mat>: ndarray, input array to pool.
            <ksize>: tuple of 2, kernel size in (ky, kx).
            <method>: str, 'max for max-pooling,
                        'mean' for mean-pooling.
            <pad>: bool, pad <mat> or not. If no pad, output has size
                n//f, n being <mat> size, f being kernel size.
                if pad, output has size ceil(n/f).

            Return <result>: pooled matrix.
            """
            if mat.ndim < 2:
                mat = mat.reshape(1, -1)
            m, n = mat.shape[:2]
            ky, kx = ksize

            def _ceil(x, y):
                return int(np.ceil(x / float(y)))

            if pad:
                ny = _ceil(m, ky)
                nx = _ceil(n, kx)
                size = (ny * ky, nx * kx) + mat.shape[2:]
                mat_pad = np.full(size, np.nan)
                mat_pad[:m, :n, ...] = mat
            else:
                ny = m // ky
                nx = n // kx
                mat_pad = mat[: ny * ky, : nx * kx, ...]

            new_shape = (ny, ky, nx, kx) + mat.shape[2:]

            if method == "max_pooling":
                result = np.nanmax(mat_pad.reshape(new_shape), axis=(1, 3))
            elif method == "mean_pooling":
                result = np.nanmean(mat_pad.reshape(new_shape), axis=(1, 3))
            else:
                raise ValueError(
                    f"Pooling method '{method}' not supported. Choose 'max' or 'mean'."
                )

            return result

        accepted_methods = ["linear", "max_pooling"]
        if method not in accepted_methods:
            raise ValueError(f"Method '{method}' not supported. Choose from {accepted_methods}.")

        if method == "linear":
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

        elif method == "max_pooling":
            # 2D DataFrame
            if isinstance(data, pd.DataFrame):
                data_array = data.to_numpy()
                pooled_array = pooling(
                    data_array,
                    (sample_granularity, data_granularity),
                    method="max_pooling",
                    pad=True,
                )
                return pd.DataFrame(
                    pooled_array,
                    columns=data.columns[::data_granularity],
                    index=data.index[::sample_granularity],
                )
            # 2D ndarray
            elif isinstance(data, np.ndarray):
                pooled_array = pooling(
                    data, (sample_granularity, data_granularity), method="max_pooling", pad=True
                )
                return pooled_array

            else:
                raise ValueError("Input must be a 2D numpy array or a pandas DataFrame.")
        else:
            raise ValueError(f"Method '{method}' not supported. Choose from {accepted_methods}.")
