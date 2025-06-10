"""Datalab plugin for In Situ NMR"""

from importlib.metadata import version

from .blocks import InsituBlock, UVVisInsituBlock
from .nmr_insitu import process_datalab_data, process_local_data

__version__ = version("datalab-app-plugin-insitu")

__all__ = ("__version__", "process_local_data", "process_datalab_data", "InsituBlock", "UVVisInsituBlock")
