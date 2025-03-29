"""Datalab plugin for In Situ NMR"""

from importlib.metadata import version
from .nmr_insitu import process_local_data, process_datalab_data
from .blocks import InsituBlock

__version__ = version("datalab-app-plugin-insitu")

__all__ = ("__version__", "process_local_data", "process_datalab_data", "InsituBlock")
