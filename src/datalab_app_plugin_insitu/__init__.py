"""Datalab plugin for In Situ NMR"""

from importlib.metadata import version
from .nmr_insitu import process_local_data, process_datalab_data

__version__ = version("datalab-app-plugin-insitu")

__all__ = ("process_local_data", "process_datalab_data")
