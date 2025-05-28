"""Datalab plugin for In Situ"""

from importlib.metadata import version

from .apps.nmr import InsituBlock, process_datalab_data, process_local_data

__version__ = version("datalab-app-plugin-insitu")

__all__ = ("__version__", "process_local_data", "process_datalab_data", "InsituBlock")
