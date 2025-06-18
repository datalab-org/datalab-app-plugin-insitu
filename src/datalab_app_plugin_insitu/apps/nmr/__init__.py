"""NMR module for in situ data."""

from .blocks import InsituBlock
from .nmr_insitu import process_datalab_data, process_local_data

__all__ = ["InsituBlock", "process_local_data", "process_datalab_data"]
