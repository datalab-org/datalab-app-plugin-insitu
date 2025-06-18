"""UVVIS module for in situ data."""

from .blocks import UVVisInsituBlock

# from .nmr_insitu import process_datalab_data, process_local_data

__all__ = ["UVVisInsituBlock", "process_local_data", "process_datalab_data"]
