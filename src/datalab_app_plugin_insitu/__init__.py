"""datalab plugin for in situ measurements."""

from ._version import __version__
from .apps.nmr import InsituBlock, process_datalab_data, process_local_data
from .apps.uvvis import UVVisInsituBlock

__all__ = (
    "__version__",
    "process_local_data",
    "process_datalab_data",
    "InsituBlock",
    "UVVisInsituBlock",
)
