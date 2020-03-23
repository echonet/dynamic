"""
The echonet.datasets submodule defines a Pytorch dataset for loading
echocardiogram videos.
"""

from .echo import Echo

__all__ = ["Echo"]
