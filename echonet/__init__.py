"""
The echonet package contains code for loading echocardiogram videos, and
functions for training and testing segmentation and ejection fraction
prediction models.
"""

from echonet.__version__ import __version__
from echonet.config import CONFIG as config
import echonet.datasets as datasets
import echonet.utils as utils

__all__ = ["__version__", "config", "datasets", "utils"]
