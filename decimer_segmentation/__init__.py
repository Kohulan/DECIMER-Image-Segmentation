# -*- coding: utf-8 -*-

"""
DECIMER Segmentation Python Package.

Segmentation of chemical structure depictions from scientific literature.


For comments, bug reports or feature requests,
please raise an issue on our GitHub repository.
"""

__version__ = "1.6.0"

__all__ = [
    "decimer_segmentation",
]

from .decimer_segmentation import *
from .complete_structure import *


def __getattr__(name):
    """Keep ``decimer_segmentation.segmentation_model`` working without loading it on import."""
    if name == "segmentation_model":
        from .decimer_segmentation import get_model

        return get_model()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
