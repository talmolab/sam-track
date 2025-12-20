"""Output writers for tracking results.

This module provides writers for persisting tracking results to disk in
various formats:

- BBoxWriter: JSON bounding box tracks with metadata
- SegmentationWriter: HDF5 segmentation masks with compression
"""

from .bbox import BBoxWriter
from .seg import SegmentationWriter

__all__ = ["BBoxWriter", "SegmentationWriter"]
