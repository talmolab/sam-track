"""Output writers for tracking results.

This module provides writers for persisting tracking results to disk in
various formats:

- BBoxWriter: JSON bounding box tracks with metadata
- SegmentationWriter: HDF5 segmentation masks with compression
- SLPWriter: SLEAP SLP files with track assignments
"""

from .bbox import BBoxWriter
from .seg import SegmentationWriter
from .slp import SLPWriter

__all__ = ["BBoxWriter", "SegmentationWriter", "SLPWriter"]
