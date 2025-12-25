"""HDF5 segmentation mask input reader.

This module provides a reader for loading segmentation masks from HDF5 format
files written by SegmentationWriter.

HDF5 Structure (expected):
    /metadata/          - Group with metadata attributes
    /masks              - (T, N, H, W) uint8 dataset
    /frame_indices      - (T,) int32 dataset mapping row to frame index
    /track_ids          - (T, N) int32 dataset with object IDs per frame
    /confidences        - (T, N) float32 dataset with scores per frame
    /num_objects        - (T,) int32 dataset with object count per frame
    /tracks/            - Track metadata
        /track_N/       - Group for each track with name, first_frame, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    pass


class SegmentationReader:
    """Reader for segmentation masks in HDF5 format.

    This class reads tracking results from an HDF5 file written by
    SegmentationWriter. It provides lazy loading and efficient frame-based
    access to masks and track information.

    Example:
        >>> reader = SegmentationReader("output.seg.h5")
        >>> print(f"Frames: {reader.num_frames}, Tracks: {reader.num_tracks}")
        >>> for frame_idx in reader.labeled_frame_indices:
        ...     masks = reader.get_masks(frame_idx)
        ...     track_ids = reader.get_track_ids(frame_idx)
        ...     print(f"Frame {frame_idx}: {len(track_ids)} objects")

        # Context manager usage
        >>> with SegmentationReader("output.seg.h5") as reader:
        ...     masks = reader.get_masks(0)

    Attributes:
        path: Path to the HDF5 file.
    """

    def __init__(self, path: str | Path):
        """Initialize the segmentation reader.

        Args:
            path: Path to the HDF5 segmentation file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"Segmentation file not found: {self.path}")

        self._file: h5py.File | None = None
        self._frame_map: dict[int, int] | None = None
        self._track_names: dict[int, str] | None = None

    def _open(self) -> h5py.File:
        """Open and return the HDF5 file handle (lazy)."""
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def _build_frame_map(self) -> dict[int, int]:
        """Build frame index -> dataset row mapping.

        Returns:
            Dictionary mapping video frame indices to dataset row indices.
        """
        if self._frame_map is None:
            f = self._open()
            frame_indices = f["frame_indices"][:]
            self._frame_map = {int(fi): row for row, fi in enumerate(frame_indices)}
        return self._frame_map

    # =========================================================================
    # Metadata properties
    # =========================================================================

    @property
    def version(self) -> str:
        """File format version."""
        return self._open()["metadata"].attrs.get("version", "unknown")

    @property
    def format(self) -> str:
        """File format identifier."""
        return self._open()["metadata"].attrs.get("format", "unknown")

    @property
    def video_source(self) -> str:
        """Original video path."""
        return self._open()["metadata"].attrs.get("video_source", "")

    @property
    def width(self) -> int:
        """Video/mask width in pixels."""
        return int(self._open()["metadata"].attrs.get("width", 0))

    @property
    def height(self) -> int:
        """Video/mask height in pixels."""
        return int(self._open()["metadata"].attrs.get("height", 0))

    @property
    def resolution(self) -> tuple[int, int]:
        """Video resolution as (width, height)."""
        return (self.width, self.height)

    @property
    def fps(self) -> float:
        """Video frames per second."""
        return float(self._open()["metadata"].attrs.get("fps", 0.0))

    @property
    def num_frames(self) -> int:
        """Number of frames with mask data."""
        return int(self._open()["metadata"].attrs.get("frames_written", 0))

    @property
    def num_tracks(self) -> int:
        """Number of unique tracks."""
        return int(self._open()["metadata"].attrs.get("total_tracks", 0))

    @property
    def max_objects(self) -> int:
        """Maximum objects per frame (dataset dimension)."""
        return int(self._open()["metadata"].attrs.get("max_objects", 0))

    # =========================================================================
    # Frame access
    # =========================================================================

    @property
    def labeled_frame_indices(self) -> list[int]:
        """Sorted list of frame indices that have mask data."""
        return sorted(self._build_frame_map().keys())

    def has_frame(self, frame_idx: int) -> bool:
        """Check if a frame has mask data.

        Args:
            frame_idx: Video frame index.

        Returns:
            True if the frame has mask data, False otherwise.
        """
        return frame_idx in self._build_frame_map()

    def get_num_objects(self, frame_idx: int) -> int:
        """Get number of objects in a frame.

        Args:
            frame_idx: Video frame index.

        Returns:
            Number of objects in the frame.

        Raises:
            KeyError: If frame_idx is not in the dataset.
        """
        frame_map = self._build_frame_map()
        if frame_idx not in frame_map:
            raise KeyError(f"Frame {frame_idx} not in segmentation file")

        row = frame_map[frame_idx]
        return int(self._open()["num_objects"][row])

    def get_masks(self, frame_idx: int) -> np.ndarray:
        """Get binary masks for a frame.

        Args:
            frame_idx: Video frame index.

        Returns:
            Binary masks as (N, H, W) uint8 array where N is num_objects.
            Each mask is 0/1 valued.

        Raises:
            KeyError: If frame_idx is not in the dataset.
        """
        frame_map = self._build_frame_map()
        if frame_idx not in frame_map:
            raise KeyError(f"Frame {frame_idx} not in segmentation file")

        row = frame_map[frame_idx]
        f = self._open()
        n_objects = int(f["num_objects"][row])

        if n_objects == 0:
            # Return empty array with correct shape
            return np.empty((0, self.height, self.width), dtype=np.uint8)

        return f["masks"][row, :n_objects]

    def get_track_ids(self, frame_idx: int) -> np.ndarray:
        """Get track IDs for objects in a frame.

        Args:
            frame_idx: Video frame index.

        Returns:
            Track IDs as (N,) int32 array where N is num_objects.

        Raises:
            KeyError: If frame_idx is not in the dataset.
        """
        frame_map = self._build_frame_map()
        if frame_idx not in frame_map:
            raise KeyError(f"Frame {frame_idx} not in segmentation file")

        row = frame_map[frame_idx]
        f = self._open()
        n_objects = int(f["num_objects"][row])

        if n_objects == 0:
            return np.empty((0,), dtype=np.int32)

        return f["track_ids"][row, :n_objects]

    def get_confidences(self, frame_idx: int) -> np.ndarray:
        """Get confidence scores for objects in a frame.

        Args:
            frame_idx: Video frame index.

        Returns:
            Confidence scores as (N,) float32 array where N is num_objects.

        Raises:
            KeyError: If frame_idx is not in the dataset.
        """
        frame_map = self._build_frame_map()
        if frame_idx not in frame_map:
            raise KeyError(f"Frame {frame_idx} not in segmentation file")

        row = frame_map[frame_idx]
        f = self._open()
        n_objects = int(f["num_objects"][row])

        if n_objects == 0:
            return np.empty((0,), dtype=np.float32)

        return f["confidences"][row, :n_objects]

    # =========================================================================
    # Track metadata
    # =========================================================================

    def get_track_names(self) -> dict[int, str]:
        """Get mapping of track IDs to names.

        Returns:
            Dictionary mapping track_id (int) to name (str).
        """
        if self._track_names is None:
            f = self._open()
            self._track_names = {}

            if "tracks" in f:
                for track_group in f["tracks"].values():
                    tid = int(track_group.attrs["track_id"])
                    name = track_group.attrs.get("name", f"object_{tid}")
                    self._track_names[tid] = name

        return self._track_names.copy()

    def get_track_info(self, track_id: int) -> dict:
        """Get metadata for a specific track.

        Args:
            track_id: The track ID to look up.

        Returns:
            Dictionary with track metadata (first_frame, last_frame,
            num_frames, avg_confidence, name).

        Raises:
            KeyError: If track_id is not found.
        """
        f = self._open()
        group_name = f"tracks/track_{track_id}"

        if group_name not in f:
            raise KeyError(f"Track {track_id} not found in segmentation file")

        track_group = f[group_name]
        return {
            "track_id": int(track_group.attrs["track_id"]),
            "name": track_group.attrs.get("name", f"object_{track_id}"),
            "first_frame": int(track_group.attrs.get("first_frame", 0)),
            "last_frame": int(track_group.attrs.get("last_frame", 0)),
            "num_frames": int(track_group.attrs.get("num_frames", 0)),
            "avg_confidence": float(track_group.attrs.get("avg_confidence", 0.0)),
        }

    # =========================================================================
    # Context manager and cleanup
    # =========================================================================

    def close(self) -> None:
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> SegmentationReader:
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit - close file."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensure file is closed."""
        if hasattr(self, "_file"):
            self.close()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SegmentationReader(path={self.path!r}, "
            f"num_frames={self.num_frames}, num_tracks={self.num_tracks})"
        )
