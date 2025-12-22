"""HDF5 segmentation mask output writer.

This module provides a writer for saving segmentation masks to HDF5 format
with compression. Masks are stored as a single contiguous dataset for
efficient storage and access.

HDF5 Structure:
    /metadata/          - Group with metadata attributes
    /masks              - (T, N, H, W) uint8 dataset, compressed
    /frame_indices      - (T,) int32 dataset mapping row to frame index
    /track_ids          - (T, N) int32 dataset with object IDs per frame
    /confidences        - (T, N) float32 dataset with scores per frame
    /tracks/            - Track metadata
        /track_0/       - Group for track 0
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from ..tracker import TrackingResult


class SegmentationWriter:
    """Writer for segmentation masks in HDF5 format.

    This class writes tracking results to an HDF5 file with compressed masks.
    All masks are stored in a single dataset that grows as frames are added.

    The HDF5 file uses:
    - GZIP level 1 compression (fast with good ratio for binary masks)
    - Shuffle filter for better compression of sparse data
    - Single resizable dataset for all masks

    Example:
        >>> with SegmentationWriter(
        ...     output_path="output.seg.h5",
        ...     video_path="video.mp4",
        ...     video_width=1920,
        ...     video_height=1080,
        ...     max_objects=5,
        ... ) as writer:
        ...     for result in tracker.propagate():
        ...         writer.add_result(result)
        # File is automatically finalized on exit

    Attributes:
        output_path: Path to the output HDF5 file.
    """

    def __init__(
        self,
        output_path: str | Path,
        video_path: str | Path,
        video_width: int,
        video_height: int,
        max_objects: int = 10,
        fps: float | None = None,
        total_frames: int = 0,
        obj_names: dict[int, str] | None = None,
    ):
        """Initialize the segmentation writer.

        Args:
            output_path: Path for the output HDF5 file.
            video_path: Path to the source video file.
            video_width: Video width in pixels.
            video_height: Video height in pixels.
            max_objects: Maximum number of objects per frame.
            fps: Video frames per second (optional).
            total_frames: Total number of frames in the video (for pre-allocation).
            obj_names: Optional mapping from object ID to name.
        """
        self.output_path = Path(output_path)
        self._obj_names = obj_names or {}
        self._max_objects = max_objects
        self._video_width = video_width
        self._video_height = video_height

        # Track info accumulated during writing
        self._track_info: dict[int, dict] = {}
        self._frames_written = 0

        # Ensure parent directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create HDF5 file
        self._h5file = h5py.File(self.output_path, "w")

        # Write metadata group
        meta = self._h5file.create_group("metadata")
        meta.attrs["version"] = "1.0"
        meta.attrs["format"] = "sam-track-seg"
        meta.attrs["video_source"] = str(video_path)
        meta.attrs["width"] = video_width
        meta.attrs["height"] = video_height
        meta.attrs["fps"] = fps if fps is not None else 0.0
        meta.attrs["total_frames"] = total_frames
        meta.attrs["max_objects"] = max_objects
        meta.attrs["mask_encoding"] = "dense"
        meta.attrs["compression"] = "gzip"
        meta.attrs["compression_level"] = 1
        meta.attrs["creation_time"] = datetime.now().isoformat()

        # Create tracks group
        self._tracks_group = self._h5file.create_group("tracks")

        # Pre-allocate or create resizable datasets
        # Use total_frames if known, otherwise start small and resize
        initial_frames = total_frames if total_frames > 0 else 100

        # Masks dataset: (T, N, H, W) uint8
        self._masks_dset = self._h5file.create_dataset(
            "masks",
            shape=(initial_frames, max_objects, video_height, video_width),
            maxshape=(None, max_objects, video_height, video_width),
            dtype=np.uint8,
            chunks=(1, max_objects, video_height, video_width),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )

        # Frame indices: maps dataset row to actual frame index
        self._frame_indices_dset = self._h5file.create_dataset(
            "frame_indices",
            shape=(initial_frames,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(min(1000, initial_frames),),
        )

        # Track IDs per frame: (T, N) int32, -1 for unused slots
        self._track_ids_dset = self._h5file.create_dataset(
            "track_ids",
            shape=(initial_frames, max_objects),
            maxshape=(None, max_objects),
            dtype=np.int32,
            chunks=(min(1000, initial_frames), max_objects),
            fillvalue=-1,
        )

        # Confidences per frame: (T, N) float32
        self._confidences_dset = self._h5file.create_dataset(
            "confidences",
            shape=(initial_frames, max_objects),
            maxshape=(None, max_objects),
            dtype=np.float32,
            chunks=(min(1000, initial_frames), max_objects),
        )

        # Number of objects per frame: (T,) int32
        self._num_objects_dset = self._h5file.create_dataset(
            "num_objects",
            shape=(initial_frames,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(min(1000, initial_frames),),
        )

    def _ensure_capacity(self, needed_rows: int) -> None:
        """Ensure datasets have capacity for the needed number of rows."""
        current_size = self._masks_dset.shape[0]
        if needed_rows > current_size:
            # Double the size or add needed rows, whichever is larger
            new_size = max(current_size * 2, needed_rows)
            self._masks_dset.resize(new_size, axis=0)
            self._frame_indices_dset.resize(new_size, axis=0)
            self._track_ids_dset.resize(new_size, axis=0)
            self._confidences_dset.resize(new_size, axis=0)
            self._num_objects_dset.resize(new_size, axis=0)

    def add_result(self, result: TrackingResult) -> None:
        """Add masks from a tracking result.

        Args:
            result: TrackingResult from the tracker containing frame data.
        """
        row_idx = self._frames_written
        self._ensure_capacity(row_idx + 1)

        # Store frame index
        self._frame_indices_dset[row_idx] = result.frame_idx

        # Store number of objects
        num_objects = min(result.num_objects, self._max_objects)
        self._num_objects_dset[row_idx] = num_objects

        if num_objects > 0:
            # Store masks (pad/truncate to max_objects)
            # Squeeze extra dimensions (SAM3 returns shape (N, 1, H, W))
            masks = np.squeeze(result.masks[:num_objects]).astype(np.uint8)
            # Handle single object case where squeeze removes too much
            if masks.ndim == 2:
                masks = masks[np.newaxis, ...]
            self._masks_dset[row_idx, :num_objects] = masks

            # Store track IDs
            track_ids = result.object_ids[:num_objects].astype(np.int32)
            self._track_ids_dset[row_idx, :num_objects] = track_ids

            # Store confidences
            confidences = result.scores[:num_objects].astype(np.float32)
            self._confidences_dset[row_idx, :num_objects] = confidences

            # Update track info
            for i in range(num_objects):
                tid = int(track_ids[i])
                confidence = float(confidences[i])

                if tid not in self._track_info:
                    self._track_info[tid] = {
                        "first_frame": result.frame_idx,
                        "last_frame": result.frame_idx,
                        "num_frames": 0,
                        "total_confidence": 0.0,
                    }

                self._track_info[tid]["last_frame"] = result.frame_idx
                self._track_info[tid]["num_frames"] += 1
                self._track_info[tid]["total_confidence"] += confidence

        self._frames_written += 1

    def apply_track_name_mapping(
        self,
        mapping: dict[int, str],
    ) -> None:
        """Apply a SAM3 obj_id -> track_name mapping to all tracks.

        This updates the internal name mapping used when finalizing the file.
        Must be called before finalize().

        Args:
            mapping: Dictionary mapping SAM3 obj_id to track name.
        """
        self._obj_names.update(mapping)

    def finalize(self) -> None:
        """Finalize the HDF5 file with track metadata and close it."""
        if self._h5file is None:
            return

        # Trim datasets to actual size
        if self._frames_written < self._masks_dset.shape[0]:
            self._masks_dset.resize(self._frames_written, axis=0)
            self._frame_indices_dset.resize(self._frames_written, axis=0)
            self._track_ids_dset.resize(self._frames_written, axis=0)
            self._confidences_dset.resize(self._frames_written, axis=0)
            self._num_objects_dset.resize(self._frames_written, axis=0)

        # Write track metadata
        for tid, info in self._track_info.items():
            track_group = self._tracks_group.create_group(f"track_{tid}")
            track_group.attrs["track_id"] = tid
            track_group.attrs["name"] = self._obj_names.get(tid, f"object_{tid}")
            track_group.attrs["first_frame"] = info["first_frame"]
            track_group.attrs["last_frame"] = info["last_frame"]
            track_group.attrs["num_frames"] = info["num_frames"]
            track_group.attrs["avg_confidence"] = (
                info["total_confidence"] / info["num_frames"]
                if info["num_frames"] > 0
                else 0.0
            )

        # Update metadata with final counts
        self._h5file["metadata"].attrs["frames_written"] = self._frames_written
        self._h5file["metadata"].attrs["total_tracks"] = len(self._track_info)

        # Close file
        self._h5file.close()
        self._h5file = None

    def close(self) -> None:
        """Close the writer and finalize the file."""
        self.finalize()

    def __enter__(self) -> SegmentationWriter:
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit - finalize and close."""
        self.finalize()

    def __del__(self) -> None:
        """Destructor - ensure file is closed."""
        if hasattr(self, "_h5file") and self._h5file is not None:
            try:
                self.finalize()
            except Exception:
                pass
