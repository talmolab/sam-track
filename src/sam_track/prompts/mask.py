"""Mask prompt handler for seg.h5 files."""

from pathlib import Path

import numpy as np

from sam_track.inputs.seg import SegmentationReader
from sam_track.prompts.base import Prompt, PromptHandler, PromptType


class MaskPromptHandler(PromptHandler):
    """Handler for mask prompts from seg.h5 files.

    This handler loads segmentation masks from HDF5 files written by
    SegmentationWriter and converts them to mask prompts for SAM3 tracking.
    This enables iterative refinement workflows where previous tracking
    results (possibly manually corrected) are used to initialize new tracking.

    The handler supports:
    - Automatic frame selection (first labeled frame) or specific frame
    - Multi-frame access via get_prompt() and labeled_frame_indices
    - Track-based object IDs and names from the source file
    - Resolution validation against target video

    Example:
        >>> handler = MaskPromptHandler("path/to/masks.seg.h5")
        >>> prompt = handler.load()
        >>> print(f"Loaded {prompt.num_objects} objects")
        >>> for obj_id in prompt.obj_ids:
        ...     print(f"  {obj_id}: {prompt.get_name(obj_id)}")

        # Multi-frame access
        >>> handler = MaskPromptHandler("masks.seg.h5")
        >>> for frame_idx in handler.labeled_frame_indices:
        ...     prompt = handler.get_prompt(frame_idx)
        ...     print(f"Frame {frame_idx}: {prompt.num_objects} objects")

        # With frame filtering
        >>> handler = MaskPromptHandler(
        ...     "masks.seg.h5",
        ...     frame_idx=100  # Only use frame 100
        ... )
    """

    def __init__(
        self,
        path: str | Path,
        frame_idx: int | None = None,
        target_resolution: tuple[int, int] | None = None,
    ):
        """Initialize mask prompt handler.

        Args:
            path: Path to seg.h5 file.
            frame_idx: Specific frame index to use. If None (default), uses
                the first labeled frame in the file when load() is called.
            target_resolution: Target video resolution as (width, height).
                If provided and differs from source, masks will be resized.
                If None, uses source resolution without validation.
        """
        self.path = Path(path)
        self.frame_idx = frame_idx
        self.target_resolution = target_resolution

        self._reader: SegmentationReader | None = None

    @property
    def prompt_type(self) -> PromptType:
        """The type of prompt this handler produces."""
        return PromptType.ROI

    def _get_reader(self) -> SegmentationReader:
        """Get or create the segmentation reader (lazy loading)."""
        if self._reader is None:
            self._reader = SegmentationReader(self.path)
        return self._reader

    @property
    def resolution(self) -> tuple[int, int]:
        """Source resolution as (width, height)."""
        return self._get_reader().resolution

    @property
    def num_frames(self) -> int:
        """Number of frames with mask data."""
        return self._get_reader().num_frames

    @property
    def num_tracks(self) -> int:
        """Number of unique tracks in the file."""
        return self._get_reader().num_tracks

    @property
    def labeled_frame_indices(self) -> list[int]:
        """Sorted list of frame indices with mask data."""
        return self._get_reader().labeled_frame_indices

    def _resize_mask(
        self,
        mask: np.ndarray,
        target_size: tuple[int, int],
    ) -> np.ndarray:
        """Resize a mask to target size using nearest neighbor.

        Args:
            mask: Binary mask as (H, W) array.
            target_size: Target size as (width, height).

        Returns:
            Resized mask as (H, W) array.
        """
        from PIL import Image

        target_w, target_h = target_size
        if mask.shape == (target_h, target_w):
            return mask

        # Use PIL for resizing with nearest neighbor (preserves binary values)
        img = Image.fromarray(mask)
        resized = img.resize((target_w, target_h), Image.Resampling.NEAREST)
        return np.array(resized, dtype=np.uint8)

    def _build_prompt(self, frame_idx: int) -> Prompt:
        """Build a Prompt from masks at a specific frame.

        Args:
            frame_idx: The frame index to extract masks from.

        Returns:
            A Prompt object with masks for each object.

        Raises:
            ValueError: If no objects are found at the frame.
        """
        reader = self._get_reader()

        # Get data from reader
        masks_array = reader.get_masks(frame_idx)
        track_ids = reader.get_track_ids(frame_idx)
        track_names = reader.get_track_names()

        if len(track_ids) == 0:
            raise ValueError(
                f"No objects at frame {frame_idx} in {self.path}"
            )

        # Determine target resolution
        if self.target_resolution is not None:
            target_size = self.target_resolution
        else:
            target_size = reader.resolution

        # Convert to list of individual masks, resizing if needed
        masks = []
        for i in range(len(track_ids)):
            mask = masks_array[i]
            if self.target_resolution is not None:
                mask = self._resize_mask(mask, target_size)
            masks.append(mask)

        # Build object ID to name mapping
        obj_ids = track_ids.tolist()
        obj_names = {tid: track_names.get(tid, f"object_{tid}") for tid in obj_ids}

        return Prompt(
            prompt_type=PromptType.ROI,
            frame_idx=frame_idx,
            obj_ids=obj_ids,
            obj_names=obj_names,
            masks=masks,
            source_path=self.path,
        )

    def get_prompt(self, frame_idx: int) -> Prompt | None:
        """Get prompt for a specific frame.

        This provides O(1) access to prompts at any labeled frame after an
        initial O(n) index build.

        Args:
            frame_idx: The frame index to get prompts for.

        Returns:
            A Prompt object if the frame has masks, None otherwise.
        """
        reader = self._get_reader()

        if not reader.has_frame(frame_idx):
            return None

        # Check if frame has objects
        if reader.get_num_objects(frame_idx) == 0:
            return None

        return self._build_prompt(frame_idx)

    def load(self) -> Prompt:
        """Load mask prompts from seg.h5 file.

        This is the primary method for single-frame prompt loading. For
        multi-frame access, use get_prompt() with labeled_frame_indices.

        Returns:
            A Prompt object with masks for each object.

        Raises:
            FileNotFoundError: If the seg.h5 file doesn't exist.
            ValueError: If the file has no labeled frames or no objects
                at the specified frame.
        """
        # FileNotFoundError is raised by SegmentationReader.__init__
        reader = self._get_reader()

        if reader.num_frames == 0:
            raise ValueError(f"No labeled frames in {self.path}")

        # Determine target frame
        if self.frame_idx is not None:
            if not reader.has_frame(self.frame_idx):
                raise ValueError(
                    f"No masks at frame {self.frame_idx} in {self.path}"
                )
            target_frame = self.frame_idx
        else:
            # Use first labeled frame
            target_frame = self.labeled_frame_indices[0]

        return self._build_prompt(target_frame)

    def close(self) -> None:
        """Close the underlying reader."""
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self) -> "MaskPromptHandler":
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        try:
            return (
                f"MaskPromptHandler(path={self.path!r}, "
                f"num_frames={self.num_frames}, num_tracks={self.num_tracks})"
            )
        except Exception:
            return f"MaskPromptHandler(path={self.path!r})"
