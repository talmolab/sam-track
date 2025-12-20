"""Video loading utilities using sleap-io."""

from pathlib import Path
from typing import Iterator

import numpy as np
import sleap_io as sio


class VideoReader:
    """Video reader wrapper around sleap-io.

    This class provides a consistent interface for loading video frames
    from various sources using sleap-io backends.

    Attributes:
        video: The sleap-io Video object.
        path: Path to the video file.
        num_frames: Total number of frames.
        height: Frame height in pixels.
        width: Frame width in pixels.
        channels: Number of color channels (1 or 3).
        fps: Frames per second (if available from backend).

    Example:
        >>> with VideoReader("video.mp4") as reader:
        ...     print(f"Video has {reader.num_frames} frames")
        ...     for idx, frame in reader.iter_frames():
        ...         process(frame)
    """

    def __init__(
        self,
        path: str | Path,
        plugin: str | None = None,
    ):
        """Initialize video reader.

        Args:
            path: Path to video file or image directory.
            plugin: Video backend to use. Options include:
                - "opencv": Fast, but limited codec support
                - "FFMPEG": Slower, but most reliable
                - "pyav": Balanced speed and compatibility
                If None, sleap-io will auto-detect the best backend.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")

        self.video = sio.load_video(str(self.path), plugin=plugin, keep_open=True)

        # Cache metadata
        self._shape = self.video.shape

    @property
    def num_frames(self) -> int:
        """Total number of frames in the video."""
        return self._shape[0] if self._shape else 0

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self._shape[1] if self._shape else 0

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self._shape[2] if self._shape else 0

    @property
    def channels(self) -> int:
        """Number of color channels (1 for grayscale, 3 for RGB)."""
        return self._shape[3] if self._shape else 0

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Video shape as (num_frames, height, width, channels)."""
        return self._shape

    @property
    def fps(self) -> float | None:
        """Frames per second (if available from backend).

        Returns:
            FPS value or None if not available.
        """
        backend = self.video.backend
        if hasattr(backend, "fps"):
            return backend.fps
        return None

    @property
    def duration(self) -> float | None:
        """Video duration in seconds (if FPS is available).

        Returns:
            Duration in seconds or None if FPS not available.
        """
        if self.fps is not None and self.fps > 0:
            return self.num_frames / self.fps
        return None

    def __len__(self) -> int:
        """Return total number of frames."""
        return self.num_frames

    def __getitem__(self, idx: int | slice) -> np.ndarray:
        """Get frame(s) by index.

        Args:
            idx: Frame index (int) or slice.

        Returns:
            For single index: (H, W, C) numpy array
            For slice: (N, H, W, C) numpy array
        """
        return self.video[idx]

    def get_frame(self, idx: int) -> np.ndarray:
        """Get a single frame by index.

        Args:
            idx: Frame index (0-based).

        Returns:
            Frame as (H, W, C) numpy array with uint8 dtype.

        Raises:
            IndexError: If idx is out of bounds.
        """
        if idx < 0 or idx >= self.num_frames:
            raise IndexError(
                f"Frame index {idx} out of range [0, {self.num_frames - 1}]"
            )
        return self.video[idx]

    def iter_frames(
        self,
        start: int = 0,
        end: int | None = None,
        step: int = 1,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over frames in the video.

        Args:
            start: Starting frame index (inclusive).
            end: Ending frame index (exclusive). Defaults to end of video.
            step: Step size between frames.

        Yields:
            Tuple of (frame_idx, frame_array) where frame_array is (H, W, C).

        Example:
            >>> for idx, frame in reader.iter_frames(start=100, end=200):
            ...     process(frame)
        """
        if end is None:
            end = self.num_frames

        # Clamp values to valid range
        start = max(0, start)
        end = min(end, self.num_frames)

        for idx in range(start, end, step):
            yield idx, self.video[idx]

    def get_frames(
        self,
        indices: list[int] | None = None,
        start: int = 0,
        end: int | None = None,
    ) -> np.ndarray:
        """Get multiple frames as a batch.

        Args:
            indices: Specific frame indices to retrieve. If provided,
                start and end are ignored.
            start: Starting frame index (if indices not provided).
            end: Ending frame index (if indices not provided).

        Returns:
            Frames as (N, H, W, C) numpy array.
        """
        if indices is not None:
            return np.stack([self.video[i] for i in indices])

        if end is None:
            end = self.num_frames

        return self.video[start:end]

    def close(self) -> None:
        """Close video backend and release resources."""
        if hasattr(self.video, "close"):
            self.video.close()

    def __enter__(self) -> "VideoReader":
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """Return string representation."""
        fps_str = f", fps={self.fps:.2f}" if self.fps else ""
        return (
            f"VideoReader('{self.path.name}', "
            f"frames={self.num_frames}, "
            f"size={self.width}x{self.height}{fps_str})"
        )
