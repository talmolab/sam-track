"""JSON bounding box track output writer.

This module provides a writer for saving bounding box tracking results to
JSON format. The output includes video metadata, track data with per-frame
detections, and summary statistics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..tracker import TrackingResult


class TrackingEncoder(json.JSONEncoder):
    """JSON encoder for tracking data with numpy and datetime support."""

    def default(self, obj: Any) -> Any:
        """Encode non-standard types to JSON-serializable values."""
        # NumPy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # NumPy scalars
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)

        # Datetime
        if isinstance(obj, datetime):
            return obj.isoformat()

        return super().default(obj)


@dataclass
class Detection:
    """Single bounding box detection in a frame.

    Attributes:
        frame_idx: Frame index in the video.
        x_min: Left edge x coordinate (pixels).
        y_min: Top edge y coordinate (pixels).
        x_max: Right edge x coordinate (pixels).
        y_max: Bottom edge y coordinate (pixels).
        confidence: Detection confidence score (0-1).
    """

    frame_idx: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float

    @property
    def width(self) -> float:
        """Bounding box width in pixels."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Bounding box height in pixels."""
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """Bounding box area in pixels squared."""
        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "frame_idx": self.frame_idx,
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "area": self.area,
        }


@dataclass
class Track:
    """A tracked object with detections across frames.

    Attributes:
        track_id: Unique identifier for this track.
        name: Human-readable name for the tracked object.
        detections: List of detections for this track.
    """

    track_id: int
    name: str | None = None
    detections: list[Detection] = field(default_factory=list)

    def add_detection(self, detection: Detection) -> None:
        """Add a detection to this track."""
        self.detections.append(detection)

    @property
    def num_detections(self) -> int:
        """Number of detections in this track."""
        return len(self.detections)

    @property
    def first_frame(self) -> int | None:
        """First frame index with a detection."""
        if not self.detections:
            return None
        return min(d.frame_idx for d in self.detections)

    @property
    def last_frame(self) -> int | None:
        """Last frame index with a detection."""
        if not self.detections:
            return None
        return max(d.frame_idx for d in self.detections)

    @property
    def avg_confidence(self) -> float:
        """Average confidence across all detections."""
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "track_id": self.track_id,
            "name": self.name,
            "num_detections": self.num_detections,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "avg_confidence": self.avg_confidence,
            "detections": [d.to_dict() for d in self.detections],
        }


class BBoxWriter:
    """Writer for bounding box tracks in JSON format.

    This class collects tracking results and writes them to a JSON file with
    metadata, track data, and statistics.

    Example:
        >>> writer = BBoxWriter(
        ...     output_path="output.bbox.json",
        ...     video_path="video.mp4",
        ...     video_width=1920,
        ...     video_height=1080,
        ... )
        >>> for result in tracker.propagate():
        ...     writer.add_result(result)
        >>> writer.save()

    Attributes:
        output_path: Path to the output JSON file.
        tracks: Dictionary mapping track IDs to Track objects.
    """

    def __init__(
        self,
        output_path: str | Path,
        video_path: str | Path,
        video_width: int,
        video_height: int,
        fps: float | None = None,
        total_frames: int = 0,
        prompt_type: str = "text",
        prompt_value: str = "",
        obj_names: dict[int, str] | None = None,
    ):
        """Initialize the bounding box writer.

        Args:
            output_path: Path for the output JSON file.
            video_path: Path to the source video file.
            video_width: Video width in pixels.
            video_height: Video height in pixels.
            fps: Video frames per second (optional).
            total_frames: Total number of frames in the video.
            prompt_type: Type of prompt used ("text", "roi", "pose").
            prompt_value: Description of the prompt.
            obj_names: Optional mapping from object ID to name.
        """
        self.output_path = Path(output_path)
        self.tracks: dict[int, Track] = {}
        self._obj_names = obj_names or {}
        self._frames_with_detections: set[int] = set()

        self._metadata = {
            "version": "1.0",
            "format": "sam-track-bbox",
            "video_source": str(video_path),
            "width": video_width,
            "height": video_height,
            "fps": fps,
            "total_frames": total_frames,
            "tracking_model": "SAM3",
            "prompt_type": prompt_type,
            "prompt_value": prompt_value,
            "timestamp": datetime.now(),
        }

    def add_result(self, result: TrackingResult) -> None:
        """Add detections from a tracking result.

        Args:
            result: TrackingResult from the tracker containing frame data.
        """
        if result.num_objects > 0:
            self._frames_with_detections.add(result.frame_idx)

        for i in range(result.num_objects):
            obj_id = int(result.object_ids[i])
            box = result.boxes[i]
            score = float(result.scores[i])

            # Create track if needed
            if obj_id not in self.tracks:
                name = self._obj_names.get(obj_id)
                self.tracks[obj_id] = Track(track_id=obj_id, name=name)

            # Add detection
            detection = Detection(
                frame_idx=result.frame_idx,
                x_min=float(box[0]),
                y_min=float(box[1]),
                x_max=float(box[2]),
                y_max=float(box[3]),
                confidence=score,
            )
            self.tracks[obj_id].add_detection(detection)

    def _compute_statistics(self) -> dict[str, Any]:
        """Compute summary statistics for all tracks."""
        total_detections = sum(t.num_detections for t in self.tracks.values())

        avg_confidence = 0.0
        if total_detections > 0:
            all_confidences = [
                d.confidence for t in self.tracks.values() for d in t.detections
            ]
            avg_confidence = sum(all_confidences) / len(all_confidences)

        return {
            "total_tracks": len(self.tracks),
            "total_detections": total_detections,
            "frames_with_detections": len(self._frames_with_detections),
            "avg_confidence": avg_confidence,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert all data to a dictionary for JSON serialization."""
        return {
            "metadata": self._metadata,
            "tracks": [t.to_dict() for t in self.tracks.values()],
            "statistics": self._compute_statistics(),
        }

    def apply_track_name_mapping(
        self,
        mapping: dict[int, str],
    ) -> None:
        """Apply a SAM3 obj_id -> track_name mapping to all tracks.

        This updates the `name` attribute of each track based on the provided
        mapping. Tracks not in the mapping keep their existing names.

        Args:
            mapping: Dictionary mapping SAM3 obj_id to track name.
        """
        for obj_id, track in self.tracks.items():
            if obj_id in mapping:
                track.name = mapping[obj_id]

    def save(self) -> None:
        """Save all tracks to the JSON file."""
        # Ensure parent directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w") as f:
            json.dump(self.to_dict(), f, cls=TrackingEncoder, indent=2)
