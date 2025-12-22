"""SLEAP SLP output writer for tracked poses.

This module provides a writer for saving pose instances with SAM3-assigned
track identities back to SLEAP SLP format. This enables integration with
SLEAP workflows for downstream analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import sleap_io as sio

if TYPE_CHECKING:
    from ..reconciliation import TrackAssignment


class SLPWriter:
    """Writer for tracked poses in SLEAP SLP format.

    This writer takes pose instances and track assignments from the ID
    reconciliation layer and outputs a new SLP file with track identities
    assigned based on SAM3 mask matching.

    Attributes:
        output_path: Path for the output SLP file.
        remove_unmatched: If True, exclude poses without track assignment.

    Example:
        >>> writer = SLPWriter(
        ...     output_path="tracked.slp",
        ...     source_labels=labels,
        ...     remove_unmatched=False,
        ... )
        >>> for frame_idx in gt_frame_indices:
        ...     writer.add_frame_assignments(
        ...         frame_idx=frame_idx,
        ...         assignments=assignments,
        ...         original_instances=lf.instances,
        ...     )
        >>> writer.finalize()
    """

    def __init__(
        self,
        output_path: str | Path,
        source_labels: sio.Labels,
        remove_unmatched: bool = False,
    ):
        """Initialize SLP writer.

        Args:
            output_path: Path for output SLP file.
            source_labels: Original SLEAP Labels to copy structure from.
            remove_unmatched: If True, exclude poses without track assignment.
        """
        self.output_path = Path(output_path)
        self.remove_unmatched = remove_unmatched

        # Clone skeleton and video from source
        self._skeleton = source_labels.skeleton
        self._videos = list(source_labels.videos)

        # Track management - create tracks on demand
        self._tracks: dict[str, sio.Track] = {}

        # Track name resolver for looking up GT names by SAM3 obj_id
        self._track_name_resolver: dict[int, str] | None = None

        # Accumulate assignments per frame: {frame_idx: (assignments, instances)}
        self._frame_data: dict[
            int, tuple[list[TrackAssignment], list[sio.Instance]]
        ] = {}

    def add_frame_assignments(
        self,
        frame_idx: int,
        assignments: list[TrackAssignment],
        original_instances: list[sio.Instance],
    ) -> None:
        """Add assignments for a frame.

        Args:
            frame_idx: Frame index.
            assignments: List of TrackAssignment objects from reconciliation.
            original_instances: Original pose instances from the frame.
        """
        self._frame_data[frame_idx] = (assignments, list(original_instances))

    def apply_track_name_mapping(
        self,
        mapping: dict[int, str],
    ) -> None:
        """Apply a SAM3 obj_id -> track_name mapping.

        This mapping is used when a pose instance doesn't have a track name
        (e.g., PredictedInstances). The SAM3 obj_id is looked up to get the
        resolved GT track name.

        Must be called before finalize().

        Args:
            mapping: Dictionary mapping SAM3 obj_id to track name.
        """
        self._track_name_resolver = mapping

    def _get_or_create_track(self, name: str) -> sio.Track:
        """Get existing track or create new one.

        Args:
            name: Track name.

        Returns:
            The Track object.
        """
        if name not in self._tracks:
            self._tracks[name] = sio.Track(name=name)
        return self._tracks[name]

    def _copy_instance_with_track(
        self,
        inst: sio.Instance,
        track: sio.Track | None,
        tracking_score: float | None = None,
    ) -> sio.Instance:
        """Copy an instance with a new track, preserving the original type.

        Args:
            inst: Original instance (Instance or PredictedInstance).
            track: Track to assign to the new instance.
            tracking_score: Optional tracking score to assign (e.g., from
                SAM3 matching). If None, preserves the original tracking_score.

        Returns:
            A new instance of the same type with the new track.
        """
        score = tracking_score if tracking_score is not None else inst.tracking_score
        if type(inst) is sio.PredictedInstance:
            return sio.PredictedInstance(
                points=inst.points.copy(),
                skeleton=self._skeleton,
                track=track,
                score=inst.score,
                tracking_score=score,
            )
        else:
            return sio.Instance(
                points=inst.points.copy(),
                skeleton=self._skeleton,
                track=track,
                tracking_score=score,
                from_predicted=inst.from_predicted,
            )

    def finalize(self) -> sio.Labels:
        """Build and save the final Labels object.

        Returns:
            The constructed Labels object.
        """
        labeled_frames = []

        for frame_idx in sorted(self._frame_data.keys()):
            assignments, original_instances = self._frame_data[frame_idx]

            # Build assignment lookup: pose_idx -> (track_name, sam3_id, confidence)
            pose_to_assignment: dict[int, tuple[str | None, int, float]] = {}
            for a in assignments:
                pose_to_assignment[a.pose_idx] = (
                    a.pose_track_name,
                    a.sam3_obj_id,
                    a.confidence,
                )

            new_instances = []
            for i, inst in enumerate(original_instances):
                if i in pose_to_assignment:
                    track_name, sam3_id, confidence = pose_to_assignment[i]

                    # Determine track name
                    if track_name:
                        # Use original track name from pose (GT instances)
                        track = self._get_or_create_track(track_name)
                    elif (
                        self._track_name_resolver
                        and sam3_id in self._track_name_resolver
                    ):
                        # Use resolved GT track name from mapping (predictions)
                        track = self._get_or_create_track(
                            self._track_name_resolver[sam3_id]
                        )
                    else:
                        # Fallback: Use SAM3 ID as track name
                        track = self._get_or_create_track(f"track_{sam3_id}")

                    # Create new instance with track assignment, preserving
                    # the original instance type and using matching confidence
                    new_inst = self._copy_instance_with_track(
                        inst, track, tracking_score=confidence
                    )
                    new_instances.append(new_inst)

                elif not self.remove_unmatched:
                    # Keep unmatched instance without track
                    # Preserve existing track if present
                    new_inst = self._copy_instance_with_track(inst, inst.track)
                    new_instances.append(new_inst)
                # else: remove_unmatched=True and no assignment, skip this instance

            if new_instances:
                lf = sio.LabeledFrame(
                    video=self._videos[0] if self._videos else None,
                    frame_idx=frame_idx,
                    instances=new_instances,
                )
                labeled_frames.append(lf)

        # Build Labels
        labels = sio.Labels(
            labeled_frames=labeled_frames,
            videos=self._videos,
            skeletons=[self._skeleton],
            tracks=list(self._tracks.values()),
        )

        # Save to disk
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        sio.save_slp(labels, str(self.output_path))

        return labels

    def close(self) -> None:
        """Alias for finalize() for context manager compatibility."""
        self.finalize()

    def __enter__(self) -> SLPWriter:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - finalize on clean exit."""
        if exc_type is None:
            self.finalize()

    @property
    def num_frames(self) -> int:
        """Number of frames with data."""
        return len(self._frame_data)

    @property
    def num_tracks(self) -> int:
        """Number of unique tracks created."""
        return len(self._tracks)
