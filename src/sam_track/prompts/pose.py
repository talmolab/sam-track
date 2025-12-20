"""Pose prompt handler for SLEAP SLP files."""

from pathlib import Path

import numpy as np
import sleap_io as sio

from sam_track.prompts.base import Prompt, PromptHandler, PromptType


class PosePromptHandler(PromptHandler):
    """Handler for pose prompts from SLEAP SLP files.

    This handler loads pose annotations from SLEAP .slp files and converts
    them to point prompts for SAM3 tracking. Each instance's visible keypoints
    become a set of positive point prompts.

    The handler supports:
    - Automatic frame selection (first labeled frame) or specific frame
    - Node filtering to use only specific keypoints
    - Track-based object IDs when available

    Example:
        >>> handler = PosePromptHandler("path/to/labels.slp")
        >>> prompt = handler.load()
        >>> print(f"Loaded {prompt.num_objects} objects")
        >>> for obj_id in prompt.obj_ids:
        ...     print(f"  {obj_id}: {prompt.get_name(obj_id)}")

        # With node filtering
        >>> handler = PosePromptHandler(
        ...     "labels.slp",
        ...     nodes=["head", "neck", "tail_base"]
        ... )
    """

    def __init__(
        self,
        path: str | Path,
        frame_idx: int | None = None,
        nodes: list[str] | None = None,
    ):
        """Initialize pose prompt handler.

        Args:
            path: Path to SLEAP .slp file.
            frame_idx: Specific frame index to use. If None (default), uses
                the first labeled frame in the file.
            nodes: List of node names to use as prompts. If None (default),
                uses all visible nodes. Use this to filter to specific
                body parts (e.g., ["head", "neck", "tail_base"]).
        """
        self.path = Path(path)
        self.frame_idx = frame_idx
        self.nodes = nodes

        self._labels: sio.Labels | None = None

    @property
    def prompt_type(self) -> PromptType:
        """The type of prompt this handler produces."""
        return PromptType.POSE

    def _load_labels(self) -> sio.Labels:
        """Load and cache SLEAP labels."""
        if self._labels is None:
            self._labels = sio.load_slp(str(self.path), open_videos=False)
        return self._labels

    @property
    def skeleton(self) -> sio.Skeleton:
        """The skeleton from the SLP file."""
        return self._load_labels().skeleton

    @property
    def node_names(self) -> list[str]:
        """List of node names in the skeleton."""
        return [n.name for n in self.skeleton.nodes]

    @property
    def num_labeled_frames(self) -> int:
        """Number of labeled frames in the file."""
        return len(self._load_labels().labeled_frames)

    @property
    def num_tracks(self) -> int:
        """Number of tracks in the file."""
        return len(self._load_labels().tracks)

    def load(self) -> Prompt:
        """Load pose prompts from SLP file.

        Returns:
            A Prompt object with point sets for each instance.

        Raises:
            FileNotFoundError: If the SLP file doesn't exist.
            ValueError: If the file has no labeled frames or no instances
                at the specified frame.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"SLP file not found: {self.path}")

        labels = self._load_labels()

        if len(labels.labeled_frames) == 0:
            raise ValueError(f"No labeled frames in {self.path}")

        # Find target frame
        if self.frame_idx is not None:
            lf = next(
                (f for f in labels.labeled_frames if f.frame_idx == self.frame_idx),
                None,
            )
            if lf is None:
                raise ValueError(f"No labels at frame {self.frame_idx} in {self.path}")
        else:
            lf = labels.labeled_frames[0]

        skeleton = labels.skeleton

        # Build node filter mask if nodes specified
        if self.nodes:
            node_filter = np.array([n.name in self.nodes for n in skeleton.nodes])
        else:
            node_filter = None

        obj_ids = []
        obj_names = {}
        points = []

        for i, inst in enumerate(lf.instances):
            coords = inst.numpy()  # (n_nodes, 2)
            visible_mask = ~np.isnan(coords[:, 0])

            # Apply node filter if specified
            if node_filter is not None:
                visible_mask = visible_mask & node_filter

            visible_coords = coords[visible_mask]

            # Skip instances with no visible points
            if len(visible_coords) == 0:
                continue

            # Get object ID from track or instance index
            if inst.track is not None:
                obj_id = labels.tracks.index(inst.track)
                name = inst.track.name
            else:
                obj_id = i
                name = f"instance_{i}"

            obj_ids.append(obj_id)
            obj_names[obj_id] = name

            # Convert to list of (x, y) tuples
            point_set = [(float(x), float(y)) for x, y in visible_coords]
            points.append(point_set)

        if not obj_ids:
            raise ValueError(
                f"No instances with visible points at frame {lf.frame_idx} "
                f"in {self.path}"
            )

        return Prompt(
            prompt_type=PromptType.POSE,
            frame_idx=lf.frame_idx,
            obj_ids=obj_ids,
            obj_names=obj_names,
            points=points,
            source_path=self.path,
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PosePromptHandler(path={self.path!r}, "
            f"num_labeled_frames={self.num_labeled_frames})"
        )
