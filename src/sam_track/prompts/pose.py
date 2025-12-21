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
    - Multi-frame access via get_prompt() and labeled_frame_indices
    - Node filtering to use only specific keypoints
    - Track-based object IDs when available
    - GT vs prediction instance handling (GT takes precedence)

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

        # Multi-frame access
        >>> handler = PosePromptHandler("labels.slp")
        >>> for frame_idx in handler.labeled_frame_indices:
        ...     prompt = handler.get_prompt(frame_idx)
        ...     print(f"Frame {frame_idx}: {prompt.num_objects} objects")
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
        self._frame_map: dict[int, sio.LabeledFrame] | None = None

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

    @property
    def video(self) -> sio.Video:
        """The video from the SLP file."""
        return self._load_labels().video

    def _build_frame_map(self) -> dict[int, sio.LabeledFrame]:
        """Build frame index on first access. O(n) once, O(1) thereafter.

        Returns:
            Dictionary mapping frame indices to LabeledFrame objects.
        """
        if self._frame_map is None:
            labels = self._load_labels()
            # Use find() to get frames for this video (tolerates multi-video projects)
            frames = labels.find(labels.video)
            self._frame_map = {lf.frame_idx: lf for lf in frames}
        return self._frame_map

    @property
    def labeled_frame_indices(self) -> list[int]:
        """Sorted frame indices that have labels."""
        return sorted(self._build_frame_map().keys())

    def _build_prompt(self, lf: sio.LabeledFrame) -> Prompt:
        """Build a Prompt from a LabeledFrame.

        This method handles GT vs prediction instance separation. If a frame
        has any GT instances (sio.Instance), only GT instances are used.
        Otherwise, predictions (sio.PredictedInstance) are used.

        Args:
            lf: The LabeledFrame to extract prompts from.

        Returns:
            A Prompt object with point sets for each instance.

        Raises:
            ValueError: If no instances have visible points.
        """
        labels = self._load_labels()
        skeleton = labels.skeleton

        # Separate GT from predictions (CRITICAL: use type(), not isinstance())
        # PredictedInstance is a subclass of Instance, so isinstance() would
        # match both types. We need exact type checking.
        gt_instances = [i for i in lf.instances if type(i) is sio.Instance]
        pred_instances = [i for i in lf.instances if type(i) is sio.PredictedInstance]

        # GT takes precedence - if any GT exists, ignore predictions
        instances = gt_instances if gt_instances else pred_instances

        # Build node filter mask if nodes specified
        if self.nodes:
            node_filter = np.array([n.name in self.nodes for n in skeleton.nodes])
        else:
            node_filter = None

        obj_ids = []
        obj_names = {}
        points = []

        for i, inst in enumerate(instances):
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

    def get_prompt(self, frame_idx: int) -> Prompt | None:
        """Get prompt for a specific frame.

        This provides O(1) access to prompts at any labeled frame after an
        initial O(n) index build.

        Args:
            frame_idx: The frame index to get prompts for.

        Returns:
            A Prompt object if the frame has labels, None otherwise.
        """
        lf = self._build_frame_map().get(frame_idx)
        if lf is None:
            return None
        return self._build_prompt(lf)

    def load(self) -> Prompt:
        """Load pose prompts from SLP file.

        This is the primary method for single-frame prompt loading. For
        multi-frame access, use get_prompt() with labeled_frame_indices.

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
            lf = self._build_frame_map().get(self.frame_idx)
            if lf is None:
                raise ValueError(f"No labels at frame {self.frame_idx} in {self.path}")
        else:
            # Use the frame with minimum frame_idx, not the first in the unsorted list
            # Note: labeled_frames list order is arbitrary (insertion order),
            # so we must use labeled_frame_indices to get sorted order
            first_frame_idx = self.labeled_frame_indices[0]
            lf = self._build_frame_map()[first_frame_idx]

        return self._build_prompt(lf)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PosePromptHandler(path={self.path!r}, "
            f"num_labeled_frames={self.num_labeled_frames})"
        )
