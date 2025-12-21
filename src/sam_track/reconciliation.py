"""ID reconciliation for matching SAM3 masks to poses.

This module provides tools for:
- Matching SAM3 segmentation masks to pose instances using Hungarian algorithm
- Detecting identity swaps across frames
- Building frame-to-ID mappings for output reconciliation

The key insight from experimentation is that SAM3 mid-propagation re-prompting
works for adding NEW objects, but NOT for correcting existing tracks. Therefore,
identity correction must be done via post-processing with mask-to-pose matching.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    import sleap_io as sio


@dataclass
class MatchContext:
    """Context for match predicate evaluation.

    Attributes:
        frame_idx: Frame index where the match was made.
        sam3_obj_id: SAM3 object ID of the matched mask.
        cost: Raw cost from the cost matrix (negative keypoints inside).
        keypoints_inside: Number of visible keypoints inside the mask.
        keypoints_visible: Total number of visible keypoints in the pose.
        mask_area: Area of the mask in pixels.
        mask_centroid: Centroid of the mask as (x, y).
    """

    frame_idx: int
    sam3_obj_id: int
    cost: float
    keypoints_inside: int
    keypoints_visible: int
    mask_area: int
    mask_centroid: tuple[float, float]


# Type alias for match predicates
MatchPredicate = Callable[["sio.Instance", np.ndarray, MatchContext], bool]


@dataclass
class TrackAssignment:
    """A single track assignment at a frame.

    Attributes:
        frame_idx: Frame index where assignment was made.
        pose_track_name: Name of the pose's track (None if untracked).
        pose_idx: Index of the pose in the frame's instance list.
        sam3_obj_id: SAM3 object ID that was matched.
        confidence: Match quality score (0-1, higher is better).
    """

    frame_idx: int
    pose_track_name: str | None
    pose_idx: int
    sam3_obj_id: int
    confidence: float


@dataclass
class SwapEvent:
    """Detected identity swap.

    Attributes:
        frame_idx: Frame where the swap was detected.
        track_name: Name of the track that swapped.
        old_sam3_id: Previous SAM3 object ID.
        new_sam3_id: New SAM3 object ID after swap.
    """

    frame_idx: int
    track_name: str
    old_sam3_id: int
    new_sam3_id: int


def default_match_predicate(
    pose: "sio.Instance", mask: np.ndarray, ctx: MatchContext
) -> bool:
    """Default match predicate: require at least 1 keypoint inside mask."""
    return ctx.keypoints_inside >= 1


@dataclass
class IDReconciler:
    """Matches SAM3 masks to poses and reconciles track IDs.

    This class implements Hungarian algorithm matching between pose instances
    and SAM3 segmentation masks, using keypoints-inside-mask as the cost metric.

    Attributes:
        skeleton: The SLEAP skeleton for node name lookups.
        exclude_nodes: Set of node names to exclude from matching.
        match_predicates: List of predicates that must all pass for a valid match.

    Example:
        >>> reconciler = IDReconciler(
        ...     skeleton=handler.skeleton,
        ...     exclude_nodes={"tail0", "tail1"},
        ... )
        >>> for frame_idx in gt_frame_indices:
        ...     assignments = reconciler.match_frame(
        ...         frame_idx=frame_idx,
        ...         poses=lf.instances,
        ...         masks=result.masks,
        ...         object_ids=result.object_ids,
        ...     )
        >>> swaps = reconciler.detect_swaps()
        >>> id_map = reconciler.build_id_map()
    """

    skeleton: "sio.Skeleton"
    exclude_nodes: set[str] = field(default_factory=set)
    match_predicates: list[MatchPredicate] = field(default_factory=list)
    _assignments: list[TrackAssignment] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Add default predicate if none provided."""
        if not self.match_predicates:
            self.match_predicates = [default_match_predicate]

    def compute_cost_matrix(
        self,
        poses: list["sio.Instance"],
        masks: np.ndarray,
    ) -> np.ndarray:
        """Compute cost matrix for Hungarian matching.

        The cost is the negative number of visible keypoints inside each mask.
        Lower cost = better match (more keypoints inside).

        Args:
            poses: List of pose instances to match.
            masks: Array of masks with shape (N, H, W).

        Returns:
            Cost matrix with shape (n_poses, n_masks).
        """
        n_poses = len(poses)
        n_masks = len(masks)

        if n_poses == 0 or n_masks == 0:
            return np.zeros((n_poses, n_masks))

        cost = np.zeros((n_poses, n_masks))

        # Get node names for filtering
        node_names = [n.name for n in self.skeleton.nodes]

        for i, pose in enumerate(poses):
            coords = pose.numpy()
            visible_mask = ~np.isnan(coords[:, 0])

            # Apply node exclusion filter
            if self.exclude_nodes:
                for j, name in enumerate(node_names):
                    if name in self.exclude_nodes:
                        visible_mask[j] = False

            visible_coords = coords[visible_mask].astype(int)

            for j, mask in enumerate(masks):
                inside_count = 0
                for x, y in visible_coords:
                    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                        if mask[y, x]:
                            inside_count += 1

                # Negative because Hungarian minimizes cost
                cost[i, j] = -inside_count

        return cost

    def match_frame(
        self,
        frame_idx: int,
        poses: list["sio.Instance"],
        masks: np.ndarray,
        object_ids: np.ndarray,
    ) -> list[TrackAssignment]:
        """Match poses to masks for a single frame.

        Uses Hungarian algorithm for optimal assignment, then filters
        matches through predicates.

        Args:
            frame_idx: Frame index for this match.
            poses: List of pose instances to match.
            masks: Array of masks with shape (N, H, W) or (N, 1, H, W).
            object_ids: Array of SAM3 object IDs corresponding to masks.

        Returns:
            List of valid TrackAssignment objects.
        """
        if len(poses) == 0 or len(masks) == 0:
            return []

        # Handle (N, 1, H, W) mask format from SAM3
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(axis=1)

        # Compute cost matrix and solve assignment
        cost = self.compute_cost_matrix(poses, masks)
        row_ind, col_ind = linear_sum_assignment(cost)

        # Get node names for visibility calculation
        node_names = [n.name for n in self.skeleton.nodes]

        assignments = []
        for pose_idx, mask_idx in zip(row_ind, col_ind):
            pose = poses[pose_idx]
            mask = masks[mask_idx]

            # Calculate visibility (excluding filtered nodes)
            coords = pose.numpy()
            visible_mask = ~np.isnan(coords[:, 0])
            if self.exclude_nodes:
                for j, name in enumerate(node_names):
                    if name in self.exclude_nodes:
                        visible_mask[j] = False
            visible_count = int(visible_mask.sum())

            # Calculate mask statistics
            ys, xs = np.where(mask)
            if len(xs) > 0:
                centroid = (float(xs.mean()), float(ys.mean()))
                mask_area = int(mask.sum())
            else:
                centroid = (0.0, 0.0)
                mask_area = 0

            # Build context for predicate evaluation
            keypoints_inside = int(-cost[pose_idx, mask_idx])
            ctx = MatchContext(
                frame_idx=frame_idx,
                sam3_obj_id=int(object_ids[mask_idx]),
                cost=float(cost[pose_idx, mask_idx]),
                keypoints_inside=keypoints_inside,
                keypoints_visible=visible_count,
                mask_area=mask_area,
                mask_centroid=centroid,
            )

            # Apply match predicates
            if all(pred(pose, mask, ctx) for pred in self.match_predicates):
                track_name = pose.track.name if pose.track else None
                confidence = (
                    keypoints_inside / visible_count if visible_count > 0 else 0.0
                )
                assignment = TrackAssignment(
                    frame_idx=frame_idx,
                    pose_track_name=track_name,
                    pose_idx=pose_idx,
                    sam3_obj_id=ctx.sam3_obj_id,
                    confidence=confidence,
                )
                assignments.append(assignment)

        self._assignments.extend(assignments)
        return assignments

    def detect_swaps(self) -> list[SwapEvent]:
        """Detect identity swaps from accumulated assignments.

        A swap occurs when a track name is matched to different SAM3 object IDs
        across frames.

        Returns:
            List of SwapEvent objects describing detected swaps.
        """
        swaps = []
        by_track: dict[str, list[TrackAssignment]] = defaultdict(list)

        for a in self._assignments:
            if a.pose_track_name:
                by_track[a.pose_track_name].append(a)

        for track_name, track_assignments in by_track.items():
            track_assignments.sort(key=lambda a: a.frame_idx)

            for i in range(1, len(track_assignments)):
                prev = track_assignments[i - 1]
                curr = track_assignments[i]

                if prev.sam3_obj_id != curr.sam3_obj_id:
                    swaps.append(
                        SwapEvent(
                            frame_idx=curr.frame_idx,
                            track_name=track_name,
                            old_sam3_id=prev.sam3_obj_id,
                            new_sam3_id=curr.sam3_obj_id,
                        )
                    )

        return swaps

    def build_id_map(self) -> dict[int, dict[int, str]]:
        """Build frame -> {sam3_id -> track_name} mapping.

        This can be used to remap SAM3 object IDs to consistent track names
        in output files.

        Returns:
            Dictionary mapping frame_idx to {sam3_obj_id: track_name}.
        """
        by_frame: dict[int, dict[int, str]] = defaultdict(dict)
        for a in self._assignments:
            if a.pose_track_name:
                by_frame[a.frame_idx][a.sam3_obj_id] = a.pose_track_name
        return dict(by_frame)

    def get_assignments(self) -> list[TrackAssignment]:
        """Get all accumulated assignments.

        Returns:
            List of all TrackAssignment objects from match_frame() calls.
        """
        return list(self._assignments)

    def clear(self) -> None:
        """Clear accumulated assignments."""
        self._assignments.clear()


# Predefined match predicates for common use cases


def require_min_keypoints_inside(min_count: int = 3) -> MatchPredicate:
    """Create predicate requiring minimum keypoints inside mask.

    Args:
        min_count: Minimum number of keypoints required inside mask.

    Returns:
        A MatchPredicate function.
    """

    def predicate(
        pose: "sio.Instance", mask: np.ndarray, ctx: MatchContext
    ) -> bool:
        return ctx.keypoints_inside >= min_count

    return predicate


def require_min_fraction_inside(min_frac: float = 0.5) -> MatchPredicate:
    """Create predicate requiring minimum fraction of keypoints inside mask.

    Args:
        min_frac: Minimum fraction (0-1) of visible keypoints inside mask.

    Returns:
        A MatchPredicate function.
    """

    def predicate(
        pose: "sio.Instance", mask: np.ndarray, ctx: MatchContext
    ) -> bool:
        if ctx.keypoints_visible == 0:
            return False
        return ctx.keypoints_inside / ctx.keypoints_visible >= min_frac

    return predicate


def require_centroid_proximity(max_dist: float = 100.0) -> MatchPredicate:
    """Create predicate requiring pose centroid near mask centroid.

    Args:
        max_dist: Maximum allowed distance between centroids in pixels.

    Returns:
        A MatchPredicate function.
    """

    def predicate(
        pose: "sio.Instance", mask: np.ndarray, ctx: MatchContext
    ) -> bool:
        coords = pose.numpy()
        pose_centroid = np.nanmean(coords, axis=0)
        if np.any(np.isnan(pose_centroid)):
            return False
        dist = np.linalg.norm(pose_centroid - np.array(ctx.mask_centroid))
        return float(dist) <= max_dist

    return predicate


def require_reasonable_mask_area(
    min_area: int = 1000, max_area: int = 500000
) -> MatchPredicate:
    """Create predicate requiring mask area within bounds.

    Args:
        min_area: Minimum mask area in pixels.
        max_area: Maximum mask area in pixels.

    Returns:
        A MatchPredicate function.
    """

    def predicate(
        pose: "sio.Instance", mask: np.ndarray, ctx: MatchContext
    ) -> bool:
        return min_area <= ctx.mask_area <= max_area

    return predicate
