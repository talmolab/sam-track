"""Tests for ID reconciliation module."""

from pathlib import Path

import numpy as np
import pytest
import sleap_io as sio

from sam_track.reconciliation import (
    IDReconciler,
    MatchContext,
    SwapEvent,
    TrackAssignment,
    TrackNameResolver,
    default_match_predicate,
    require_min_fraction_inside,
    require_min_keypoints_inside,
    require_reasonable_mask_area,
)

# Test data path
DATA_DIR = Path(__file__).parent / "data"
SLP_3NODE = DATA_DIR / "labels.3node.first_frame.slp"


@pytest.fixture
def simple_skeleton():
    """Create a simple 3-node skeleton for testing."""
    return sio.Skeleton(
        nodes=[
            sio.Node(name="head"),
            sio.Node(name="body"),
            sio.Node(name="tail"),
        ]
    )


@pytest.fixture
def reconciler(simple_skeleton):
    """Create a basic reconciler."""
    return IDReconciler(skeleton=simple_skeleton)


@pytest.fixture
def sample_poses(simple_skeleton):
    """Create sample pose instances."""
    # Instance 1: all nodes visible at (10, 10), (50, 50), (90, 90)
    coords1 = np.array([[10, 10], [50, 50], [90, 90]], dtype=np.float64)
    inst1 = sio.Instance.from_numpy(coords1, skeleton=simple_skeleton)

    # Instance 2: all nodes visible at (200, 200), (250, 250), (300, 300)
    coords2 = np.array([[200, 200], [250, 250], [300, 300]], dtype=np.float64)
    inst2 = sio.Instance.from_numpy(coords2, skeleton=simple_skeleton)

    return [inst1, inst2]


@pytest.fixture
def sample_masks():
    """Create sample masks that match the poses."""
    # Mask 1: covers area around (10-90, 10-90)
    mask1 = np.zeros((400, 400), dtype=bool)
    mask1[5:95, 5:95] = True

    # Mask 2: covers area around (200-300, 200-300)
    mask2 = np.zeros((400, 400), dtype=bool)
    mask2[195:305, 195:305] = True

    return np.stack([mask1, mask2])


class TestIDReconciler:
    """Tests for IDReconciler class."""

    def test_init(self, simple_skeleton):
        """Test reconciler initialization."""
        reconciler = IDReconciler(skeleton=simple_skeleton)
        assert reconciler.skeleton == simple_skeleton
        assert reconciler.exclude_nodes == set()
        assert len(reconciler.match_predicates) == 1  # Default predicate

    def test_init_with_exclude_nodes(self, simple_skeleton):
        """Test initialization with excluded nodes."""
        reconciler = IDReconciler(
            skeleton=simple_skeleton,
            exclude_nodes={"tail"},
        )
        assert "tail" in reconciler.exclude_nodes

    def test_compute_cost_matrix(self, reconciler, sample_poses, sample_masks):
        """Test cost matrix computation."""
        cost = reconciler.compute_cost_matrix(sample_poses, sample_masks)

        assert cost.shape == (2, 2)
        # Pose 1 should match mask 1 better (negative cost = more keypoints inside)
        assert cost[0, 0] < cost[0, 1]
        # Pose 2 should match mask 2 better
        assert cost[1, 1] < cost[1, 0]

    def test_compute_cost_matrix_empty_poses(self, reconciler, sample_masks):
        """Test cost matrix with empty poses."""
        cost = reconciler.compute_cost_matrix([], sample_masks)
        assert cost.shape == (0, 2)

    def test_compute_cost_matrix_empty_masks(self, reconciler, sample_poses):
        """Test cost matrix with empty masks."""
        cost = reconciler.compute_cost_matrix(sample_poses, np.array([]))
        assert cost.shape == (2, 0)

    def test_match_frame(self, reconciler, sample_poses, sample_masks):
        """Test frame matching."""
        object_ids = np.array([0, 1])
        assignments = reconciler.match_frame(
            frame_idx=0,
            poses=sample_poses,
            masks=sample_masks,
            object_ids=object_ids,
        )

        assert len(assignments) == 2
        # Pose 0 should match mask 0
        assert assignments[0].pose_idx == 0
        assert assignments[0].sam3_obj_id == 0
        # Pose 1 should match mask 1
        assert assignments[1].pose_idx == 1
        assert assignments[1].sam3_obj_id == 1

    def test_match_frame_accumulates(self, reconciler, sample_poses, sample_masks):
        """Test that match_frame accumulates assignments."""
        object_ids = np.array([0, 1])

        # First frame
        reconciler.match_frame(0, sample_poses, sample_masks, object_ids)
        assert len(reconciler.get_assignments()) == 2

        # Second frame
        reconciler.match_frame(10, sample_poses, sample_masks, object_ids)
        assert len(reconciler.get_assignments()) == 4

    def test_detect_swaps_no_swaps(self, simple_skeleton):
        """Test swap detection with consistent assignments."""
        reconciler = IDReconciler(skeleton=simple_skeleton)

        # Add consistent assignments
        reconciler._assignments = [
            TrackAssignment(
                frame_idx=0,
                pose_track_name="mouse1",
                pose_idx=0,
                sam3_obj_id=0,
                confidence=1.0,
            ),
            TrackAssignment(
                frame_idx=10,
                pose_track_name="mouse1",
                pose_idx=0,
                sam3_obj_id=0,
                confidence=1.0,
            ),
        ]

        swaps = reconciler.detect_swaps()
        assert len(swaps) == 0

    def test_detect_swaps_with_swap(self, simple_skeleton):
        """Test swap detection when identity changes."""
        reconciler = IDReconciler(skeleton=simple_skeleton)

        # mouse1 is assigned to obj 0 at frame 0, but obj 1 at frame 10
        reconciler._assignments = [
            TrackAssignment(
                frame_idx=0,
                pose_track_name="mouse1",
                pose_idx=0,
                sam3_obj_id=0,
                confidence=1.0,
            ),
            TrackAssignment(
                frame_idx=10,
                pose_track_name="mouse1",
                pose_idx=0,
                sam3_obj_id=1,
                confidence=1.0,
            ),
        ]

        swaps = reconciler.detect_swaps()
        assert len(swaps) == 1
        assert swaps[0].track_name == "mouse1"
        assert swaps[0].old_sam3_id == 0
        assert swaps[0].new_sam3_id == 1

    def test_build_id_map(self, simple_skeleton):
        """Test building frame-to-ID mapping."""
        reconciler = IDReconciler(skeleton=simple_skeleton)

        reconciler._assignments = [
            TrackAssignment(
                frame_idx=0,
                pose_track_name="mouse1",
                pose_idx=0,
                sam3_obj_id=0,
                confidence=1.0,
            ),
            TrackAssignment(
                frame_idx=0,
                pose_track_name="mouse2",
                pose_idx=1,
                sam3_obj_id=1,
                confidence=1.0,
            ),
            TrackAssignment(
                frame_idx=10,
                pose_track_name="mouse1",
                pose_idx=0,
                sam3_obj_id=0,
                confidence=1.0,
            ),
        ]

        id_map = reconciler.build_id_map()
        assert 0 in id_map
        assert 10 in id_map
        assert id_map[0][0] == "mouse1"
        assert id_map[0][1] == "mouse2"
        assert id_map[10][0] == "mouse1"

    def test_clear(self, reconciler, sample_poses, sample_masks):
        """Test clearing assignments."""
        reconciler.match_frame(0, sample_poses, sample_masks, np.array([0, 1]))
        assert len(reconciler.get_assignments()) > 0

        reconciler.clear()
        assert len(reconciler.get_assignments()) == 0

    def test_node_exclusion(self, simple_skeleton, sample_masks):
        """Test that excluded nodes are not counted."""
        reconciler = IDReconciler(
            skeleton=simple_skeleton,
            exclude_nodes={"head", "body"},  # Only tail counts
        )

        # Create pose with all nodes: head (10,10), body (50,50), tail (90,90)
        coords = np.array([[10, 10], [50, 50], [90, 90]], dtype=np.float64)
        inst = sio.Instance.from_numpy(coords, skeleton=simple_skeleton)

        cost = reconciler.compute_cost_matrix([inst], sample_masks)
        # Only tail (90, 90) should be counted, which is in mask 0 (covers 5:95, 5:95)
        assert cost[0, 0] == -1  # 1 keypoint inside


class TestMatchPredicates:
    """Tests for match predicate functions."""

    def test_default_predicate_passes(self):
        """Test default predicate passes with keypoints inside."""
        ctx = MatchContext(
            frame_idx=0,
            sam3_obj_id=0,
            cost=-3.0,
            keypoints_inside=3,
            keypoints_visible=5,
            mask_area=1000,
            mask_centroid=(50.0, 50.0),
        )
        assert default_match_predicate(None, None, ctx)

    def test_default_predicate_fails(self):
        """Test default predicate fails with no keypoints inside."""
        ctx = MatchContext(
            frame_idx=0,
            sam3_obj_id=0,
            cost=0.0,
            keypoints_inside=0,
            keypoints_visible=5,
            mask_area=1000,
            mask_centroid=(50.0, 50.0),
        )
        assert not default_match_predicate(None, None, ctx)

    def test_require_min_keypoints_inside(self):
        """Test minimum keypoints predicate."""
        predicate = require_min_keypoints_inside(min_count=3)

        ctx_pass = MatchContext(
            frame_idx=0,
            sam3_obj_id=0,
            cost=0,
            keypoints_inside=3,
            keypoints_visible=5,
            mask_area=1000,
            mask_centroid=(0, 0),
        )
        ctx_fail = MatchContext(
            frame_idx=0,
            sam3_obj_id=0,
            cost=0,
            keypoints_inside=2,
            keypoints_visible=5,
            mask_area=1000,
            mask_centroid=(0, 0),
        )

        assert predicate(None, None, ctx_pass)
        assert not predicate(None, None, ctx_fail)

    def test_require_min_fraction_inside(self):
        """Test minimum fraction predicate."""
        predicate = require_min_fraction_inside(min_frac=0.5)

        ctx_pass = MatchContext(
            frame_idx=0,
            sam3_obj_id=0,
            cost=0,
            keypoints_inside=3,
            keypoints_visible=5,
            mask_area=1000,
            mask_centroid=(0, 0),
        )  # 3/5 = 0.6 >= 0.5
        ctx_fail = MatchContext(
            frame_idx=0,
            sam3_obj_id=0,
            cost=0,
            keypoints_inside=1,
            keypoints_visible=5,
            mask_area=1000,
            mask_centroid=(0, 0),
        )  # 1/5 = 0.2 < 0.5

        assert predicate(None, None, ctx_pass)
        assert not predicate(None, None, ctx_fail)

    def test_require_reasonable_mask_area(self):
        """Test mask area bounds predicate."""
        predicate = require_reasonable_mask_area(min_area=100, max_area=10000)

        ctx_pass = MatchContext(
            frame_idx=0,
            sam3_obj_id=0,
            cost=0,
            keypoints_inside=3,
            keypoints_visible=5,
            mask_area=5000,
            mask_centroid=(0, 0),
        )
        ctx_too_small = MatchContext(
            frame_idx=0,
            sam3_obj_id=0,
            cost=0,
            keypoints_inside=3,
            keypoints_visible=5,
            mask_area=50,
            mask_centroid=(0, 0),
        )
        ctx_too_large = MatchContext(
            frame_idx=0,
            sam3_obj_id=0,
            cost=0,
            keypoints_inside=3,
            keypoints_visible=5,
            mask_area=50000,
            mask_centroid=(0, 0),
        )

        assert predicate(None, None, ctx_pass)
        assert not predicate(None, None, ctx_too_small)
        assert not predicate(None, None, ctx_too_large)


class TestTrackAssignment:
    """Tests for TrackAssignment dataclass."""

    def test_creation(self):
        """Test creating a track assignment."""
        assignment = TrackAssignment(
            frame_idx=10,
            pose_track_name="mouse1",
            pose_idx=0,
            sam3_obj_id=5,
            confidence=0.95,
        )
        assert assignment.frame_idx == 10
        assert assignment.pose_track_name == "mouse1"
        assert assignment.pose_idx == 0
        assert assignment.sam3_obj_id == 5
        assert assignment.confidence == 0.95

    def test_none_track_name(self):
        """Test assignment with no track name."""
        assignment = TrackAssignment(
            frame_idx=0,
            pose_track_name=None,
            pose_idx=0,
            sam3_obj_id=0,
            confidence=0.5,
        )
        assert assignment.pose_track_name is None


class TestSwapEvent:
    """Tests for SwapEvent dataclass."""

    def test_creation(self):
        """Test creating a swap event."""
        swap = SwapEvent(
            frame_idx=50,
            track_name="mouse1",
            old_sam3_id=0,
            new_sam3_id=1,
        )
        assert swap.frame_idx == 50
        assert swap.track_name == "mouse1"
        assert swap.old_sam3_id == 0
        assert swap.new_sam3_id == 1


class TestTrackNameResolver:
    """Tests for TrackNameResolver class."""

    def test_init_empty(self):
        """Test initialization with no anchors."""
        resolver = TrackNameResolver()
        assert resolver.gt_anchors == {}
        assert resolver.fallback_names == {}
        assert resolver.get_anchor_frames() == []

    def test_init_with_anchors(self):
        """Test initialization with anchor mappings."""
        anchors = {
            0: {0: "mouse1", 1: "mouse2"},
            100: {0: "mouse1", 1: "mouse2"},
        }
        resolver = TrackNameResolver(gt_anchors=anchors)
        assert resolver.get_anchor_frames() == [0, 100]

    def test_from_id_map(self):
        """Test creating resolver from ID map."""
        id_map = {
            0: {0: "mouse1", 1: "mouse2"},
            50: {1: "mouse1", 0: "mouse2"},  # Swap!
        }
        resolver = TrackNameResolver.from_id_map(id_map)
        assert resolver.get_anchor_frames() == [0, 50]

    def test_from_reconciler(self, simple_skeleton):
        """Test creating resolver from IDReconciler."""
        reconciler = IDReconciler(skeleton=simple_skeleton)
        reconciler._assignments = [
            TrackAssignment(
                frame_idx=0,
                pose_track_name="mouse1",
                pose_idx=0,
                sam3_obj_id=0,
                confidence=1.0,
            ),
            TrackAssignment(
                frame_idx=0,
                pose_track_name="mouse2",
                pose_idx=1,
                sam3_obj_id=1,
                confidence=1.0,
            ),
        ]

        resolver = TrackNameResolver.from_reconciler(reconciler)
        assert resolver.get_anchor_frames() == [0]
        assert resolver.get_track_name(0, 0) == "mouse1"
        assert resolver.get_track_name(0, 1) == "mouse2"

    def test_get_track_name_at_anchor(self):
        """Test getting track name at an anchor frame."""
        anchors = {0: {0: "mouse1", 1: "mouse2"}}
        resolver = TrackNameResolver(gt_anchors=anchors)

        assert resolver.get_track_name(0, 0) == "mouse1"
        assert resolver.get_track_name(0, 1) == "mouse2"

    def test_get_track_name_forward_propagation(self):
        """Test track names propagate forward from anchor."""
        anchors = {0: {0: "mouse1", 1: "mouse2"}}
        resolver = TrackNameResolver(gt_anchors=anchors)

        # Frames after anchor 0 should use anchor 0's mapping
        assert resolver.get_track_name(50, 0) == "mouse1"
        assert resolver.get_track_name(100, 1) == "mouse2"

    def test_get_track_name_backward_propagation(self):
        """Test track names propagate backward from anchor."""
        anchors = {100: {0: "mouse1", 1: "mouse2"}}
        resolver = TrackNameResolver(gt_anchors=anchors)

        # Frames before anchor 100 should use anchor 100's mapping
        assert resolver.get_track_name(0, 0) == "mouse1"
        assert resolver.get_track_name(50, 1) == "mouse2"

    def test_get_track_name_nearest_anchor(self):
        """Test that nearest anchor is used for mapping."""
        # Anchors at 0 and 100, with different mappings (swap)
        anchors = {
            0: {0: "mouse1", 1: "mouse2"},
            100: {0: "mouse2", 1: "mouse1"},  # Swapped IDs
        }
        resolver = TrackNameResolver(gt_anchors=anchors)

        # Frame 25: closer to anchor 0
        assert resolver.get_track_name(25, 0) == "mouse1"
        assert resolver.get_track_name(25, 1) == "mouse2"

        # Frame 75: closer to anchor 100
        assert resolver.get_track_name(75, 0) == "mouse2"
        assert resolver.get_track_name(75, 1) == "mouse1"

        # Frame 50: equidistant, should pick one (min picks first, which is 0)
        assert resolver.get_track_name(50, 0) == "mouse1"

    def test_get_track_name_fallback(self):
        """Test fallback names for unknown obj_ids."""
        anchors = {0: {0: "mouse1"}}
        fallback = {2: "extra_mouse"}
        resolver = TrackNameResolver(gt_anchors=anchors, fallback_names=fallback)

        # obj_id 2 not in anchor, should use fallback
        assert resolver.get_track_name(0, 2) == "extra_mouse"

    def test_get_track_name_generated_fallback(self):
        """Test generated fallback names when not in anchors or fallback."""
        anchors = {0: {0: "mouse1"}}
        resolver = TrackNameResolver(gt_anchors=anchors)

        # obj_id 5 not anywhere, should generate name
        assert resolver.get_track_name(0, 5) == "track_5"

    def test_get_track_name_custom_default(self):
        """Test custom default name."""
        anchors = {0: {0: "mouse1"}}
        resolver = TrackNameResolver(gt_anchors=anchors)

        assert resolver.get_track_name(0, 5, default="unknown") == "unknown"

    def test_get_track_name_no_anchors(self):
        """Test behavior with no anchors."""
        resolver = TrackNameResolver()

        # Should generate name since no anchors exist
        assert resolver.get_track_name(0, 0) == "track_0"

    def test_get_mapping_at_frame(self):
        """Test getting full mapping at a frame."""
        anchors = {
            0: {0: "mouse1", 1: "mouse2"},
            100: {0: "mouse2", 1: "mouse1"},
        }
        resolver = TrackNameResolver(gt_anchors=anchors)

        # At anchor
        assert resolver.get_mapping_at_frame(0) == {0: "mouse1", 1: "mouse2"}

        # Between anchors (closer to 0)
        assert resolver.get_mapping_at_frame(25) == {0: "mouse1", 1: "mouse2"}

        # Between anchors (closer to 100)
        assert resolver.get_mapping_at_frame(75) == {0: "mouse2", 1: "mouse1"}

    def test_get_mapping_at_frame_no_anchors(self):
        """Test mapping returns empty dict when no anchors."""
        resolver = TrackNameResolver()
        assert resolver.get_mapping_at_frame(50) == {}

    def test_resolve_all_frames(self):
        """Test resolving mappings for all frames."""
        anchors = {
            0: {0: "mouse1", 1: "mouse2"},
            10: {0: "mouse2", 1: "mouse1"},  # Swap at frame 10
        }
        resolver = TrackNameResolver(gt_anchors=anchors)

        all_mappings = resolver.resolve_all_frames(total_frames=15)

        # Should have mappings for all 15 frames
        assert len(all_mappings) == 15

        # Frames 0-5 use anchor 0 (closer to 0, or equidistant at frame 5)
        # Note: frame 5 is equidistant (dist 5 to both), min() picks first (anchor 0)
        for f in range(6):
            assert all_mappings[f][0] == "mouse1"

        # Frames 6-14 use anchor 10 (strictly closer to 10)
        for f in range(6, 15):
            assert all_mappings[f][0] == "mouse2"

    def test_resolve_all_frames_no_anchors(self):
        """Test resolve_all_frames with no anchors returns empty."""
        resolver = TrackNameResolver()
        assert resolver.resolve_all_frames(total_frames=100) == {}

    def test_get_all_track_names(self):
        """Test getting all unique track names."""
        anchors = {
            0: {0: "mouse1", 1: "mouse2"},
            100: {0: "mouse2", 1: "mouse1", 2: "mouse3"},
        }
        resolver = TrackNameResolver(gt_anchors=anchors)

        names = resolver.get_all_track_names()
        assert names == {"mouse1", "mouse2", "mouse3"}

    def test_get_all_sam3_obj_ids(self):
        """Test getting all unique SAM3 obj_ids."""
        anchors = {
            0: {0: "mouse1", 1: "mouse2"},
            100: {0: "mouse2", 2: "mouse3"},
        }
        resolver = TrackNameResolver(gt_anchors=anchors)

        obj_ids = resolver.get_all_sam3_obj_ids()
        assert obj_ids == {0, 1, 2}

    def test_get_anchor_source(self):
        """Test getting anchor source info for frames."""
        anchors = {
            10: {0: "mouse1"},
            50: {0: "mouse2"},
        }
        resolver = TrackNameResolver(gt_anchors=anchors)

        # At anchor
        assert resolver.get_anchor_source(10) == (10, "anchor")
        assert resolver.get_anchor_source(50) == (50, "anchor")

        # Before first anchor (backward from 10)
        assert resolver.get_anchor_source(5) == (10, "backward")

        # After anchor 10, before 50 (forward from 10)
        assert resolver.get_anchor_source(20) == (10, "forward")

        # After anchor 50 (forward from 50)
        assert resolver.get_anchor_source(60) == (50, "forward")

    def test_get_anchor_source_no_anchors(self):
        """Test anchor source with no anchors."""
        resolver = TrackNameResolver()
        assert resolver.get_anchor_source(0) == (None, "none")

    def test_get_canonical_mapping(self):
        """Test getting canonical global mapping."""
        anchors = {
            0: {0: "mouse1", 1: "mouse2"},
            100: {0: "mouse2", 1: "mouse1", 2: "mouse3"},
        }
        resolver = TrackNameResolver(gt_anchors=anchors)

        canonical = resolver.get_canonical_mapping()

        # Should have all obj_ids
        assert 0 in canonical
        assert 1 in canonical
        assert 2 in canonical

        # obj_ids 0 and 1 use first anchor's names
        assert canonical[0] == "mouse1"
        assert canonical[1] == "mouse2"

        # obj_id 2 only appears in second anchor
        assert canonical[2] == "mouse3"

    def test_get_canonical_mapping_empty(self):
        """Test canonical mapping with no anchors."""
        resolver = TrackNameResolver()
        assert resolver.get_canonical_mapping() == {}

    def test_flood_fill_scenario_from_readme(self):
        """Test the flood fill scenario described in the README.

        GT frames at 2 and 7, SAM swap occurs at frame 4.
        Frames 1-4 should use frame 2's mapping.
        Frames 5-8 should use frame 7's mapping.
        """
        anchors = {
            2: {1: "mouse1", 2: "mouse2"},  # Before swap
            7: {2: "mouse1", 1: "mouse2"},  # After swap (IDs swapped)
        }
        resolver = TrackNameResolver(gt_anchors=anchors)

        # Frames 1-4: closer to anchor 2
        # Frame 1: dist to 2 is 1, dist to 7 is 6 -> use anchor 2
        assert resolver.get_track_name(1, 1) == "mouse1"
        assert resolver.get_track_name(1, 2) == "mouse2"

        # Frame 4: dist to 2 is 2, dist to 7 is 3 -> use anchor 2
        assert resolver.get_track_name(4, 1) == "mouse1"
        assert resolver.get_track_name(4, 2) == "mouse2"

        # Frames 5-8: closer to anchor 7
        # Frame 5: dist to 2 is 3, dist to 7 is 2 -> use anchor 7
        assert resolver.get_track_name(5, 1) == "mouse2"  # Note: swapped!
        assert resolver.get_track_name(5, 2) == "mouse1"

        # Frame 8: dist to 2 is 6, dist to 7 is 1 -> use anchor 7
        assert resolver.get_track_name(8, 1) == "mouse2"
        assert resolver.get_track_name(8, 2) == "mouse1"
