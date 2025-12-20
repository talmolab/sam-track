"""Tests for pose prompt handler."""

from pathlib import Path

import numpy as np
import pytest

from sam_track.prompts import PosePromptHandler, Prompt, PromptType

# Test data paths
DATA_DIR = Path(__file__).parent.parent / "data"
SLP_3NODE = DATA_DIR / "labels.3node.first_frame.slp"
SLP_15NODE = DATA_DIR / "labels.15node.first_frame.slp"


class TestPosePromptHandler:
    """Tests for PosePromptHandler."""

    @pytest.fixture
    def handler_3node(self):
        """Create handler with 3-node test data."""
        return PosePromptHandler(SLP_3NODE)

    @pytest.fixture
    def handler_15node(self):
        """Create handler with 15-node test data."""
        return PosePromptHandler(SLP_15NODE)

    def test_init(self, handler_3node):
        """Test handler initialization."""
        assert handler_3node.path == SLP_3NODE
        assert handler_3node.frame_idx is None
        assert handler_3node.nodes is None

    def test_init_with_options(self):
        """Test initialization with options."""
        handler = PosePromptHandler(
            SLP_3NODE,
            frame_idx=0,
            nodes=["snout", "neck"],
        )
        assert handler.frame_idx == 0
        assert handler.nodes == ["snout", "neck"]

    def test_prompt_type(self, handler_3node):
        """Test prompt type is POSE."""
        assert handler_3node.prompt_type == PromptType.POSE

    def test_skeleton_property(self, handler_3node):
        """Test skeleton property."""
        skeleton = handler_3node.skeleton
        assert skeleton is not None
        assert len(skeleton.nodes) == 3

    def test_node_names_3node(self, handler_3node):
        """Test node names for 3-node skeleton."""
        assert handler_3node.node_names == ["snout", "neck", "tail_base"]

    def test_node_names_15node(self, handler_15node):
        """Test node names for 15-node skeleton."""
        names = handler_15node.node_names
        assert len(names) == 15
        assert "nose" in names
        assert "head" in names
        assert "tail_tip" in names

    def test_num_labeled_frames(self, handler_3node):
        """Test number of labeled frames."""
        assert handler_3node.num_labeled_frames == 1

    def test_num_tracks(self, handler_3node):
        """Test number of tracks (should be 0 for untracked)."""
        assert handler_3node.num_tracks == 0

    def test_load_returns_prompt(self, handler_3node):
        """Test that load returns a Prompt object."""
        prompt = handler_3node.load()
        assert isinstance(prompt, Prompt)

    def test_load_prompt_type(self, handler_3node):
        """Test loaded prompt has correct type."""
        prompt = handler_3node.load()
        assert prompt.prompt_type == PromptType.POSE

    def test_load_obj_ids(self, handler_3node):
        """Test loaded prompt has object IDs."""
        prompt = handler_3node.load()
        # Two instances at frame 0, both untracked so IDs are 0, 1
        assert prompt.obj_ids == [0, 1]

    def test_load_obj_names_untracked(self, handler_3node):
        """Test object names for untracked instances."""
        prompt = handler_3node.load()
        assert prompt.obj_names == {0: "instance_0", 1: "instance_1"}

    def test_load_frame_idx(self, handler_3node):
        """Test loaded prompt has correct frame index."""
        prompt = handler_3node.load()
        assert prompt.frame_idx == 0

    def test_load_points(self, handler_3node):
        """Test loaded prompt has points."""
        prompt = handler_3node.load()

        assert prompt.points is not None
        assert len(prompt.points) == 2  # Two instances

        # First instance has all 3 nodes visible
        assert len(prompt.points[0]) == 3

        # Second instance has 2 nodes visible (snout missing)
        assert len(prompt.points[1]) == 2

    def test_load_points_format(self, handler_3node):
        """Test that points are (x, y) tuples."""
        prompt = handler_3node.load()

        for point_set in prompt.points:
            for point in point_set:
                assert isinstance(point, tuple)
                assert len(point) == 2
                x, y = point
                assert isinstance(x, float)
                assert isinstance(y, float)

    def test_load_points_values(self, handler_3node):
        """Test that point values are reasonable."""
        prompt = handler_3node.load()

        for point_set in prompt.points:
            for x, y in point_set:
                # Points should be positive pixel coordinates
                assert x > 0
                assert y > 0
                # Should be within a reasonable range (< 2000 for typical video)
                assert x < 2000
                assert y < 2000

    def test_load_source_path(self, handler_3node):
        """Test loaded prompt has source path."""
        prompt = handler_3node.load()
        assert prompt.source_path == SLP_3NODE

    def test_load_15node(self, handler_15node):
        """Test loading 15-node skeleton file."""
        prompt = handler_15node.load()

        assert prompt.num_objects == 2
        assert len(prompt.points) == 2

        # Instance 0: 14/15 nodes visible
        assert len(prompt.points[0]) == 14

        # Instance 1: 11/15 nodes visible
        assert len(prompt.points[1]) == 11

    def test_load_with_node_filter(self):
        """Test loading with node filter."""
        handler = PosePromptHandler(
            SLP_3NODE,
            nodes=["snout", "neck"],
        )
        prompt = handler.load()

        # First instance: both snout and neck visible
        assert len(prompt.points[0]) == 2

        # Second instance: only neck visible (snout is NaN)
        assert len(prompt.points[1]) == 1

    def test_load_with_single_node_filter(self):
        """Test loading with single node filter."""
        handler = PosePromptHandler(
            SLP_3NODE,
            nodes=["tail_base"],
        )
        prompt = handler.load()

        # Both instances should have tail_base
        assert len(prompt.points[0]) == 1
        assert len(prompt.points[1]) == 1

    def test_load_with_nonexistent_node_filter(self):
        """Test that filtering to nonexistent nodes raises error."""
        handler = PosePromptHandler(
            SLP_3NODE,
            nodes=["nonexistent_node"],
        )
        with pytest.raises(ValueError, match="No instances with visible points"):
            handler.load()

    def test_load_with_frame_idx(self):
        """Test loading with specific frame index."""
        handler = PosePromptHandler(SLP_3NODE, frame_idx=0)
        prompt = handler.load()
        assert prompt.frame_idx == 0

    def test_load_with_invalid_frame_idx(self):
        """Test error when frame index doesn't exist."""
        handler = PosePromptHandler(SLP_3NODE, frame_idx=999)
        with pytest.raises(ValueError, match="No labels at frame 999"):
            handler.load()

    def test_get_name(self, handler_3node):
        """Test get_name method on loaded prompt."""
        prompt = handler_3node.load()

        assert prompt.get_name(0) == "instance_0"
        assert prompt.get_name(1) == "instance_1"
        assert prompt.get_name(99) == "object_99"  # Unknown ID

    def test_num_objects(self, handler_3node):
        """Test num_objects property."""
        prompt = handler_3node.load()
        assert prompt.num_objects == 2

    def test_repr(self, handler_3node):
        """Test string representation."""
        repr_str = repr(handler_3node)
        assert "PosePromptHandler" in repr_str
        assert "labels.3node.first_frame.slp" in repr_str

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        handler = PosePromptHandler("nonexistent.slp")
        with pytest.raises(FileNotFoundError):
            handler.load()


class TestPosePromptValidation:
    """Tests for Prompt validation with points."""

    def test_points_count_mismatch(self):
        """Test that mismatched points count raises error."""
        with pytest.raises(ValueError, match="Number of point sets"):
            Prompt(
                prompt_type=PromptType.POSE,
                obj_ids=[0, 1],
                points=[[(0, 0)]],  # Only 1 point set for 2 objects
            )

    def test_valid_points(self):
        """Test valid points configuration."""
        prompt = Prompt(
            prompt_type=PromptType.POSE,
            obj_ids=[0, 1],
            points=[
                [(10, 20), (30, 40)],
                [(50, 60)],
            ],
        )
        assert len(prompt.points) == 2
        assert len(prompt.points[0]) == 2
        assert len(prompt.points[1]) == 1
