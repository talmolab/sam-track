"""Tests for mask prompt handler."""

import numpy as np
import pytest

from sam_track.outputs.seg import SegmentationWriter
from sam_track.prompts import MaskPromptHandler, Prompt, PromptType
from sam_track.tracker import TrackingResult


@pytest.fixture
def sample_seg_file(tmp_path):
    """Create a sample seg.h5 file for testing."""
    output_path = tmp_path / "test.seg.h5"

    # Create specific mask patterns for verification
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[20:40, 30:60] = 1

    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[50:80, 10:90] = 1

    result_0 = TrackingResult(
        frame_idx=0,
        object_ids=np.array([1, 2]),
        masks=np.stack([mask1, mask2]),
        boxes=np.array([[30.0, 20.0, 60.0, 40.0], [10.0, 50.0, 90.0, 80.0]]),
        scores=np.array([0.95, 0.88]),
    )

    result_10 = TrackingResult(
        frame_idx=10,
        object_ids=np.array([1]),
        masks=np.random.randint(0, 2, (1, 100, 100)).astype(np.uint8),
        boxes=np.array([[15.0, 25.0, 55.0, 65.0]]),
        scores=np.array([0.92]),
    )

    with SegmentationWriter(
        output_path=output_path,
        video_path="video.mp4",
        video_width=100,
        video_height=100,
        fps=30.0,
        total_frames=100,
        obj_names={1: "mouse_1", 2: "mouse_2"},
    ) as writer:
        writer.add_result(result_0)
        writer.add_result(result_10)

    return output_path


@pytest.fixture
def single_frame_seg_file(tmp_path):
    """Create a seg.h5 file with only one frame."""
    output_path = tmp_path / "single.seg.h5"

    result = TrackingResult(
        frame_idx=5,
        object_ids=np.array([0]),
        masks=np.random.randint(0, 2, (1, 50, 50)).astype(np.uint8),
        boxes=np.array([[10.0, 10.0, 40.0, 40.0]]),
        scores=np.array([0.9]),
    )

    with SegmentationWriter(
        output_path=output_path,
        video_path="video.mp4",
        video_width=50,
        video_height=50,
        obj_names={0: "animal"},
    ) as writer:
        writer.add_result(result)

    return output_path


@pytest.fixture
def empty_frame_seg_file(tmp_path):
    """Create a seg.h5 file with an empty frame."""
    output_path = tmp_path / "empty_frame.seg.h5"

    result_0 = TrackingResult(
        frame_idx=0,
        object_ids=np.array([1]),
        masks=np.random.randint(0, 2, (1, 50, 50)).astype(np.uint8),
        boxes=np.array([[10.0, 20.0, 30.0, 40.0]]),
        scores=np.array([0.9]),
    )

    result_empty = TrackingResult(
        frame_idx=5,
        object_ids=np.array([], dtype=np.int64),
        masks=np.zeros((0, 50, 50), dtype=np.uint8),
        boxes=np.zeros((0, 4)),
        scores=np.array([]),
    )

    with SegmentationWriter(
        output_path=output_path,
        video_path="video.mp4",
        video_width=50,
        video_height=50,
    ) as writer:
        writer.add_result(result_0)
        writer.add_result(result_empty)

    return output_path


class TestMaskPromptHandler:
    """Tests for MaskPromptHandler."""

    def test_file_not_found(self, tmp_path):
        """Test FileNotFoundError for missing file."""
        handler = MaskPromptHandler(tmp_path / "nonexistent.seg.h5")
        with pytest.raises(FileNotFoundError):
            handler.load()

    def test_init(self, sample_seg_file):
        """Test handler initialization."""
        handler = MaskPromptHandler(sample_seg_file)
        assert handler.path == sample_seg_file
        assert handler.frame_idx is None
        assert handler.target_resolution is None

    def test_init_with_options(self, sample_seg_file):
        """Test initialization with options."""
        handler = MaskPromptHandler(
            sample_seg_file,
            frame_idx=10,
            target_resolution=(200, 200),
        )
        assert handler.frame_idx == 10
        assert handler.target_resolution == (200, 200)

    def test_prompt_type(self, sample_seg_file):
        """Test prompt type is ROI (reusing existing mask type)."""
        handler = MaskPromptHandler(sample_seg_file)
        assert handler.prompt_type == PromptType.ROI

    def test_resolution_property(self, sample_seg_file):
        """Test resolution property."""
        handler = MaskPromptHandler(sample_seg_file)
        assert handler.resolution == (100, 100)

    def test_num_frames_property(self, sample_seg_file):
        """Test num_frames property."""
        handler = MaskPromptHandler(sample_seg_file)
        assert handler.num_frames == 2

    def test_num_tracks_property(self, sample_seg_file):
        """Test num_tracks property."""
        handler = MaskPromptHandler(sample_seg_file)
        assert handler.num_tracks == 2

    def test_labeled_frame_indices(self, sample_seg_file):
        """Test labeled_frame_indices property."""
        handler = MaskPromptHandler(sample_seg_file)
        assert handler.labeled_frame_indices == [0, 10]

    def test_load_returns_prompt(self, sample_seg_file):
        """Test that load returns a Prompt object."""
        handler = MaskPromptHandler(sample_seg_file)
        prompt = handler.load()
        assert isinstance(prompt, Prompt)

    def test_load_prompt_type(self, sample_seg_file):
        """Test loaded prompt has correct type."""
        handler = MaskPromptHandler(sample_seg_file)
        prompt = handler.load()
        assert prompt.prompt_type == PromptType.ROI

    def test_load_first_frame(self, sample_seg_file):
        """Test load uses first labeled frame by default."""
        handler = MaskPromptHandler(sample_seg_file)
        prompt = handler.load()
        assert prompt.frame_idx == 0

    def test_load_specific_frame(self, sample_seg_file):
        """Test load with specific frame_idx."""
        handler = MaskPromptHandler(sample_seg_file, frame_idx=10)
        prompt = handler.load()
        assert prompt.frame_idx == 10
        assert len(prompt.obj_ids) == 1

    def test_load_obj_ids(self, sample_seg_file):
        """Test loaded prompt has correct object IDs."""
        handler = MaskPromptHandler(sample_seg_file)
        prompt = handler.load()
        assert prompt.obj_ids == [1, 2]

    def test_load_obj_names(self, sample_seg_file):
        """Test loaded prompt has correct object names."""
        handler = MaskPromptHandler(sample_seg_file)
        prompt = handler.load()
        assert prompt.obj_names == {1: "mouse_1", 2: "mouse_2"}

    def test_load_masks(self, sample_seg_file):
        """Test loaded prompt has masks."""
        handler = MaskPromptHandler(sample_seg_file)
        prompt = handler.load()

        assert prompt.masks is not None
        assert len(prompt.masks) == 2

        # Check mask shapes
        for mask in prompt.masks:
            assert mask.shape == (100, 100)
            assert mask.dtype == np.uint8

    def test_load_masks_values(self, sample_seg_file):
        """Test that loaded masks have correct values."""
        handler = MaskPromptHandler(sample_seg_file)
        prompt = handler.load()

        # First mask should have 1s in region [20:40, 30:60]
        mask1 = prompt.masks[0]
        assert mask1[30, 45] == 1  # Inside region
        assert mask1[0, 0] == 0  # Outside region

        # Second mask should have 1s in region [50:80, 10:90]
        mask2 = prompt.masks[1]
        assert mask2[65, 50] == 1  # Inside region
        assert mask2[0, 0] == 0  # Outside region

    def test_load_source_path(self, sample_seg_file):
        """Test loaded prompt has source path."""
        handler = MaskPromptHandler(sample_seg_file)
        prompt = handler.load()
        assert prompt.source_path == sample_seg_file

    def test_load_missing_frame(self, sample_seg_file):
        """Test load raises for non-existent frame."""
        handler = MaskPromptHandler(sample_seg_file, frame_idx=999)
        with pytest.raises(ValueError, match="No masks at frame 999"):
            handler.load()

    def test_get_prompt(self, sample_seg_file):
        """Test get_prompt method."""
        handler = MaskPromptHandler(sample_seg_file)

        prompt_0 = handler.get_prompt(0)
        assert prompt_0 is not None
        assert prompt_0.frame_idx == 0
        assert len(prompt_0.masks) == 2

        prompt_10 = handler.get_prompt(10)
        assert prompt_10 is not None
        assert prompt_10.frame_idx == 10
        assert len(prompt_10.masks) == 1

    def test_get_prompt_missing_frame(self, sample_seg_file):
        """Test get_prompt returns None for missing frame."""
        handler = MaskPromptHandler(sample_seg_file)
        assert handler.get_prompt(999) is None

    def test_get_prompt_empty_frame(self, empty_frame_seg_file):
        """Test get_prompt returns None for empty frame."""
        handler = MaskPromptHandler(empty_frame_seg_file)
        assert handler.get_prompt(5) is None

    def test_single_frame_file(self, single_frame_seg_file):
        """Test handling single-frame file."""
        handler = MaskPromptHandler(single_frame_seg_file)
        prompt = handler.load()

        assert prompt.frame_idx == 5
        assert prompt.obj_ids == [0]
        assert prompt.obj_names == {0: "animal"}
        assert len(prompt.masks) == 1

    def test_mask_resizing(self, sample_seg_file):
        """Test mask resizing when target_resolution differs."""
        handler = MaskPromptHandler(
            sample_seg_file,
            target_resolution=(50, 50),  # Half the original size
        )
        prompt = handler.load()

        # Masks should be resized to 50x50
        for mask in prompt.masks:
            assert mask.shape == (50, 50)

    def test_mask_resizing_upscale(self, sample_seg_file):
        """Test mask upscaling when target is larger."""
        handler = MaskPromptHandler(
            sample_seg_file,
            target_resolution=(200, 200),
        )
        prompt = handler.load()

        for mask in prompt.masks:
            assert mask.shape == (200, 200)

    def test_no_resize_when_same_resolution(self, sample_seg_file):
        """Test no resize when target matches source."""
        handler = MaskPromptHandler(
            sample_seg_file,
            target_resolution=(100, 100),  # Same as source
        )
        prompt = handler.load()

        for mask in prompt.masks:
            assert mask.shape == (100, 100)

    def test_context_manager(self, sample_seg_file):
        """Test context manager usage."""
        with MaskPromptHandler(sample_seg_file) as handler:
            prompt = handler.load()
            assert len(prompt.masks) == 2

    def test_repr(self, sample_seg_file):
        """Test string representation."""
        handler = MaskPromptHandler(sample_seg_file)

        repr_str = repr(handler)
        assert "MaskPromptHandler" in repr_str
        assert "num_frames=2" in repr_str
        assert "num_tracks=2" in repr_str

    def test_repr_before_load(self, tmp_path):
        """Test repr handles unopened file gracefully."""
        # File doesn't exist, so repr should still work
        handler = MaskPromptHandler(tmp_path / "nonexistent.seg.h5")
        repr_str = repr(handler)
        assert "MaskPromptHandler" in repr_str

    def test_multi_frame_iteration(self, sample_seg_file):
        """Test iterating over all labeled frames."""
        handler = MaskPromptHandler(sample_seg_file)

        prompts = []
        for frame_idx in handler.labeled_frame_indices:
            prompt = handler.get_prompt(frame_idx)
            if prompt is not None:
                prompts.append(prompt)

        assert len(prompts) == 2
        assert prompts[0].frame_idx == 0
        assert prompts[1].frame_idx == 10

    def test_masks_are_binary(self, sample_seg_file):
        """Test that masks are binary (0 or 1 only)."""
        handler = MaskPromptHandler(sample_seg_file)
        prompt = handler.load()

        for mask in prompt.masks:
            unique_values = np.unique(mask)
            assert all(v in [0, 1] for v in unique_values)
