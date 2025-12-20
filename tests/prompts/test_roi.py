"""Tests for ROI prompt handler."""

from pathlib import Path

import numpy as np
import pytest

from sam_track.prompts import (
    Prompt,
    PromptType,
    ROIPromptHandler,
    polygon_to_bbox,
    polygon_to_mask,
)

# Test data path
DATA_DIR = Path(__file__).parent.parent / "data"
ROI_FILE = DATA_DIR / "first_frame_mice.yml"


class TestPolygonToMask:
    """Tests for polygon_to_mask function."""

    def test_simple_square(self):
        """Test converting a simple square polygon to mask."""
        coords = [[10, 10], [20, 10], [20, 20], [10, 20]]
        mask = polygon_to_mask(coords, 100, 100)

        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        assert mask.max() == 1
        assert mask.min() == 0
        # Area includes boundary pixels, so 11x11 = 121 pixels
        assert 100 <= mask.sum() <= 150

    def test_triangle(self):
        """Test converting a triangle polygon to mask."""
        coords = [[50, 10], [90, 90], [10, 90]]
        mask = polygon_to_mask(coords, 100, 100)

        assert mask.shape == (100, 100)
        assert mask.sum() > 0  # Should have some pixels

    def test_respects_dimensions(self):
        """Test that mask has correct dimensions."""
        coords = [[10, 10], [20, 10], [20, 20], [10, 20]]
        mask = polygon_to_mask(coords, 200, 150)

        assert mask.shape == (150, 200)  # (height, width)


class TestPolygonToBbox:
    """Tests for polygon_to_bbox function."""

    def test_simple_square(self):
        """Test bounding box of a square."""
        coords = [[10, 20], [30, 20], [30, 40], [10, 40]]
        bbox = polygon_to_bbox(coords)

        assert bbox == (10.0, 20.0, 30.0, 40.0)

    def test_irregular_polygon(self):
        """Test bounding box of an irregular polygon."""
        coords = [[5, 10], [15, 5], [25, 15], [20, 25], [10, 20]]
        bbox = polygon_to_bbox(coords)

        assert bbox == (5.0, 5.0, 25.0, 25.0)


class TestROIPromptHandler:
    """Tests for ROIPromptHandler."""

    @pytest.fixture
    def handler(self):
        """Create handler with test data."""
        return ROIPromptHandler(ROI_FILE)

    def test_init(self, handler):
        """Test handler initialization."""
        assert handler.path == ROI_FILE
        assert handler.use_masks is True
        assert handler.include_boxes is False

    def test_prompt_type(self, handler):
        """Test prompt type is ROI."""
        assert handler.prompt_type == PromptType.ROI

    def test_resolution(self, handler):
        """Test resolution property."""
        assert handler.resolution == (1024, 768)

    def test_frame_idx(self, handler):
        """Test frame index property."""
        assert handler.frame_idx == 0

    def test_source_video(self, handler):
        """Test source video property."""
        assert handler.source_video == "mice.mp4"

    def test_roi_count(self, handler):
        """Test ROI count property."""
        assert handler.roi_count == 2

    def test_load_returns_prompt(self, handler):
        """Test that load returns a Prompt object."""
        prompt = handler.load()
        assert isinstance(prompt, Prompt)

    def test_load_prompt_type(self, handler):
        """Test loaded prompt has correct type."""
        prompt = handler.load()
        assert prompt.prompt_type == PromptType.ROI

    def test_load_obj_ids(self, handler):
        """Test loaded prompt has correct object IDs."""
        prompt = handler.load()
        assert prompt.obj_ids == [1, 2]

    def test_load_obj_names(self, handler):
        """Test loaded prompt has object name mapping."""
        prompt = handler.load()
        assert prompt.obj_names == {1: "mouse1", 2: "mouse2"}

    def test_load_masks(self, handler):
        """Test loaded prompt has masks."""
        prompt = handler.load()

        assert prompt.masks is not None
        assert len(prompt.masks) == 2

        # Check mask properties
        for mask in prompt.masks:
            assert mask.shape == (768, 1024)
            assert mask.dtype == np.uint8
            assert mask.sum() > 0

    def test_load_masks_have_correct_area(self, handler):
        """Test that mask areas approximately match YAML properties."""
        prompt = handler.load()

        # From YAML: mouse1 area ~18815, mouse2 area ~5532
        # Mask areas will be slightly different due to rasterization
        assert 18000 <= prompt.masks[0].sum() <= 20000
        assert 5000 <= prompt.masks[1].sum() <= 6000

    def test_load_source_path(self, handler):
        """Test loaded prompt has source path."""
        prompt = handler.load()
        assert prompt.source_path == ROI_FILE

    def test_load_with_boxes(self):
        """Test loading with include_boxes=True."""
        handler = ROIPromptHandler(ROI_FILE, include_boxes=True)
        prompt = handler.load()

        assert prompt.boxes is not None
        assert len(prompt.boxes) == 2

        # Check box format
        for box in prompt.boxes:
            assert len(box) == 4
            x_min, y_min, x_max, y_max = box
            assert x_min < x_max
            assert y_min < y_max

    def test_load_without_masks(self):
        """Test loading with use_masks=False."""
        handler = ROIPromptHandler(ROI_FILE, use_masks=False, include_boxes=True)
        prompt = handler.load()

        assert prompt.masks is None
        assert prompt.boxes is not None

    def test_get_name(self, handler):
        """Test get_name method on loaded prompt."""
        prompt = handler.load()

        assert prompt.get_name(1) == "mouse1"
        assert prompt.get_name(2) == "mouse2"
        assert prompt.get_name(99) == "object_99"  # Unknown ID

    def test_num_objects(self, handler):
        """Test num_objects property."""
        prompt = handler.load()
        assert prompt.num_objects == 2

    def test_repr(self, handler):
        """Test string representation."""
        repr_str = repr(handler)
        assert "ROIPromptHandler" in repr_str
        assert "first_frame_mice.yml" in repr_str

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        handler = ROIPromptHandler("nonexistent.yml")
        with pytest.raises(FileNotFoundError):
            handler.load()
