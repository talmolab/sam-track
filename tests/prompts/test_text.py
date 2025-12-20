"""Tests for text prompt handler."""

import pytest

from sam_track.prompts import Prompt, PromptType, TextPromptHandler


class TestTextPromptHandler:
    """Tests for TextPromptHandler."""

    def test_init(self):
        """Test handler initialization."""
        handler = TextPromptHandler("mouse")
        assert handler.text == "mouse"
        assert handler._frame_idx == 0

    def test_init_with_frame_idx(self):
        """Test handler initialization with custom frame index."""
        handler = TextPromptHandler("fly", frame_idx=10)
        assert handler.text == "fly"
        assert handler._frame_idx == 10

    def test_prompt_type(self):
        """Test prompt type is TEXT."""
        handler = TextPromptHandler("mouse")
        assert handler.prompt_type == PromptType.TEXT

    def test_load_returns_prompt(self):
        """Test that load returns a Prompt object."""
        handler = TextPromptHandler("mouse")
        prompt = handler.load()
        assert isinstance(prompt, Prompt)

    def test_load_prompt_type(self):
        """Test loaded prompt has correct type."""
        handler = TextPromptHandler("mouse")
        prompt = handler.load()
        assert prompt.prompt_type == PromptType.TEXT

    def test_load_text(self):
        """Test loaded prompt has correct text."""
        handler = TextPromptHandler("mouse")
        prompt = handler.load()
        assert prompt.text == "mouse"

    def test_load_frame_idx(self):
        """Test loaded prompt has correct frame index."""
        handler = TextPromptHandler("mouse", frame_idx=5)
        prompt = handler.load()
        assert prompt.frame_idx == 5

    def test_load_empty_obj_ids(self):
        """Test loaded prompt has empty obj_ids (SAM3 assigns them)."""
        handler = TextPromptHandler("mouse")
        prompt = handler.load()
        assert prompt.obj_ids == []

    def test_load_no_masks(self):
        """Test loaded prompt has no masks."""
        handler = TextPromptHandler("mouse")
        prompt = handler.load()
        assert prompt.masks is None

    def test_load_no_boxes(self):
        """Test loaded prompt has no boxes."""
        handler = TextPromptHandler("mouse")
        prompt = handler.load()
        assert prompt.boxes is None

    def test_repr(self):
        """Test string representation."""
        handler = TextPromptHandler("mouse", frame_idx=5)
        repr_str = repr(handler)
        assert "TextPromptHandler" in repr_str
        assert "mouse" in repr_str
        assert "5" in repr_str


class TestPromptValidation:
    """Tests for Prompt validation."""

    def test_text_prompt_requires_text(self):
        """Test that text prompt requires text field."""
        with pytest.raises(ValueError, match="Text prompt requires"):
            Prompt(prompt_type=PromptType.TEXT, text=None)

    def test_roi_prompt_requires_obj_ids(self):
        """Test that ROI prompt requires obj_ids."""
        with pytest.raises(ValueError, match="requires 'obj_ids'"):
            Prompt(prompt_type=PromptType.ROI, obj_ids=[])

    def test_mask_count_must_match_obj_ids(self):
        """Test that mask count must match obj_ids count."""
        import numpy as np

        masks = [np.zeros((10, 10), dtype=np.uint8)]
        with pytest.raises(ValueError, match="Number of masks"):
            Prompt(
                prompt_type=PromptType.ROI,
                obj_ids=[1, 2],  # 2 obj_ids
                masks=masks,  # 1 mask
            )

    def test_valid_roi_prompt(self):
        """Test creating a valid ROI prompt."""
        import numpy as np

        masks = [
            np.zeros((10, 10), dtype=np.uint8),
            np.zeros((10, 10), dtype=np.uint8),
        ]
        prompt = Prompt(
            prompt_type=PromptType.ROI,
            obj_ids=[1, 2],
            obj_names={1: "mouse1", 2: "mouse2"},
            masks=masks,
        )
        assert prompt.num_objects == 2
        assert prompt.get_name(1) == "mouse1"
