"""Tests for SAM3 tracker module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from sam_track.prompts.base import Prompt, PromptType
from sam_track.tracker import DEFAULT_MODEL_ID, SAM3Tracker, TrackingResult


class TestTrackingResult:
    """Tests for TrackingResult dataclass."""

    def test_basic_creation(self):
        """Test creating a TrackingResult."""
        result = TrackingResult(
            frame_idx=0,
            object_ids=np.array([1, 2]),
            masks=np.zeros((2, 100, 100), dtype=bool),
            boxes=np.array([[10, 10, 50, 50], [60, 60, 90, 90]]),
            scores=np.array([0.9, 0.8]),
        )
        assert result.frame_idx == 0
        assert result.num_objects == 2

    def test_num_objects(self):
        """Test num_objects property."""
        result = TrackingResult(
            frame_idx=5,
            object_ids=np.array([1, 2, 3]),
            masks=np.zeros((3, 100, 100), dtype=bool),
            boxes=np.zeros((3, 4)),
            scores=np.ones(3),
        )
        assert result.num_objects == 3

    def test_empty_result(self):
        """Test TrackingResult with no objects."""
        result = TrackingResult(
            frame_idx=0,
            object_ids=np.array([]),
            masks=np.zeros((0, 100, 100), dtype=bool),
            boxes=np.zeros((0, 4)),
            scores=np.array([]),
        )
        assert result.num_objects == 0


class TestSAM3TrackerInit:
    """Tests for SAM3Tracker initialization."""

    def test_default_init(self):
        """Test default initialization."""
        tracker = SAM3Tracker()
        assert tracker.model_id == DEFAULT_MODEL_ID
        assert tracker.dtype == torch.bfloat16
        assert not tracker.is_session_active

    def test_custom_model_id(self):
        """Test initialization with custom model ID."""
        tracker = SAM3Tracker(model_id="custom/model")
        assert tracker.model_id == "custom/model"

    def test_custom_device(self):
        """Test initialization with custom device."""
        tracker = SAM3Tracker(device="cpu")
        assert tracker.device == torch.device("cpu")

    def test_custom_dtype(self):
        """Test initialization with custom dtype."""
        tracker = SAM3Tracker(dtype=torch.float32)
        assert tracker.dtype == torch.float32

    def test_device_auto_detection(self):
        """Test that device is auto-detected."""
        tracker = SAM3Tracker()
        # Should be one of cuda, mps, or cpu
        assert tracker.device.type in ("cuda", "mps", "cpu")


class TestSAM3TrackerSessionErrors:
    """Tests for SAM3Tracker session error handling."""

    def test_add_text_prompt_without_session(self):
        """Test add_text_prompt raises error without active session."""
        tracker = SAM3Tracker()
        with pytest.raises(RuntimeError, match="No active session"):
            tracker.add_text_prompt("mouse")

    def test_add_prompt_without_session(self):
        """Test add_prompt raises error without active session."""
        tracker = SAM3Tracker()
        prompt = Prompt(
            prompt_type=PromptType.ROI,
            obj_ids=[1],
            masks=[np.zeros((100, 100), dtype=np.uint8)],
        )
        with pytest.raises(RuntimeError, match="No active session"):
            tracker.add_prompt(prompt)

    def test_propagate_without_session(self):
        """Test propagate raises error without active session."""
        tracker = SAM3Tracker()
        with pytest.raises(RuntimeError, match="No active session"):
            list(tracker.propagate())

    def test_init_session_empty_video(self):
        """Test init_session raises error with empty video."""
        tracker = SAM3Tracker()
        with pytest.raises(ValueError, match="at least one frame"):
            tracker.init_session([])


class TestSAM3TrackerModeErrors:
    """Tests for SAM3Tracker mode mismatch errors."""

    @patch("sam_track.tracker.SAM3Tracker._load_visual_model")
    def test_add_text_prompt_in_visual_mode(self, mock_load):
        """Test add_text_prompt raises error in visual mode."""
        tracker = SAM3Tracker()
        # Mock the session initialization
        tracker._inference_session = MagicMock()
        tracker._use_text = False

        with pytest.raises(RuntimeError, match="visual prompts"):
            tracker.add_text_prompt("mouse")

    @patch("sam_track.tracker.SAM3Tracker._load_text_model")
    def test_add_prompt_in_text_mode(self, mock_load):
        """Test add_prompt raises error in text mode."""
        tracker = SAM3Tracker()
        # Mock the session initialization
        tracker._inference_session = MagicMock()
        tracker._use_text = True

        prompt = Prompt(
            prompt_type=PromptType.ROI,
            obj_ids=[1],
            masks=[np.zeros((100, 100), dtype=np.uint8)],
        )
        with pytest.raises(RuntimeError, match="text prompts"):
            tracker.add_prompt(prompt)

    def test_add_prompt_with_text_type(self):
        """Test add_prompt raises error for TEXT prompt type."""
        tracker = SAM3Tracker()
        tracker._inference_session = MagicMock()
        tracker._use_text = False

        prompt = Prompt(prompt_type=PromptType.TEXT, text="mouse")
        with pytest.raises(ValueError, match="Use add_text_prompt"):
            tracker.add_prompt(prompt)


class TestSAM3TrackerPromptValidation:
    """Tests for SAM3Tracker prompt validation."""

    def test_add_prompt_without_visual_data(self):
        """Test add_prompt raises error when no visual data provided."""
        tracker = SAM3Tracker()
        tracker._inference_session = MagicMock()
        tracker._use_text = False
        tracker._visual_processor = MagicMock()

        # Prompt with no masks, points, or boxes
        prompt = Prompt(
            prompt_type=PromptType.ROI,
            obj_ids=[1],
            masks=None,
            points=None,
            boxes=None,
        )
        with pytest.raises(ValueError, match="No valid prompt data"):
            tracker.add_prompt(prompt)


class TestSAM3TrackerComputeBoxes:
    """Tests for bounding box computation from masks."""

    def test_compute_boxes_from_masks(self):
        """Test computing bounding boxes from masks."""
        tracker = SAM3Tracker()

        # Create a simple mask with known bounds
        masks = np.zeros((1, 100, 100), dtype=bool)
        masks[0, 20:40, 30:60] = True  # y: 20-39, x: 30-59

        boxes = tracker._compute_boxes_from_masks(masks)

        assert boxes.shape == (1, 4)
        assert boxes[0, 0] == 30.0  # x_min
        assert boxes[0, 1] == 20.0  # y_min
        assert boxes[0, 2] == 59.0  # x_max
        assert boxes[0, 3] == 39.0  # y_max

    def test_compute_boxes_empty_mask(self):
        """Test computing bounding box from empty mask."""
        tracker = SAM3Tracker()

        masks = np.zeros((1, 100, 100), dtype=bool)
        boxes = tracker._compute_boxes_from_masks(masks)

        assert boxes.shape == (1, 4)
        assert np.all(boxes[0] == 0.0)

    def test_compute_boxes_multiple_masks(self):
        """Test computing bounding boxes from multiple masks."""
        tracker = SAM3Tracker()

        masks = np.zeros((2, 100, 100), dtype=bool)
        masks[0, 10:20, 10:30] = True  # First object
        masks[1, 50:80, 60:90] = True  # Second object

        boxes = tracker._compute_boxes_from_masks(masks)

        assert boxes.shape == (2, 4)
        # First box
        assert boxes[0, 0] == 10.0  # x_min
        assert boxes[0, 1] == 10.0  # y_min
        assert boxes[0, 2] == 29.0  # x_max
        assert boxes[0, 3] == 19.0  # y_max
        # Second box
        assert boxes[1, 0] == 60.0  # x_min
        assert boxes[1, 1] == 50.0  # y_min
        assert boxes[1, 2] == 89.0  # x_max
        assert boxes[1, 3] == 79.0  # y_max

    def test_compute_boxes_no_masks(self):
        """Test computing bounding boxes with no masks."""
        tracker = SAM3Tracker()

        masks = np.zeros((0, 100, 100), dtype=bool)
        boxes = tracker._compute_boxes_from_masks(masks)

        assert boxes.shape == (0, 4)


class TestSAM3TrackerContextManager:
    """Tests for SAM3Tracker context manager."""

    def test_context_manager(self):
        """Test using tracker as context manager."""
        with SAM3Tracker() as tracker:
            assert isinstance(tracker, SAM3Tracker)

    def test_context_manager_closes_session(self):
        """Test context manager closes session on exit."""
        tracker = SAM3Tracker()
        tracker._inference_session = MagicMock()

        with tracker:
            assert tracker._inference_session is not None

        assert tracker._inference_session is None


class TestSAM3TrackerReset:
    """Tests for SAM3Tracker reset functionality."""

    def test_reset_calls_session_reset(self):
        """Test reset calls inference session reset."""
        tracker = SAM3Tracker()
        mock_session = MagicMock()
        tracker._inference_session = mock_session

        tracker.reset()

        mock_session.reset_inference_session.assert_called_once()

    def test_reset_without_session(self):
        """Test reset does nothing without session."""
        tracker = SAM3Tracker()
        # Should not raise
        tracker.reset()


class TestSAM3TrackerAddPromptFormats:
    """Tests for SAM3Tracker prompt format handling."""

    def test_add_prompt_with_mask(self):
        """Test add_prompt with mask input."""
        tracker = SAM3Tracker()
        tracker._inference_session = MagicMock()
        tracker._use_text = False
        mock_processor = MagicMock()
        tracker._visual_processor = mock_processor

        mask = np.ones((100, 100), dtype=np.uint8)
        prompt = Prompt(
            prompt_type=PromptType.ROI,
            frame_idx=0,
            obj_ids=[1],
            masks=[mask],
        )

        tracker.add_prompt(prompt)

        mock_processor.add_inputs_to_inference_session.assert_called_once()
        call_kwargs = mock_processor.add_inputs_to_inference_session.call_args[1]
        assert "input_masks" in call_kwargs
        assert call_kwargs["obj_ids"] == 1
        assert call_kwargs["frame_idx"] == 0

    def test_add_prompt_with_points(self):
        """Test add_prompt with point input."""
        tracker = SAM3Tracker()
        tracker._inference_session = MagicMock()
        tracker._use_text = False
        mock_processor = MagicMock()
        tracker._visual_processor = mock_processor

        prompt = Prompt(
            prompt_type=PromptType.POSE,
            frame_idx=5,
            obj_ids=[2],
            points=[[(50.0, 60.0), (70.0, 80.0)]],
        )

        tracker.add_prompt(prompt)

        mock_processor.add_inputs_to_inference_session.assert_called_once()
        call_kwargs = mock_processor.add_inputs_to_inference_session.call_args[1]
        assert "input_points" in call_kwargs
        assert "input_labels" in call_kwargs
        assert call_kwargs["input_points"] == [[[[50.0, 60.0], [70.0, 80.0]]]]
        assert call_kwargs["input_labels"] == [[[1, 1]]]

    def test_add_prompt_with_box(self):
        """Test add_prompt with box input."""
        tracker = SAM3Tracker()
        tracker._inference_session = MagicMock()
        tracker._use_text = False
        mock_processor = MagicMock()
        tracker._visual_processor = mock_processor

        prompt = Prompt(
            prompt_type=PromptType.ROI,
            frame_idx=3,
            obj_ids=[5],
            boxes=[(10.0, 20.0, 100.0, 150.0)],
        )

        tracker.add_prompt(prompt)

        mock_processor.add_inputs_to_inference_session.assert_called_once()
        call_kwargs = mock_processor.add_inputs_to_inference_session.call_args[1]
        assert "input_boxes" in call_kwargs
        assert call_kwargs["input_boxes"] == [[[10.0, 20.0, 100.0, 150.0]]]

    def test_add_prompt_multiple_objects(self):
        """Test add_prompt with multiple objects."""
        tracker = SAM3Tracker()
        tracker._inference_session = MagicMock()
        tracker._use_text = False
        mock_processor = MagicMock()
        tracker._visual_processor = mock_processor

        prompt = Prompt(
            prompt_type=PromptType.ROI,
            frame_idx=0,
            obj_ids=[1, 2, 3],
            masks=[
                np.ones((100, 100), dtype=np.uint8),
                np.ones((100, 100), dtype=np.uint8),
                np.ones((100, 100), dtype=np.uint8),
            ],
        )

        tracker.add_prompt(prompt)

        # Should be called once per object
        assert mock_processor.add_inputs_to_inference_session.call_count == 3


# Helper to check if we should skip GPU tests
def should_skip_gpu_tests():
    """Check if GPU tests should be skipped."""
    import os

    # Skip on CI
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return True
    # Skip if no CUDA
    if not torch.cuda.is_available():
        return True
    return False


@pytest.mark.skipif(should_skip_gpu_tests(), reason="Requires GPU and not running on CI")
class TestSAM3TrackerIntegration:
    """Integration tests for SAM3Tracker (require GPU and model)."""

    def test_visual_tracking_workflow(self):
        """Test complete visual tracking workflow."""
        # Create dummy video frames
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)
        ]

        tracker = SAM3Tracker()
        tracker.init_session(frames, use_text=False)

        # Add a box prompt
        prompt = Prompt(
            prompt_type=PromptType.ROI,
            frame_idx=0,
            obj_ids=[1],
            boxes=[(100, 100, 200, 200)],
        )
        tracker.add_prompt(prompt)

        # Propagate and collect results
        results = list(tracker.propagate(max_frames=5))
        assert len(results) > 0

        for result in results:
            assert isinstance(result, TrackingResult)
            assert result.masks.shape[0] == result.num_objects

    def test_text_tracking_workflow(self):
        """Test complete text tracking workflow."""
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)
        ]

        tracker = SAM3Tracker()
        tracker.init_session(frames, use_text=True)
        tracker.add_text_prompt("person")

        results = list(tracker.propagate(max_frames=5))
        assert len(results) > 0
