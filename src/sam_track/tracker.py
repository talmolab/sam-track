"""SAM3 video tracker wrapper.

This module provides a high-level interface for video object tracking using
SAM3 (Segment Anything Model 3) from the transformers library.

SAM3 supports two modes:
- **Text prompts (PCS)**: Promptable Concept Segmentation - detect and track all
  instances matching a text description using Sam3VideoModel.
- **Visual prompts (PVS)**: Promptable Visual Segmentation - track specific objects
  using points, boxes, or masks using Sam3TrackerVideoModel.
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from .prompts.base import Prompt, PromptType


# Default model ID for SAM3
DEFAULT_MODEL_ID = "facebook/sam3"


@dataclass
class TrackingResult:
    """Result for a single frame of video tracking.

    Attributes:
        frame_idx: Frame index in the video.
        object_ids: Array of tracked object IDs, shape (N,).
        masks: Binary segmentation masks, shape (N, H, W).
        boxes: Bounding boxes in XYXY format, shape (N, 4).
        scores: Confidence scores per object, shape (N,).
    """

    frame_idx: int
    object_ids: np.ndarray
    masks: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray

    @property
    def num_objects(self) -> int:
        """Number of tracked objects in this frame."""
        return len(self.object_ids)


class SAM3Tracker:
    """High-level wrapper for SAM3 video tracking.

    This class provides a simplified interface for tracking objects in videos
    using SAM3. It handles model loading, session management, and result
    post-processing.

    SAM3 supports two tracking modes:
    - Text prompts: Use Sam3VideoModel to detect and track all instances
      matching a text description (e.g., "mouse", "person").
    - Visual prompts: Use Sam3TrackerVideoModel to track specific objects
      specified by points, boxes, or masks.

    Example:
        >>> tracker = SAM3Tracker()
        >>> tracker.init_session(video_frames, use_text=True)
        >>> tracker.add_text_prompt("mouse")
        >>> for result in tracker.propagate():
        ...     print(f"Frame {result.frame_idx}: {result.num_objects} objects")

    Attributes:
        model_id: HuggingFace model ID.
        device: Device for inference (cuda, mps, cpu).
        dtype: Model dtype (bfloat16 recommended).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize SAM3 tracker.

        Args:
            model_id: HuggingFace model ID for SAM3.
            device: Device for inference. If None, auto-detects best available.
            dtype: Model dtype. bfloat16 is recommended for efficiency.
        """
        self.model_id = model_id
        self.dtype = dtype

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        # Lazy-loaded models and processors
        self._text_model = None
        self._text_processor = None
        self._visual_model = None
        self._visual_processor = None

        # Session state
        self._inference_session = None
        self._use_text: bool = False
        self._video_height: int = 0
        self._video_width: int = 0
        self._prompt_frame_idx: int | None = None  # Track first prompt frame for propagation

    def _load_text_model(self):
        """Lazy-load SAM3 video model for text prompts (PCS)."""
        if self._text_model is None:
            from transformers import Sam3VideoModel, Sam3VideoProcessor

            self._text_model = Sam3VideoModel.from_pretrained(self.model_id).to(
                self.device, dtype=self.dtype
            )
            self._text_processor = Sam3VideoProcessor.from_pretrained(self.model_id)

    def _load_visual_model(self):
        """Lazy-load SAM3 tracker video model for visual prompts (PVS)."""
        if self._visual_model is None:
            from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

            self._visual_model = Sam3TrackerVideoModel.from_pretrained(self.model_id).to(
                self.device, dtype=self.dtype
            )
            self._visual_processor = Sam3TrackerVideoProcessor.from_pretrained(
                self.model_id
            )

    @property
    def is_session_active(self) -> bool:
        """Check if a tracking session is currently active."""
        return self._inference_session is not None

    def init_session(
        self,
        video_frames: list[np.ndarray],
        use_text: bool = False,
    ) -> None:
        """Initialize a new tracking session with video frames.

        This loads all video frames into memory for SAM3 processing. For long
        videos, consider using streaming mode instead.

        Args:
            video_frames: List of video frames as numpy arrays (H, W, C) in RGB.
            use_text: If True, initialize for text prompts (PCS mode).
                If False, initialize for visual prompts (PVS mode).
        """
        if len(video_frames) == 0:
            raise ValueError("Video must have at least one frame")

        # Store video dimensions and mode
        self._video_height = video_frames[0].shape[0]
        self._video_width = video_frames[0].shape[1]
        self._use_text = use_text
        self._prompt_frame_idx = None

        if use_text:
            # Load text model and initialize session
            self._load_text_model()
            self._inference_session = self._text_processor.init_video_session(
                video=video_frames,
                inference_device=self.device,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=self.dtype,
            )
        else:
            # Load visual model and initialize session
            self._load_visual_model()
            self._inference_session = self._visual_processor.init_video_session(
                video=video_frames,
                inference_device=self.device,
                dtype=self.dtype,
            )

    def add_text_prompt(self, text: str) -> None:
        """Add a text prompt to detect and track objects.

        This uses SAM3's Promptable Concept Segmentation (PCS) to detect all
        instances matching the text description and track them through the video.

        Args:
            text: Text description of objects to track (e.g., "mouse", "person").

        Raises:
            RuntimeError: If no session is active or session is not in text mode.
        """
        if not self.is_session_active:
            raise RuntimeError("No active session. Call init_session() first.")

        if not self._use_text:
            raise RuntimeError(
                "Session initialized for visual prompts. "
                "Use init_session(use_text=True) for text prompts."
            )

        self._inference_session = self._text_processor.add_text_prompt(
            inference_session=self._inference_session,
            text=text,
        )

    def add_prompt(self, prompt: Prompt) -> None:
        """Add a visual prompt to the tracking session.

        The prompt specifies which objects to track using visual cues such as
        points, bounding boxes, or segmentation masks.

        Args:
            prompt: A Prompt object containing the tracking specification.

        Raises:
            RuntimeError: If no session is active or session is in text mode.
            ValueError: If prompt type is TEXT (use add_text_prompt instead).
        """
        if not self.is_session_active:
            raise RuntimeError("No active session. Call init_session() first.")

        if prompt.prompt_type == PromptType.TEXT:
            raise ValueError(
                "Use add_text_prompt() for text prompts, not add_prompt()."
            )

        if self._use_text:
            raise RuntimeError(
                "Session initialized for text prompts. "
                "Use init_session(use_text=False) for visual prompts."
            )

        # Add prompts for each object
        for i, obj_id in enumerate(prompt.obj_ids):
            # Prepare inputs for this object
            kwargs: dict = {
                "inference_session": self._inference_session,
                "frame_idx": prompt.frame_idx,
                "obj_ids": obj_id,
            }

            # Add mask if available (preferred - most accurate)
            if prompt.masks is not None and i < len(prompt.masks):
                kwargs["input_masks"] = prompt.masks[i]

            # Add points if available
            elif prompt.points is not None and i < len(prompt.points):
                points = prompt.points[i]
                # Format: [[[[x, y], [x, y], ...]]]
                kwargs["input_points"] = [[[[p[0], p[1]] for p in points]]]
                # All points are positive (foreground)
                kwargs["input_labels"] = [[[1] * len(points)]]

            # Add box if available
            elif prompt.boxes is not None and i < len(prompt.boxes):
                box = prompt.boxes[i]
                # Format: [[[x_min, y_min, x_max, y_max]]]
                kwargs["input_boxes"] = [[[box[0], box[1], box[2], box[3]]]]

            else:
                raise ValueError(
                    f"No valid prompt data for object {obj_id}. "
                    "Prompt must have masks, points, or boxes."
                )

            # Add to session
            self._visual_processor.add_inputs_to_inference_session(**kwargs)

        # Track the prompt frame for propagation
        if self._prompt_frame_idx is None:
            self._prompt_frame_idx = prompt.frame_idx

    def propagate(
        self,
        max_frames: int | None = None,
        reverse: bool = False,
    ) -> Iterator[TrackingResult]:
        """Propagate tracking through the video.

        Yields tracking results for each frame, starting from the prompt frame
        and propagating forward (or backward if reverse=True).

        Args:
            max_frames: Maximum number of frames to process. If None, processes
                all frames.
            reverse: If True, propagate backward from the prompt frame.

        Yields:
            TrackingResult for each processed frame.

        Raises:
            RuntimeError: If no session is active.
        """
        if not self.is_session_active:
            raise RuntimeError("No active session. Call init_session() first.")

        # Get original video size for mask post-processing
        original_size = [[self._video_height, self._video_width]]

        if self._use_text:
            # Text prompt mode - use Sam3VideoModel
            for output in self._text_model.propagate_in_video_iterator(
                inference_session=self._inference_session,
                max_frame_num_to_track=max_frames,
            ):
                # Post-process outputs using text processor
                processed = self._text_processor.postprocess_outputs(
                    self._inference_session, output
                )

                yield TrackingResult(
                    frame_idx=output.frame_idx,
                    object_ids=processed["object_ids"].numpy(),
                    masks=processed["masks"].numpy().astype(bool),
                    boxes=processed["boxes"].numpy(),
                    scores=processed["scores"].numpy(),
                )
        else:
            # Visual prompt mode - use Sam3TrackerVideoModel
            # Run inference on the prompt frame first to initialize tracking
            if self._prompt_frame_idx is not None:
                self._visual_model(
                    inference_session=self._inference_session,
                    frame_idx=self._prompt_frame_idx,
                )

            for output in self._visual_model.propagate_in_video_iterator(
                inference_session=self._inference_session,
                max_frame_num_to_track=max_frames,
                reverse=reverse,
            ):
                # Post-process masks to original resolution
                masks = self._visual_processor.post_process_masks(
                    [output.pred_masks],
                    original_sizes=original_size,
                    binarize=True,
                )[0]

                # Convert to numpy
                masks_np = masks.cpu().numpy().astype(bool)

                # Extract object IDs
                obj_ids = (
                    np.array(output.object_ids) if output.object_ids else np.array([])
                )

                # Compute bounding boxes from masks
                boxes = self._compute_boxes_from_masks(masks_np)

                # Compute confidence scores from object score logits if available
                if (
                    output.object_score_logits is not None
                    and isinstance(output.object_score_logits, torch.Tensor)
                ):
                    scores = (
                        torch.sigmoid(output.object_score_logits)
                        .float()
                        .cpu()
                        .numpy()
                    )
                else:
                    # Default to 1.0 if no scores available
                    scores = np.ones(len(obj_ids))

                yield TrackingResult(
                    frame_idx=output.frame_idx,
                    object_ids=obj_ids,
                    masks=masks_np,
                    boxes=boxes,
                    scores=scores,
                )

    def _compute_boxes_from_masks(self, masks: np.ndarray) -> np.ndarray:
        """Compute bounding boxes from binary masks.

        Args:
            masks: Binary masks, shape (N, H, W).

        Returns:
            Array of bounding boxes in XYXY format, shape (N, 4).
        """
        boxes = []
        for mask in masks:
            if mask.sum() == 0:
                # Empty mask - return zero box
                boxes.append([0.0, 0.0, 0.0, 0.0])
            else:
                # Find bounding box from mask
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                boxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])
        return np.array(boxes) if boxes else np.zeros((0, 4))

    def reset(self) -> None:
        """Reset the tracking session for new prompts.

        This clears all tracking state while keeping the video loaded.
        Use this to start tracking different objects without reloading
        the video.
        """
        if self._inference_session is not None:
            self._inference_session.reset_inference_session()

    def close(self) -> None:
        """Close the tracking session and free resources."""
        self._inference_session = None
        # Note: We don't unload the model as it may be reused

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()
