"""SAM3 video tracker wrapper.

This module provides a high-level interface for video object tracking using
SAM3 (Segment Anything Model 3) from the transformers library.

SAM3 supports two modes:
- **Text prompts (PCS)**: Promptable Concept Segmentation - detect and track all
  instances matching a text description using Sam3VideoModel.
- **Visual prompts (PVS)**: Promptable Visual Segmentation - track specific objects
  using points, boxes, or masks using Sam3TrackerVideoModel.

Supports both batch mode (all frames loaded) and streaming mode (frame-by-frame).
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm

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
        self._streaming: bool = False
        self._video_height: int = 0
        self._video_width: int = 0
        self._num_frames: int | None = None
        # Track first prompt frame for propagation
        self._prompt_frame_idx: int | None = None
        self._frame_idx: int = 0  # Current frame index for streaming mode

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

            self._visual_model = Sam3TrackerVideoModel.from_pretrained(
                self.model_id
            ).to(self.device, dtype=self.dtype)
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
        video_storage_device: str = "cpu",
        max_vision_cache_size: int = 1,
    ) -> None:
        """Initialize a new tracking session with video frames.

        This loads all video frames into memory for SAM3 processing. For long
        videos, consider using streaming mode via init_streaming_session() instead.

        Args:
            video_frames: List of video frames as numpy arrays (H, W, C) in RGB.
            use_text: If True, initialize for text prompts (PCS mode).
                If False, initialize for visual prompts (PVS mode).
            video_storage_device: Device to store video frames on. Use "cpu" for
                memory optimization with long videos.
            max_vision_cache_size: Maximum number of vision features to cache.
                Lower values use less memory but may be slower.
        """
        if len(video_frames) == 0:
            raise ValueError("Video must have at least one frame")

        # Store video dimensions and mode
        self._video_height = video_frames[0].shape[0]
        self._video_width = video_frames[0].shape[1]
        self._num_frames = len(video_frames)
        self._use_text = use_text
        self._streaming = False
        self._prompt_frame_idx = None
        self._frame_idx = 0

        if use_text:
            # Load text model and initialize session
            self._load_text_model()
            self._inference_session = self._text_processor.init_video_session(
                video=video_frames,
                inference_device=self.device,
                processing_device="cpu",
                video_storage_device=video_storage_device,
                dtype=self.dtype,
            )
        else:
            # Load visual model and initialize session
            self._load_visual_model()
            self._inference_session = self._visual_processor.init_video_session(
                video=video_frames,
                inference_device=self.device,
                video_storage_device=video_storage_device,
                max_vision_features_cache_size=max_vision_cache_size,
                dtype=self.dtype,
            )

    def init_streaming_session(
        self,
        use_text: bool = False,
        num_frames: int | None = None,
    ) -> None:
        """Initialize a streaming session for frame-by-frame processing.

        Streaming mode processes frames one at a time, which is memory-efficient
        for long videos or real-time applications. Frames are provided via
        process_frame() instead of loading all at once.

        Note: Streaming mode may have lower tracking quality than batch mode
        because it cannot use future frames for object matching.

        Args:
            use_text: If True, initialize for text prompts (PCS mode).
                If False, initialize for visual prompts (PVS mode).
            num_frames: Optional total number of frames (for progress reporting).
        """
        self._use_text = use_text
        self._streaming = True
        self._num_frames = num_frames
        self._prompt_frame_idx = None
        self._frame_idx = 0
        self._video_height = 0
        self._video_width = 0

        if use_text:
            self._load_text_model()
            self._inference_session = self._text_processor.init_video_session(
                inference_device=self.device,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=self.dtype,
            )
        else:
            self._load_visual_model()
            self._inference_session = self._visual_processor.init_video_session(
                inference_device=self.device,
                dtype=self.dtype,
            )

    def process_frame(self, frame: np.ndarray) -> TrackingResult:
        """Process a single frame in streaming mode.

        This method processes one frame at a time, yielding tracking results
        for that frame. Use this for real-time processing or memory-constrained
        environments.

        Args:
            frame: Video frame as numpy array (H, W, C) in RGB.

        Returns:
            TrackingResult for the processed frame.

        Raises:
            RuntimeError: If no session is active or not in streaming mode.
        """
        if not self.is_session_active:
            raise RuntimeError(
                "No active session. Call init_streaming_session() first."
            )

        if not self._streaming:
            raise RuntimeError(
                "Not in streaming mode. Use init_streaming_session() for "
                "frame-by-frame processing, or use propagate() for batch mode."
            )

        # Update video dimensions from first frame
        if self._frame_idx == 0:
            self._video_height = frame.shape[0]
            self._video_width = frame.shape[1]

        frame_idx = self._frame_idx
        self._frame_idx += 1

        if self._use_text:
            return self._process_frame_text(frame, frame_idx)
        else:
            return self._process_frame_visual(frame, frame_idx)

    def _process_frame_text(self, frame: np.ndarray, frame_idx: int) -> TrackingResult:
        """Process a frame in streaming text mode."""
        # Process frame using the processor
        inputs = self._text_processor(
            images=frame, device=self.device, return_tensors="pt"
        )

        # Run model with processed frame
        output = self._text_model(
            inference_session=self._inference_session,
            frame=inputs.pixel_values[0],
            reverse=False,
        )

        # Post-process outputs
        processed = self._text_processor.postprocess_outputs(
            self._inference_session,
            output,
            original_sizes=inputs.original_sizes,
        )

        return TrackingResult(
            frame_idx=frame_idx,
            object_ids=processed["object_ids"].cpu().numpy(),
            masks=processed["masks"].cpu().numpy().astype(bool),
            boxes=processed["boxes"].cpu().numpy(),
            scores=processed["scores"].cpu().numpy(),
        )

    def _process_frame_visual(
        self, frame: np.ndarray, frame_idx: int
    ) -> TrackingResult:
        """Process a frame in streaming visual mode."""
        # Process frame using the processor
        inputs = self._visual_processor(
            images=frame, device=self.device, return_tensors="pt"
        )

        # Run model with processed frame
        output = self._visual_model(
            inference_session=self._inference_session,
            frame=inputs.pixel_values[0],
        )

        # Post-process masks to original resolution
        masks = self._visual_processor.post_process_masks(
            [output.pred_masks],
            original_sizes=inputs.original_sizes,
            binarize=True,
        )[0]

        # Convert to numpy
        masks_np = masks.cpu().numpy().astype(bool)

        # Extract object IDs
        obj_ids = np.array(output.object_ids) if output.object_ids else np.array([])

        # Compute bounding boxes from masks
        boxes = self._compute_boxes_from_masks(masks_np)

        # Compute confidence scores
        if output.object_score_logits is not None and isinstance(
            output.object_score_logits, torch.Tensor
        ):
            scores = torch.sigmoid(output.object_score_logits).float().cpu().numpy()
            # Squeeze extra dimensions (SAM3 may return shape (N, 1))
            scores = np.squeeze(scores)
            # Handle single object case
            if scores.ndim == 0:
                scores = np.array([float(scores)])
        else:
            scores = np.ones(len(obj_ids))

        return TrackingResult(
            frame_idx=frame_idx,
            object_ids=obj_ids,
            masks=masks_np,
            boxes=boxes,
            scores=scores,
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

    def add_prompt(
        self, prompt: Prompt, original_size: tuple[int, int] | None = None
    ) -> None:
        """Add a visual prompt to the tracking session.

        The prompt specifies which objects to track using visual cues such as
        points, bounding boxes, or segmentation masks.

        Args:
            prompt: A Prompt object containing the tracking specification.
            original_size: Original frame size as (height, width). Required for
                streaming mode, optional for batch mode.

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

        # Prepare base kwargs
        kwargs: dict = {
            "inference_session": self._inference_session,
            "frame_idx": prompt.frame_idx,
            "obj_ids": prompt.obj_ids,  # Pass all object IDs at once
        }

        # Add original_size for streaming mode
        if self._streaming and original_size is not None:
            kwargs["original_size"] = original_size

        # Add masks if available (preferred - most accurate)
        # Note: Masks must be batched in a single call for SAM3 to work correctly
        if prompt.masks is not None and len(prompt.masks) > 0:
            kwargs["input_masks"] = prompt.masks

        # Add points if available (when no masks)
        elif prompt.points is not None and len(prompt.points) > 0:
            # Format for batched points: [[obj1_points, obj2_points, ...]]
            # Each obj_points is [[x, y], [x, y], ...]
            all_points = []
            all_labels = []
            for points in prompt.points:
                all_points.append([[p[0], p[1]] for p in points])
                all_labels.append([1] * len(points))  # All positive
            kwargs["input_points"] = [all_points]
            kwargs["input_labels"] = [all_labels]

        # Add boxes if available (when no masks or points)
        elif prompt.boxes is not None and len(prompt.boxes) > 0:
            # Format for batched boxes: [[box1, box2, ...]]
            all_boxes = []
            for box in prompt.boxes:
                all_boxes.append([box[0], box[1], box[2], box[3]])
            kwargs["input_boxes"] = [all_boxes]

        else:
            raise ValueError(
                "No valid prompt data. Prompt must have masks, points, or boxes."
            )

        # Add all prompts in a single batched call
        self._visual_processor.add_inputs_to_inference_session(**kwargs)

        # Track the prompt frame for propagation
        if self._prompt_frame_idx is None:
            self._prompt_frame_idx = prompt.frame_idx

        # Track which objects have been prompted (for streaming mode)
        if not hasattr(self, "_tracked_obj_ids"):
            self._tracked_obj_ids: set[int] = set()
        self._tracked_obj_ids.update(prompt.obj_ids)

    def add_streaming_prompt(
        self,
        prompt: Prompt,
        original_size: tuple[int, int],
    ) -> list[int]:
        """Add prompts for NEW objects during streaming mode.

        This method allows adding new objects to track mid-stream. Based on
        SAM3 experimentation, this works for NEW objects but NOT for correcting
        existing tracks (re-prompting existing objects destabilizes tracking).

        Args:
            prompt: A Prompt object containing the new objects to track.
            original_size: Frame size as (height, width).

        Returns:
            List of object IDs that were actually added (excludes already-tracked).

        Raises:
            RuntimeError: If not in streaming mode or no active session.
            ValueError: If prompt type is TEXT.
        """
        if not self.is_session_active:
            raise RuntimeError(
                "No active session. Call init_streaming_session() first."
            )

        if not self._streaming:
            raise RuntimeError(
                "add_streaming_prompt() only works in streaming mode. "
                "Use add_prompt() before propagate() for batch mode."
            )

        if prompt.prompt_type == PromptType.TEXT:
            raise ValueError(
                "Cannot add text prompts mid-stream. "
                "Use visual prompts (POSE, ROI) for streaming mode."
            )

        # Initialize tracking set if needed
        if not hasattr(self, "_tracked_obj_ids"):
            self._tracked_obj_ids = set()

        # Current frame index (use last processed frame)
        current_frame = max(0, self._frame_idx - 1)

        added_ids = []
        for i, obj_id in enumerate(prompt.obj_ids):
            # Skip objects already being tracked
            if obj_id in self._tracked_obj_ids:
                continue

            # Prepare inputs for this object
            kwargs: dict = {
                "inference_session": self._inference_session,
                "frame_idx": current_frame,
                "obj_ids": obj_id,
                "original_size": original_size,
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
                # Skip objects without valid prompt data
                continue

            # Add to session
            self._visual_processor.add_inputs_to_inference_session(**kwargs)

            # Track this object
            self._tracked_obj_ids.add(obj_id)
            added_ids.append(obj_id)

        return added_ids

    def propagate(
        self,
        max_frames: int | None = None,
        reverse: bool = False,
        show_progress: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Iterator[TrackingResult]:
        """Propagate tracking through the video.

        Yields tracking results for each frame, starting from the prompt frame
        and propagating forward (or backward if reverse=True).

        Args:
            max_frames: Maximum number of frames to process. If None, processes
                all frames.
            reverse: If True, propagate backward from the prompt frame.
            show_progress: If True, display a progress bar using tqdm.
            progress_callback: Optional callback function called with
                (current_frame, total_frames) for custom progress reporting.

        Yields:
            TrackingResult for each processed frame.

        Raises:
            RuntimeError: If no session is active or in streaming mode.
        """
        if not self.is_session_active:
            raise RuntimeError("No active session. Call init_session() first.")

        if self._streaming:
            raise RuntimeError(
                "Cannot use propagate() in streaming mode. "
                "Use process_frame() instead for frame-by-frame processing."
            )

        # Get original video size for mask post-processing
        original_size = [[self._video_height, self._video_width]]

        # Determine total frames for progress reporting
        total_frames = max_frames if max_frames else self._num_frames

        if self._use_text:
            yield from self._propagate_text(
                max_frames,
                original_size,
                total_frames,
                show_progress,
                progress_callback,
            )
        else:
            yield from self._propagate_visual(
                max_frames,
                reverse,
                original_size,
                total_frames,
                show_progress,
                progress_callback,
            )

    def _propagate_text(
        self,
        max_frames: int | None,
        original_size: list[list[int]],
        total_frames: int | None,
        show_progress: bool,
        progress_callback: Callable[[int, int], None] | None,
    ) -> Iterator[TrackingResult]:
        """Propagate text prompts through video."""
        iterator = self._text_model.propagate_in_video_iterator(
            inference_session=self._inference_session,
            max_frame_num_to_track=max_frames,
        )

        # Wrap with progress bar if requested
        if show_progress:
            iterator = tqdm(
                iterator,
                total=total_frames,
                desc="Tracking",
                unit="frame",
            )

        frame_count = 0
        for output in iterator:
            # Post-process outputs using text processor
            processed = self._text_processor.postprocess_outputs(
                self._inference_session, output
            )

            frame_count += 1
            if progress_callback and total_frames:
                progress_callback(frame_count, total_frames)

            yield TrackingResult(
                frame_idx=output.frame_idx,
                object_ids=processed["object_ids"].cpu().numpy(),
                masks=processed["masks"].cpu().numpy().astype(bool),
                boxes=processed["boxes"].cpu().numpy(),
                scores=processed["scores"].cpu().numpy(),
            )

    def _propagate_visual(
        self,
        max_frames: int | None,
        reverse: bool,
        original_size: list[list[int]],
        total_frames: int | None,
        show_progress: bool,
        progress_callback: Callable[[int, int], None] | None,
    ) -> Iterator[TrackingResult]:
        """Propagate visual prompts through video."""
        # Note: propagate_in_video_iterator handles the initial conditioning frame
        # automatically - no need to call model() explicitly before propagation.
        iterator = self._visual_model.propagate_in_video_iterator(
            inference_session=self._inference_session,
            start_frame_idx=self._prompt_frame_idx,
            max_frame_num_to_track=max_frames,
            reverse=reverse,
        )

        # Wrap with progress bar if requested
        if show_progress:
            iterator = tqdm(
                iterator,
                total=total_frames,
                desc="Tracking",
                unit="frame",
            )

        frame_count = 0
        for output in iterator:
            # Post-process masks to original resolution
            masks = self._visual_processor.post_process_masks(
                [output.pred_masks],
                original_sizes=original_size,
                binarize=True,
            )[0]

            # Convert to numpy
            masks_np = masks.cpu().numpy().astype(bool)

            # Extract object IDs
            obj_ids = np.array(output.object_ids) if output.object_ids else np.array([])

            # Compute bounding boxes from masks
            boxes = self._compute_boxes_from_masks(masks_np)

            # Compute confidence scores from object score logits if available
            if output.object_score_logits is not None and isinstance(
                output.object_score_logits, torch.Tensor
            ):
                scores = torch.sigmoid(output.object_score_logits).float().cpu().numpy()
                # Squeeze extra dimensions (SAM3 may return shape (N, 1))
                scores = np.squeeze(scores)
                # Handle single object case
                if scores.ndim == 0:
                    scores = np.array([float(scores)])
            else:
                # Default to 1.0 if no scores available
                scores = np.ones(len(obj_ids))

            frame_count += 1
            if progress_callback and total_frames:
                progress_callback(frame_count, total_frames)

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
            masks: Binary masks, shape (N, H, W) or (N, 1, H, W).

        Returns:
            Array of bounding boxes in XYXY format, shape (N, 4).
        """
        boxes = []
        for mask in masks:
            # Squeeze any extra dimensions (SAM3 outputs shape (1, H, W) per object)
            mask = np.squeeze(mask)

            if mask.sum() == 0:
                # Empty mask - return zero box
                boxes.append([0.0, 0.0, 0.0, 0.0])
            else:
                # Find bounding box from mask (H, W) where rows=Y, cols=X
                rows = np.any(mask, axis=1)  # Shape (H,) - which rows have content
                cols = np.any(mask, axis=0)  # Shape (W,) - which cols have content
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
