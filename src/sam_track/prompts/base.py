"""Base classes for prompt handlers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np


class PromptType(Enum):
    """Types of prompts supported by SAM3."""

    TEXT = "text"
    ROI = "roi"
    POSE = "pose"


@dataclass
class Prompt:
    """A prompt for SAM3 tracking.

    This represents the data needed to initialize tracking for one or more
    objects. For visual prompts (ROI, pose), this includes masks derived
    from the input annotations. For text prompts, this includes the text
    description.

    Attributes:
        prompt_type: The type of prompt (text, roi, pose).
        frame_idx: Frame index where the prompt should be applied.
        obj_ids: List of object IDs for tracking.
        obj_names: Mapping from object ID to human-readable name.
        masks: List of binary masks (H, W) for each object. None for text prompts.
        boxes: List of bounding boxes [x_min, y_min, x_max, y_max] for each object.
        points: List of point sets for each object. Each point set is a list of
            (x, y) tuples representing keypoints. Used for pose prompts.
        text: Text description for text prompts. None for visual prompts.
        source_path: Path to the source file (YAML, SLP, etc.) if applicable.
    """

    prompt_type: PromptType
    frame_idx: int = 0
    obj_ids: list[int] = field(default_factory=list)
    obj_names: dict[int, str] = field(default_factory=dict)
    masks: list[np.ndarray] | None = None
    boxes: list[tuple[float, float, float, float]] | None = None
    points: list[list[tuple[float, float]]] | None = None
    text: str | None = None
    source_path: Path | None = None

    def __post_init__(self):
        """Validate prompt data."""
        if self.prompt_type == PromptType.TEXT:
            if self.text is None:
                raise ValueError("Text prompt requires 'text' field")
        else:
            if not self.obj_ids:
                raise ValueError(f"{self.prompt_type.value} prompt requires 'obj_ids'")
            if self.masks is not None and len(self.masks) != len(self.obj_ids):
                raise ValueError(
                    f"Number of masks ({len(self.masks)}) must match "
                    f"number of obj_ids ({len(self.obj_ids)})"
                )
            if self.points is not None and len(self.points) != len(self.obj_ids):
                raise ValueError(
                    f"Number of point sets ({len(self.points)}) must match "
                    f"number of obj_ids ({len(self.obj_ids)})"
                )

    @property
    def num_objects(self) -> int:
        """Number of objects in this prompt."""
        return len(self.obj_ids)

    def get_name(self, obj_id: int) -> str:
        """Get human-readable name for an object ID.

        Args:
            obj_id: The object ID.

        Returns:
            The name if available, otherwise a default like "object_1".
        """
        return self.obj_names.get(obj_id, f"object_{obj_id}")


class PromptHandler(ABC):
    """Abstract base class for prompt handlers.

    A prompt handler is responsible for loading prompt data from a source
    (file, string, etc.) and converting it to a Prompt object that can be
    used with SAM3.
    """

    @property
    @abstractmethod
    def prompt_type(self) -> PromptType:
        """The type of prompt this handler produces."""
        pass

    @abstractmethod
    def load(self) -> Prompt:
        """Load and return the prompt.

        Returns:
            A Prompt object ready for use with SAM3.
        """
        pass
