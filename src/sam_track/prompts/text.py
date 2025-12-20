"""Text prompt handler for SAM3."""

from sam_track.prompts.base import Prompt, PromptHandler, PromptType


class TextPromptHandler(PromptHandler):
    """Handler for text prompts.

    Text prompts use SAM3's open-vocabulary detection to find all instances
    matching a text description in the video. This is useful for tracking
    objects by category (e.g., "mouse", "fly", "person").

    Note: Text prompts work differently from visual prompts:
    - SAM3 will detect all matching instances automatically
    - Object IDs are assigned by SAM3 during detection
    - No initial masks are provided; SAM3 generates them

    Example:
        >>> handler = TextPromptHandler("mouse")
        >>> prompt = handler.load()
        >>> print(f"Text prompt: {prompt.text}")
        Text prompt: mouse
    """

    def __init__(
        self,
        text: str,
        frame_idx: int = 0,
    ):
        """Initialize text prompt handler.

        Args:
            text: Text description of object to track (e.g., "mouse", "fly").
                This should be a short noun phrase describing the object class.
            frame_idx: Frame index to apply the text prompt on (default 0).
                SAM3 will detect objects matching the text on this frame and
                track them through the video.
        """
        self.text = text
        self._frame_idx = frame_idx

    @property
    def prompt_type(self) -> PromptType:
        """The type of prompt this handler produces."""
        return PromptType.TEXT

    def load(self) -> Prompt:
        """Load text prompt.

        Returns:
            A Prompt object with the text description.

        Note:
            Text prompts don't have obj_ids or masks initially.
            SAM3 will assign object IDs during detection.
        """
        return Prompt(
            prompt_type=PromptType.TEXT,
            frame_idx=self._frame_idx,
            obj_ids=[],  # SAM3 assigns IDs during detection
            obj_names={},  # Names come from SAM3 detection
            masks=None,
            boxes=None,
            text=self.text,
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"TextPromptHandler(text={self.text!r}, frame_idx={self._frame_idx})"
