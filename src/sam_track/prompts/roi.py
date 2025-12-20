"""ROI YAML prompt handler for labelroi format."""

from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from sam_track.prompts.base import Prompt, PromptHandler, PromptType


def polygon_to_mask(
    coordinates: list[list[float]],
    width: int,
    height: int,
) -> np.ndarray:
    """Convert polygon coordinates to a binary mask.

    Args:
        coordinates: List of [x, y] polygon vertices.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Binary mask as numpy array of shape (height, width) with dtype uint8.
        Pixels inside the polygon are 1, outside are 0.
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    polygon = [(x, y) for x, y in coordinates]
    draw.polygon(polygon, fill=1)
    return np.array(mask, dtype=np.uint8)


def polygon_to_bbox(
    coordinates: list[list[float]],
) -> tuple[float, float, float, float]:
    """Convert polygon coordinates to bounding box.

    Args:
        coordinates: List of [x, y] polygon vertices.

    Returns:
        Bounding box as (x_min, y_min, x_max, y_max).
    """
    coords = np.array(coordinates)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    return float(x_min), float(y_min), float(x_max), float(y_max)


class ROIPromptHandler(PromptHandler):
    """Handler for ROI prompts from labelroi YAML files.

    This handler loads polygon ROI annotations from YAML files produced by
    labelroi and converts them to binary masks for SAM3 tracking.

    The YAML format is:
    ```yaml
    source: video.mp4
    resolution: [1920, 1080]
    fps: 30.0
    total_frames: 1000
    frame: 0
    roi_count: 2
    rois:
      - id: 1
        name: "animal_1"
        type: polygon
        color: "#1f77b4"
        coordinates:
          - [100.5, 200.3]
          - [150.2, 210.5]
          - ...
        properties:
          vertex_count: 3
          perimeter: 120.5
          area: 1500.0
    ```

    Example:
        >>> handler = ROIPromptHandler("path/to/rois.yaml")
        >>> prompt = handler.load()
        >>> print(f"Loaded {prompt.num_objects} objects")
        >>> for obj_id in prompt.obj_ids:
        ...     print(f"  {obj_id}: {prompt.get_name(obj_id)}")
    """

    def __init__(
        self,
        path: str | Path,
        use_masks: bool = True,
        include_boxes: bool = False,
    ):
        """Initialize ROI prompt handler.

        Args:
            path: Path to YAML file from labelroi.
            use_masks: If True (default), convert polygons to binary masks.
                This provides the best accuracy for SAM3 initialization.
            include_boxes: If True, also compute bounding boxes from polygons.
                Useful for output/debugging but not required for SAM3.
        """
        self.path = Path(path)
        self.use_masks = use_masks
        self.include_boxes = include_boxes

        self._data: dict | None = None

    @property
    def prompt_type(self) -> PromptType:
        """The type of prompt this handler produces."""
        return PromptType.ROI

    def _load_yaml(self) -> dict:
        """Load and cache YAML data."""
        if self._data is None:
            with open(self.path) as f:
                self._data = yaml.safe_load(f)
        return self._data

    @property
    def resolution(self) -> tuple[int, int]:
        """Video resolution as (width, height)."""
        data = self._load_yaml()
        return tuple(data["resolution"])

    @property
    def frame_idx(self) -> int:
        """Frame index where ROIs are annotated."""
        data = self._load_yaml()
        return data.get("frame", 0)

    @property
    def source_video(self) -> str | None:
        """Source video filename from YAML."""
        data = self._load_yaml()
        return data.get("source")

    @property
    def roi_count(self) -> int:
        """Number of ROIs in the file."""
        data = self._load_yaml()
        return data.get("roi_count", len(data.get("rois", [])))

    def load(self) -> Prompt:
        """Load ROI prompts from YAML file.

        Returns:
            A Prompt object with masks and object ID to name mapping.

        Raises:
            FileNotFoundError: If the YAML file doesn't exist.
            ValueError: If the YAML file has no ROIs.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"ROI file not found: {self.path}")

        data = self._load_yaml()
        rois = data.get("rois", [])

        if not rois:
            raise ValueError(f"No ROIs found in {self.path}")

        width, height = self.resolution
        frame_idx = self.frame_idx

        obj_ids = []
        obj_names = {}
        masks = [] if self.use_masks else None
        boxes = [] if self.include_boxes else None

        for roi in rois:
            obj_id = roi["id"]
            obj_ids.append(obj_id)

            # Store name mapping
            name = roi.get("name", f"roi_{obj_id}")
            obj_names[obj_id] = name

            coordinates = roi["coordinates"]

            # Convert polygon to mask
            if self.use_masks:
                mask = polygon_to_mask(coordinates, width, height)
                masks.append(mask)

            # Compute bounding box
            if self.include_boxes:
                bbox = polygon_to_bbox(coordinates)
                boxes.append(bbox)

        return Prompt(
            prompt_type=PromptType.ROI,
            frame_idx=frame_idx,
            obj_ids=obj_ids,
            obj_names=obj_names,
            masks=masks,
            boxes=boxes,
            source_path=self.path,
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"ROIPromptHandler(path={self.path!r}, roi_count={self.roi_count})"
