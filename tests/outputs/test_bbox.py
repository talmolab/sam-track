"""Tests for BBoxWriter."""

import json

import numpy as np
import pytest

from sam_track.outputs.bbox import BBoxWriter, Detection, Track, TrackingEncoder
from sam_track.tracker import TrackingResult


@pytest.fixture
def sample_result():
    """Create a sample TrackingResult for testing."""
    return TrackingResult(
        frame_idx=0,
        object_ids=np.array([1, 2]),
        masks=np.zeros((2, 100, 100), dtype=bool),
        boxes=np.array([[10.0, 20.0, 50.0, 60.0], [100.0, 150.0, 200.0, 250.0]]),
        scores=np.array([0.95, 0.88]),
    )


@pytest.fixture
def sample_result_frame_5():
    """Create a sample TrackingResult for frame 5."""
    return TrackingResult(
        frame_idx=5,
        object_ids=np.array([1]),
        masks=np.zeros((1, 100, 100), dtype=bool),
        boxes=np.array([[15.0, 25.0, 55.0, 65.0]]),
        scores=np.array([0.92]),
    )


@pytest.fixture
def empty_result():
    """Create an empty TrackingResult."""
    return TrackingResult(
        frame_idx=10,
        object_ids=np.array([], dtype=np.int64),
        masks=np.zeros((0, 100, 100), dtype=bool),
        boxes=np.zeros((0, 4)),
        scores=np.array([]),
    )


class TestDetection:
    """Tests for Detection dataclass."""

    def test_detection_properties(self):
        """Test computed properties."""
        det = Detection(
            frame_idx=0,
            x_min=10.0,
            y_min=20.0,
            x_max=50.0,
            y_max=60.0,
            confidence=0.95,
        )

        assert det.width == 40.0
        assert det.height == 40.0
        assert det.area == 1600.0

    def test_detection_to_dict(self):
        """Test dictionary conversion."""
        det = Detection(
            frame_idx=0,
            x_min=10.0,
            y_min=20.0,
            x_max=50.0,
            y_max=60.0,
            confidence=0.95,
        )

        d = det.to_dict()
        assert d["frame_idx"] == 0
        assert d["x_min"] == 10.0
        assert d["width"] == 40.0
        assert d["area"] == 1600.0


class TestTrack:
    """Tests for Track dataclass."""

    def test_empty_track(self):
        """Test empty track properties."""
        track = Track(track_id=1)

        assert track.num_detections == 0
        assert track.first_frame is None
        assert track.last_frame is None
        assert track.avg_confidence == 0.0

    def test_track_with_detections(self):
        """Test track with detections."""
        track = Track(track_id=1, name="mouse")
        track.add_detection(Detection(0, 10, 20, 50, 60, 0.9))
        track.add_detection(Detection(5, 15, 25, 55, 65, 0.8))

        assert track.num_detections == 2
        assert track.first_frame == 0
        assert track.last_frame == 5
        assert abs(track.avg_confidence - 0.85) < 1e-9

    def test_track_to_dict(self):
        """Test dictionary conversion."""
        track = Track(track_id=1, name="mouse")
        track.add_detection(Detection(0, 10, 20, 50, 60, 0.9))

        d = track.to_dict()
        assert d["track_id"] == 1
        assert d["name"] == "mouse"
        assert d["num_detections"] == 1
        assert len(d["detections"]) == 1


class TestTrackingEncoder:
    """Tests for custom JSON encoder."""

    def test_encode_numpy_array(self):
        """Test numpy array encoding."""
        arr = np.array([1, 2, 3])
        result = json.dumps(arr, cls=TrackingEncoder)
        assert result == "[1, 2, 3]"

    def test_encode_numpy_scalars(self):
        """Test numpy scalar encoding."""
        assert json.dumps(np.int64(42), cls=TrackingEncoder) == "42"
        assert json.dumps(np.float64(3.14), cls=TrackingEncoder) == "3.14"
        assert json.dumps(np.bool_(True), cls=TrackingEncoder) == "true"

    def test_encode_datetime(self):
        """Test datetime encoding."""
        from datetime import datetime

        dt = datetime(2025, 12, 20, 10, 30, 0)
        result = json.dumps(dt, cls=TrackingEncoder)
        assert "2025-12-20" in result


class TestBBoxWriter:
    """Tests for BBoxWriter."""

    def test_creates_file(self, tmp_path, sample_result):
        """Test that save creates the output file."""
        output_path = tmp_path / "output.bbox.json"

        writer = BBoxWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=1920,
            video_height=1080,
        )
        writer.add_result(sample_result)
        writer.save()

        assert output_path.exists()

    def test_creates_parent_dirs(self, tmp_path, sample_result):
        """Test that save creates parent directories."""
        output_path = tmp_path / "subdir" / "nested" / "output.bbox.json"

        writer = BBoxWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=1920,
            video_height=1080,
        )
        writer.add_result(sample_result)
        writer.save()

        assert output_path.exists()

    def test_metadata(self, tmp_path, sample_result):
        """Test metadata is written correctly."""
        output_path = tmp_path / "output.bbox.json"

        writer = BBoxWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=1920,
            video_height=1080,
            fps=30.0,
            total_frames=1000,
            prompt_type="text",
            prompt_value="mouse",
        )
        writer.add_result(sample_result)
        writer.save()

        with open(output_path) as f:
            data = json.load(f)

        assert data["metadata"]["version"] == "1.0"
        assert data["metadata"]["format"] == "sam-track-bbox"
        assert data["metadata"]["video_source"] == "video.mp4"
        assert data["metadata"]["width"] == 1920
        assert data["metadata"]["height"] == 1080
        assert data["metadata"]["fps"] == 30.0
        assert data["metadata"]["prompt_type"] == "text"
        assert data["metadata"]["prompt_value"] == "mouse"

    def test_single_track(self, tmp_path, sample_result, sample_result_frame_5):
        """Test writing a single track across frames."""
        output_path = tmp_path / "output.bbox.json"

        writer = BBoxWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=1920,
            video_height=1080,
        )
        writer.add_result(sample_result)
        writer.add_result(sample_result_frame_5)
        writer.save()

        with open(output_path) as f:
            data = json.load(f)

        # Track 1 should have 2 detections (frame 0 and 5)
        track_1 = next(t for t in data["tracks"] if t["track_id"] == 1)
        assert track_1["num_detections"] == 2
        assert track_1["first_frame"] == 0
        assert track_1["last_frame"] == 5

        # Track 2 should have 1 detection (only frame 0)
        track_2 = next(t for t in data["tracks"] if t["track_id"] == 2)
        assert track_2["num_detections"] == 1

    def test_multi_track(self, tmp_path, sample_result):
        """Test writing multiple tracks."""
        output_path = tmp_path / "output.bbox.json"

        writer = BBoxWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=1920,
            video_height=1080,
        )
        writer.add_result(sample_result)
        writer.save()

        with open(output_path) as f:
            data = json.load(f)

        assert len(data["tracks"]) == 2
        track_ids = {t["track_id"] for t in data["tracks"]}
        assert track_ids == {1, 2}

    def test_statistics(self, tmp_path, sample_result, sample_result_frame_5):
        """Test statistics computation."""
        output_path = tmp_path / "output.bbox.json"

        writer = BBoxWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=1920,
            video_height=1080,
        )
        writer.add_result(sample_result)
        writer.add_result(sample_result_frame_5)
        writer.save()

        with open(output_path) as f:
            data = json.load(f)

        stats = data["statistics"]
        assert stats["total_tracks"] == 2
        assert stats["total_detections"] == 3
        assert stats["frames_with_detections"] == 2
        assert 0 < stats["avg_confidence"] < 1

    def test_empty_result(self, tmp_path, empty_result):
        """Test handling of empty results."""
        output_path = tmp_path / "output.bbox.json"

        writer = BBoxWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=1920,
            video_height=1080,
        )
        writer.add_result(empty_result)
        writer.save()

        with open(output_path) as f:
            data = json.load(f)

        assert len(data["tracks"]) == 0
        assert data["statistics"]["total_tracks"] == 0
        assert data["statistics"]["total_detections"] == 0

    def test_object_names(self, tmp_path, sample_result):
        """Test object name mapping."""
        output_path = tmp_path / "output.bbox.json"

        writer = BBoxWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=1920,
            video_height=1080,
            obj_names={1: "mouse_1", 2: "mouse_2"},
        )
        writer.add_result(sample_result)
        writer.save()

        with open(output_path) as f:
            data = json.load(f)

        track_1 = next(t for t in data["tracks"] if t["track_id"] == 1)
        track_2 = next(t for t in data["tracks"] if t["track_id"] == 2)

        assert track_1["name"] == "mouse_1"
        assert track_2["name"] == "mouse_2"

    def test_detection_coordinates(self, tmp_path, sample_result):
        """Test detection coordinates are written correctly."""
        output_path = tmp_path / "output.bbox.json"

        writer = BBoxWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=1920,
            video_height=1080,
        )
        writer.add_result(sample_result)
        writer.save()

        with open(output_path) as f:
            data = json.load(f)

        track_1 = next(t for t in data["tracks"] if t["track_id"] == 1)
        det = track_1["detections"][0]

        assert det["x_min"] == 10.0
        assert det["y_min"] == 20.0
        assert det["x_max"] == 50.0
        assert det["y_max"] == 60.0
        assert det["width"] == 40.0
        assert det["height"] == 40.0
        assert det["confidence"] == 0.95
