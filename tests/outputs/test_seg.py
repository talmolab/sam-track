"""Tests for SegmentationWriter."""

import h5py
import numpy as np
import pytest

from sam_track.outputs.seg import SegmentationWriter
from sam_track.tracker import TrackingResult


@pytest.fixture
def sample_result():
    """Create a sample TrackingResult for testing."""
    return TrackingResult(
        frame_idx=0,
        object_ids=np.array([1, 2]),
        masks=np.random.randint(0, 2, (2, 100, 100)).astype(bool),
        boxes=np.array([[10.0, 20.0, 50.0, 60.0], [100.0, 150.0, 200.0, 250.0]]),
        scores=np.array([0.95, 0.88]),
    )


@pytest.fixture
def sample_result_frame_5():
    """Create a sample TrackingResult for frame 5."""
    return TrackingResult(
        frame_idx=5,
        object_ids=np.array([1]),
        masks=np.random.randint(0, 2, (1, 100, 100)).astype(bool),
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


class TestSegmentationWriter:
    """Tests for SegmentationWriter."""

    def test_creates_file(self, tmp_path, sample_result):
        """Test that finalize creates the output file."""
        output_path = tmp_path / "output.seg.h5"

        with SegmentationWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=100,
            video_height=100,
        ) as writer:
            writer.add_result(sample_result)

        assert output_path.exists()

    def test_creates_parent_dirs(self, tmp_path, sample_result):
        """Test that writer creates parent directories."""
        output_path = tmp_path / "subdir" / "nested" / "output.seg.h5"

        with SegmentationWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=100,
            video_height=100,
        ) as writer:
            writer.add_result(sample_result)

        assert output_path.exists()

    def test_metadata(self, tmp_path, sample_result):
        """Test metadata is written correctly."""
        output_path = tmp_path / "output.seg.h5"

        with SegmentationWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=100,
            video_height=100,
            fps=30.0,
            total_frames=1000,
        ) as writer:
            writer.add_result(sample_result)

        with h5py.File(output_path, "r") as f:
            meta = f["metadata"]
            assert meta.attrs["version"] == "1.0"
            assert meta.attrs["format"] == "sam-track-seg"
            assert meta.attrs["video_source"] == "video.mp4"
            assert meta.attrs["width"] == 100
            assert meta.attrs["height"] == 100
            assert meta.attrs["fps"] == 30.0
            assert meta.attrs["compression"] == "gzip"
            assert meta.attrs["compression_level"] == 1

    def test_masks_dataset_structure(self, tmp_path, sample_result):
        """Test masks dataset has correct structure."""
        output_path = tmp_path / "output.seg.h5"

        with SegmentationWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=100,
            video_height=100,
            max_objects=5,
        ) as writer:
            writer.add_result(sample_result)

        with h5py.File(output_path, "r") as f:
            masks = f["masks"]
            # Shape should be (T, N, H, W)
            assert masks.shape == (1, 5, 100, 100)
            assert masks.dtype == np.uint8
            assert masks.compression == "gzip"
            assert masks.shuffle is True

    def test_single_dataset_multiple_frames(
        self, tmp_path, sample_result, sample_result_frame_5
    ):
        """Test that multiple frames are stored in single dataset."""
        output_path = tmp_path / "output.seg.h5"

        with SegmentationWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=100,
            video_height=100,
            max_objects=5,
        ) as writer:
            writer.add_result(sample_result)
            writer.add_result(sample_result_frame_5)

        with h5py.File(output_path, "r") as f:
            masks = f["masks"]
            # Should have 2 frames
            assert masks.shape[0] == 2

            frame_indices = f["frame_indices"][:]
            assert frame_indices[0] == 0
            assert frame_indices[1] == 5

            num_objects = f["num_objects"][:]
            assert num_objects[0] == 2
            assert num_objects[1] == 1

    def test_track_ids(self, tmp_path, sample_result):
        """Test track IDs are stored correctly."""
        output_path = tmp_path / "output.seg.h5"

        with SegmentationWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=100,
            video_height=100,
            max_objects=5,
        ) as writer:
            writer.add_result(sample_result)

        with h5py.File(output_path, "r") as f:
            track_ids = f["track_ids"][:]
            # First two slots should have track IDs 1 and 2
            assert track_ids[0, 0] == 1
            assert track_ids[0, 1] == 2
            # Remaining slots should be -1 (fillvalue)
            assert track_ids[0, 2] == -1

    def test_confidences(self, tmp_path, sample_result):
        """Test confidences are stored correctly."""
        output_path = tmp_path / "output.seg.h5"

        with SegmentationWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=100,
            video_height=100,
            max_objects=5,
        ) as writer:
            writer.add_result(sample_result)

        with h5py.File(output_path, "r") as f:
            confidences = f["confidences"][:]
            assert np.isclose(confidences[0, 0], 0.95)
            assert np.isclose(confidences[0, 1], 0.88)

    def test_track_metadata(self, tmp_path, sample_result, sample_result_frame_5):
        """Test track metadata is written correctly."""
        output_path = tmp_path / "output.seg.h5"

        with SegmentationWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=100,
            video_height=100,
        ) as writer:
            writer.add_result(sample_result)
            writer.add_result(sample_result_frame_5)

        with h5py.File(output_path, "r") as f:
            # Track 1 should have 2 frames
            track_1 = f["tracks/track_1"]
            assert track_1.attrs["track_id"] == 1
            assert track_1.attrs["first_frame"] == 0
            assert track_1.attrs["last_frame"] == 5
            assert track_1.attrs["num_frames"] == 2

            # Track 2 should have 1 frame
            track_2 = f["tracks/track_2"]
            assert track_2.attrs["track_id"] == 2
            assert track_2.attrs["num_frames"] == 1

    def test_empty_result(self, tmp_path, empty_result):
        """Test handling of empty results."""
        output_path = tmp_path / "output.seg.h5"

        with SegmentationWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=100,
            video_height=100,
        ) as writer:
            writer.add_result(empty_result)

        with h5py.File(output_path, "r") as f:
            masks = f["masks"]
            assert masks.shape[0] == 1
            num_objects = f["num_objects"][0]
            assert num_objects == 0

    def test_object_names(self, tmp_path, sample_result):
        """Test object name mapping."""
        output_path = tmp_path / "output.seg.h5"

        with SegmentationWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=100,
            video_height=100,
            obj_names={1: "mouse_1", 2: "mouse_2"},
        ) as writer:
            writer.add_result(sample_result)

        with h5py.File(output_path, "r") as f:
            assert f["tracks/track_1"].attrs["name"] == "mouse_1"
            assert f["tracks/track_2"].attrs["name"] == "mouse_2"

    def test_dataset_resizing(self, tmp_path):
        """Test that dataset resizes for many frames."""
        output_path = tmp_path / "output.seg.h5"

        with SegmentationWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=50,
            video_height=50,
            total_frames=10,  # Start small
        ) as writer:
            # Add more frames than initial allocation
            for i in range(150):
                result = TrackingResult(
                    frame_idx=i,
                    object_ids=np.array([1]),
                    masks=np.zeros((1, 50, 50), dtype=bool),
                    boxes=np.array([[10.0, 20.0, 30.0, 40.0]]),
                    scores=np.array([0.9]),
                )
                writer.add_result(result)

        with h5py.File(output_path, "r") as f:
            assert f["masks"].shape[0] == 150
            assert f["frame_indices"].shape[0] == 150

    def test_context_manager(self, tmp_path, sample_result):
        """Test context manager properly finalizes."""
        output_path = tmp_path / "output.seg.h5"

        with SegmentationWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=100,
            video_height=100,
        ) as writer:
            writer.add_result(sample_result)

        # File should be readable after context exit
        with h5py.File(output_path, "r") as f:
            assert "masks" in f
            assert f["metadata"].attrs["frames_written"] == 1

    def test_max_objects_truncation(self, tmp_path):
        """Test that objects are truncated to max_objects."""
        output_path = tmp_path / "output.seg.h5"

        result = TrackingResult(
            frame_idx=0,
            object_ids=np.array([1, 2, 3, 4, 5]),
            masks=np.zeros((5, 50, 50), dtype=bool),
            boxes=np.zeros((5, 4)),
            scores=np.ones(5) * 0.9,
        )

        with SegmentationWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=50,
            video_height=50,
            max_objects=3,  # Only allow 3 objects
        ) as writer:
            writer.add_result(result)

        with h5py.File(output_path, "r") as f:
            num_objects = f["num_objects"][0]
            assert num_objects == 3
            # Only first 3 track IDs should be stored
            track_ids = f["track_ids"][0, :]
            assert track_ids[0] == 1
            assert track_ids[1] == 2
            assert track_ids[2] == 3

    def test_masks_are_compressed(self, tmp_path, sample_result):
        """Test that masks are actually compressed."""
        output_path = tmp_path / "output.seg.h5"

        with SegmentationWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=100,
            video_height=100,
        ) as writer:
            writer.add_result(sample_result)

        with h5py.File(output_path, "r") as f:
            masks = f["masks"]
            assert masks.compression == "gzip"
            assert masks.compression_opts == 1
