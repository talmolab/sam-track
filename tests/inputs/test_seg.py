"""Tests for SegmentationReader."""

import numpy as np
import pytest

from sam_track.inputs.seg import SegmentationReader
from sam_track.outputs.seg import SegmentationWriter
from sam_track.tracker import TrackingResult


@pytest.fixture
def sample_seg_file(tmp_path):
    """Create a sample seg.h5 file for testing."""
    output_path = tmp_path / "test.seg.h5"

    result_0 = TrackingResult(
        frame_idx=0,
        object_ids=np.array([1, 2]),
        masks=np.random.randint(0, 2, (2, 100, 100)).astype(bool),
        boxes=np.array([[10.0, 20.0, 50.0, 60.0], [100.0, 150.0, 200.0, 250.0]]),
        scores=np.array([0.95, 0.88]),
    )

    result_5 = TrackingResult(
        frame_idx=5,
        object_ids=np.array([1]),
        masks=np.random.randint(0, 2, (1, 100, 100)).astype(bool),
        boxes=np.array([[15.0, 25.0, 55.0, 65.0]]),
        scores=np.array([0.92]),
    )

    result_10 = TrackingResult(
        frame_idx=10,
        object_ids=np.array([1, 2, 3]),
        masks=np.random.randint(0, 2, (3, 100, 100)).astype(bool),
        boxes=np.array([
            [10.0, 20.0, 50.0, 60.0],
            [100.0, 150.0, 200.0, 250.0],
            [50.0, 50.0, 80.0, 80.0],
        ]),
        scores=np.array([0.91, 0.85, 0.78]),
    )

    with SegmentationWriter(
        output_path=output_path,
        video_path="video.mp4",
        video_width=100,
        video_height=100,
        fps=30.0,
        total_frames=100,
        obj_names={1: "mouse_1", 2: "mouse_2", 3: "mouse_3"},
    ) as writer:
        writer.add_result(result_0)
        writer.add_result(result_5)
        writer.add_result(result_10)

    return output_path


@pytest.fixture
def empty_frame_seg_file(tmp_path):
    """Create a seg.h5 file with an empty frame."""
    output_path = tmp_path / "empty_frame.seg.h5"

    result_0 = TrackingResult(
        frame_idx=0,
        object_ids=np.array([1]),
        masks=np.random.randint(0, 2, (1, 50, 50)).astype(bool),
        boxes=np.array([[10.0, 20.0, 30.0, 40.0]]),
        scores=np.array([0.9]),
    )

    result_empty = TrackingResult(
        frame_idx=5,
        object_ids=np.array([], dtype=np.int64),
        masks=np.zeros((0, 50, 50), dtype=bool),
        boxes=np.zeros((0, 4)),
        scores=np.array([]),
    )

    with SegmentationWriter(
        output_path=output_path,
        video_path="video.mp4",
        video_width=50,
        video_height=50,
    ) as writer:
        writer.add_result(result_0)
        writer.add_result(result_empty)

    return output_path


class TestSegmentationReader:
    """Tests for SegmentationReader."""

    def test_file_not_found(self, tmp_path):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Segmentation file not found"):
            SegmentationReader(tmp_path / "nonexistent.seg.h5")

    def test_metadata_properties(self, sample_seg_file):
        """Test metadata is read correctly."""
        reader = SegmentationReader(sample_seg_file)

        assert reader.version == "1.0"
        assert reader.format == "sam-track-seg"
        assert reader.video_source == "video.mp4"
        assert reader.width == 100
        assert reader.height == 100
        assert reader.resolution == (100, 100)
        assert reader.fps == 30.0
        assert reader.num_frames == 3
        assert reader.num_tracks == 3

        reader.close()

    def test_labeled_frame_indices(self, sample_seg_file):
        """Test labeled frame indices are correct."""
        reader = SegmentationReader(sample_seg_file)

        indices = reader.labeled_frame_indices
        assert indices == [0, 5, 10]

        reader.close()

    def test_has_frame(self, sample_seg_file):
        """Test has_frame method."""
        reader = SegmentationReader(sample_seg_file)

        assert reader.has_frame(0) is True
        assert reader.has_frame(5) is True
        assert reader.has_frame(10) is True
        assert reader.has_frame(3) is False
        assert reader.has_frame(100) is False

        reader.close()

    def test_get_num_objects(self, sample_seg_file):
        """Test get_num_objects method."""
        reader = SegmentationReader(sample_seg_file)

        assert reader.get_num_objects(0) == 2
        assert reader.get_num_objects(5) == 1
        assert reader.get_num_objects(10) == 3

        reader.close()

    def test_get_num_objects_missing_frame(self, sample_seg_file):
        """Test get_num_objects raises for missing frame."""
        reader = SegmentationReader(sample_seg_file)

        with pytest.raises(KeyError, match="Frame 99 not in segmentation file"):
            reader.get_num_objects(99)

        reader.close()

    def test_get_masks(self, sample_seg_file):
        """Test get_masks returns correct shape and type."""
        reader = SegmentationReader(sample_seg_file)

        masks_0 = reader.get_masks(0)
        assert masks_0.shape == (2, 100, 100)
        assert masks_0.dtype == np.uint8

        masks_5 = reader.get_masks(5)
        assert masks_5.shape == (1, 100, 100)

        masks_10 = reader.get_masks(10)
        assert masks_10.shape == (3, 100, 100)

        reader.close()

    def test_get_masks_missing_frame(self, sample_seg_file):
        """Test get_masks raises for missing frame."""
        reader = SegmentationReader(sample_seg_file)

        with pytest.raises(KeyError, match="Frame 99 not in segmentation file"):
            reader.get_masks(99)

        reader.close()

    def test_get_track_ids(self, sample_seg_file):
        """Test get_track_ids method."""
        reader = SegmentationReader(sample_seg_file)

        track_ids_0 = reader.get_track_ids(0)
        assert list(track_ids_0) == [1, 2]

        track_ids_5 = reader.get_track_ids(5)
        assert list(track_ids_5) == [1]

        track_ids_10 = reader.get_track_ids(10)
        assert list(track_ids_10) == [1, 2, 3]

        reader.close()

    def test_get_confidences(self, sample_seg_file):
        """Test get_confidences method."""
        reader = SegmentationReader(sample_seg_file)

        confidences_0 = reader.get_confidences(0)
        assert np.allclose(confidences_0, [0.95, 0.88])

        confidences_5 = reader.get_confidences(5)
        assert np.allclose(confidences_5, [0.92])

        reader.close()

    def test_get_track_names(self, sample_seg_file):
        """Test get_track_names method."""
        reader = SegmentationReader(sample_seg_file)

        names = reader.get_track_names()
        assert names == {1: "mouse_1", 2: "mouse_2", 3: "mouse_3"}

        reader.close()

    def test_get_track_info(self, sample_seg_file):
        """Test get_track_info method."""
        reader = SegmentationReader(sample_seg_file)

        info_1 = reader.get_track_info(1)
        assert info_1["track_id"] == 1
        assert info_1["name"] == "mouse_1"
        assert info_1["first_frame"] == 0
        assert info_1["last_frame"] == 10
        assert info_1["num_frames"] == 3

        info_3 = reader.get_track_info(3)
        assert info_3["track_id"] == 3
        assert info_3["name"] == "mouse_3"
        assert info_3["first_frame"] == 10
        assert info_3["last_frame"] == 10
        assert info_3["num_frames"] == 1

        reader.close()

    def test_get_track_info_missing(self, sample_seg_file):
        """Test get_track_info raises for missing track."""
        reader = SegmentationReader(sample_seg_file)

        with pytest.raises(KeyError, match="Track 99 not found"):
            reader.get_track_info(99)

        reader.close()

    def test_empty_frame(self, empty_frame_seg_file):
        """Test reading frame with no objects."""
        reader = SegmentationReader(empty_frame_seg_file)

        assert reader.get_num_objects(5) == 0

        masks = reader.get_masks(5)
        assert masks.shape == (0, 50, 50)

        track_ids = reader.get_track_ids(5)
        assert track_ids.shape == (0,)

        confidences = reader.get_confidences(5)
        assert confidences.shape == (0,)

        reader.close()

    def test_context_manager(self, sample_seg_file):
        """Test context manager usage."""
        with SegmentationReader(sample_seg_file) as reader:
            assert reader.num_frames == 3
            masks = reader.get_masks(0)
            assert masks.shape == (2, 100, 100)

    def test_lazy_loading(self, sample_seg_file):
        """Test that file is lazily loaded."""
        reader = SegmentationReader(sample_seg_file)

        # File handle should be None before any access
        assert reader._file is None

        # Access something that requires loading
        _ = reader.num_frames

        # Now file should be open
        assert reader._file is not None

        reader.close()

    def test_frame_map_caching(self, sample_seg_file):
        """Test that frame map is built once and cached."""
        reader = SegmentationReader(sample_seg_file)

        # First access builds map
        _ = reader.labeled_frame_indices
        map1 = reader._frame_map

        # Second access should return same object
        _ = reader.has_frame(0)
        map2 = reader._frame_map

        assert map1 is map2

        reader.close()

    def test_repr(self, sample_seg_file):
        """Test string representation."""
        reader = SegmentationReader(sample_seg_file)

        repr_str = repr(reader)
        assert "SegmentationReader" in repr_str
        assert "num_frames=3" in repr_str
        assert "num_tracks=3" in repr_str

        reader.close()

    def test_close_twice(self, sample_seg_file):
        """Test that close can be called multiple times safely."""
        reader = SegmentationReader(sample_seg_file)
        _ = reader.num_frames  # Open file

        reader.close()
        reader.close()  # Should not raise

    def test_roundtrip_masks(self, tmp_path):
        """Test that masks survive a write-read roundtrip."""
        output_path = tmp_path / "roundtrip.seg.h5"

        # Create specific mask pattern
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[20:40, 30:60] = 1

        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[50:80, 10:90] = 1

        result = TrackingResult(
            frame_idx=0,
            object_ids=np.array([1, 2]),
            masks=np.stack([mask1, mask2]),
            boxes=np.array([[30.0, 20.0, 60.0, 40.0], [10.0, 50.0, 90.0, 80.0]]),
            scores=np.array([0.9, 0.8]),
        )

        # Write
        with SegmentationWriter(
            output_path=output_path,
            video_path="video.mp4",
            video_width=100,
            video_height=100,
        ) as writer:
            writer.add_result(result)

        # Read
        with SegmentationReader(output_path) as reader:
            masks = reader.get_masks(0)
            assert np.array_equal(masks[0], mask1)
            assert np.array_equal(masks[1], mask2)
