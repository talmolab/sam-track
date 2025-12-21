"""Tests for SLP output writer."""

from pathlib import Path

import pytest
import sleap_io as sio

from sam_track.outputs import SLPWriter
from sam_track.reconciliation import TrackAssignment

# Test data path
DATA_DIR = Path(__file__).parent.parent / "data"
SLP_3NODE = DATA_DIR / "labels.3node.first_frame.slp"


@pytest.fixture
def source_labels():
    """Load source labels for testing."""
    return sio.load_slp(str(SLP_3NODE), open_videos=False)


@pytest.fixture
def sample_assignments():
    """Create sample track assignments."""
    return [
        TrackAssignment(
            frame_idx=0,
            pose_track_name="mouse1",
            pose_idx=0,
            sam3_obj_id=0,
            confidence=0.9,
        ),
        TrackAssignment(
            frame_idx=0,
            pose_track_name="mouse2",
            pose_idx=1,
            sam3_obj_id=1,
            confidence=0.85,
        ),
    ]


class TestSLPWriter:
    """Tests for SLPWriter class."""

    def test_init(self, source_labels, tmp_path):
        """Test writer initialization."""
        output_path = tmp_path / "output.slp"
        writer = SLPWriter(
            output_path=output_path,
            source_labels=source_labels,
            remove_unmatched=False,
        )
        assert writer.output_path == output_path
        assert writer.remove_unmatched is False
        assert writer.num_frames == 0
        assert writer.num_tracks == 0

    def test_add_frame_assignments(self, source_labels, sample_assignments, tmp_path):
        """Test adding frame assignments."""
        output_path = tmp_path / "output.slp"
        writer = SLPWriter(
            output_path=output_path,
            source_labels=source_labels,
        )

        # Get instances from source
        lf = source_labels.labeled_frames[0]
        writer.add_frame_assignments(
            frame_idx=0,
            assignments=sample_assignments,
            original_instances=list(lf.instances),
        )

        assert writer.num_frames == 1

    def test_finalize_creates_file(self, source_labels, sample_assignments, tmp_path):
        """Test that finalize creates output file."""
        output_path = tmp_path / "output.slp"
        writer = SLPWriter(
            output_path=output_path,
            source_labels=source_labels,
        )

        lf = source_labels.labeled_frames[0]
        writer.add_frame_assignments(
            frame_idx=0,
            assignments=sample_assignments,
            original_instances=list(lf.instances),
        )

        result = writer.finalize()

        assert output_path.exists()
        assert isinstance(result, sio.Labels)

    def test_finalize_assigns_tracks(self, source_labels, sample_assignments, tmp_path):
        """Test that finalize assigns track names correctly."""
        output_path = tmp_path / "output.slp"
        writer = SLPWriter(
            output_path=output_path,
            source_labels=source_labels,
        )

        lf = source_labels.labeled_frames[0]
        writer.add_frame_assignments(
            frame_idx=0,
            assignments=sample_assignments,
            original_instances=list(lf.instances),
        )

        result = writer.finalize()

        # Check tracks were created
        assert len(result.tracks) == 2
        track_names = {t.name for t in result.tracks}
        assert "mouse1" in track_names
        assert "mouse2" in track_names

        # Check instances have tracks
        for inst in result.labeled_frames[0].instances:
            assert inst.track is not None

    def test_remove_unmatched_true(self, source_labels, tmp_path):
        """Test that unmatched instances are removed when flag is set."""
        output_path = tmp_path / "output.slp"
        writer = SLPWriter(
            output_path=output_path,
            source_labels=source_labels,
            remove_unmatched=True,
        )

        lf = source_labels.labeled_frames[0]
        # Only assign one instance
        assignments = [
            TrackAssignment(
                frame_idx=0,
                pose_track_name="mouse1",
                pose_idx=0,
                sam3_obj_id=0,
                confidence=0.9,
            ),
        ]
        writer.add_frame_assignments(
            frame_idx=0,
            assignments=assignments,
            original_instances=list(lf.instances),
        )

        result = writer.finalize()

        # Should only have 1 instance (the matched one)
        assert len(result.labeled_frames[0].instances) == 1

    def test_remove_unmatched_false(self, source_labels, tmp_path):
        """Test that unmatched instances are kept when flag is false."""
        output_path = tmp_path / "output.slp"
        writer = SLPWriter(
            output_path=output_path,
            source_labels=source_labels,
            remove_unmatched=False,
        )

        lf = source_labels.labeled_frames[0]
        # Only assign one instance
        assignments = [
            TrackAssignment(
                frame_idx=0,
                pose_track_name="mouse1",
                pose_idx=0,
                sam3_obj_id=0,
                confidence=0.9,
            ),
        ]
        writer.add_frame_assignments(
            frame_idx=0,
            assignments=assignments,
            original_instances=list(lf.instances),
        )

        result = writer.finalize()

        # Should have both instances
        assert len(result.labeled_frames[0].instances) == 2

    def test_sam3_id_fallback(self, source_labels, tmp_path):
        """Test that SAM3 ID is used when no track name provided."""
        output_path = tmp_path / "output.slp"
        writer = SLPWriter(
            output_path=output_path,
            source_labels=source_labels,
        )

        lf = source_labels.labeled_frames[0]
        # Assignment without track name
        assignments = [
            TrackAssignment(
                frame_idx=0,
                pose_track_name=None,  # No track name
                pose_idx=0,
                sam3_obj_id=42,
                confidence=0.9,
            ),
        ]
        writer.add_frame_assignments(
            frame_idx=0,
            assignments=assignments,
            original_instances=list(lf.instances),
        )

        result = writer.finalize()

        # Track should be named "track_42"
        track_names = {t.name for t in result.tracks}
        assert "track_42" in track_names

    def test_creates_parent_directory(
        self, source_labels, sample_assignments, tmp_path
    ):
        """Test that parent directory is created if needed."""
        output_path = tmp_path / "nested" / "dir" / "output.slp"
        writer = SLPWriter(
            output_path=output_path,
            source_labels=source_labels,
        )

        lf = source_labels.labeled_frames[0]
        writer.add_frame_assignments(
            frame_idx=0,
            assignments=sample_assignments,
            original_instances=list(lf.instances),
        )

        writer.finalize()

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_context_manager(self, source_labels, sample_assignments, tmp_path):
        """Test context manager interface."""
        output_path = tmp_path / "output.slp"

        with SLPWriter(
            output_path=output_path,
            source_labels=source_labels,
        ) as writer:
            lf = source_labels.labeled_frames[0]
            writer.add_frame_assignments(
                frame_idx=0,
                assignments=sample_assignments,
                original_instances=list(lf.instances),
            )

        # File should exist after context manager exits
        assert output_path.exists()

    def test_multiple_frames(self, source_labels, tmp_path):
        """Test writing multiple frames."""
        output_path = tmp_path / "output.slp"
        writer = SLPWriter(
            output_path=output_path,
            source_labels=source_labels,
        )

        lf = source_labels.labeled_frames[0]
        instances = list(lf.instances)

        # Add same instances at different frames
        for frame_idx in [0, 10, 20]:
            assignments = [
                TrackAssignment(
                    frame_idx=frame_idx,
                    pose_track_name="mouse1",
                    pose_idx=0,
                    sam3_obj_id=0,
                    confidence=0.9,
                ),
            ]
            writer.add_frame_assignments(
                frame_idx=frame_idx,
                assignments=assignments,
                original_instances=instances,
            )

        result = writer.finalize()

        assert len(result.labeled_frames) == 3
        assert writer.num_frames == 3

    def test_preserves_skeleton(self, source_labels, sample_assignments, tmp_path):
        """Test that skeleton structure is preserved."""
        output_path = tmp_path / "output.slp"
        writer = SLPWriter(
            output_path=output_path,
            source_labels=source_labels,
        )

        lf = source_labels.labeled_frames[0]
        writer.add_frame_assignments(
            frame_idx=0,
            assignments=sample_assignments,
            original_instances=list(lf.instances),
        )

        result = writer.finalize()

        # Skeleton should match source
        assert len(result.skeletons) == 1
        assert len(result.skeleton.nodes) == len(source_labels.skeleton.nodes)
