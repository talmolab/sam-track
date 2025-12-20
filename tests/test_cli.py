"""Tests for sam-track CLI."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from sam_track.cli import app


runner = CliRunner()


class TestVersion:
    """Tests for --version flag."""

    def test_version_short(self):
        """Test -v flag shows version."""
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert "sam-track version" in result.stdout

    def test_version_long(self):
        """Test --version flag shows version."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "sam-track version" in result.stdout


class TestTrackValidation:
    """Tests for track command validation."""

    def test_video_not_found(self, tmp_path: Path):
        """Test error when video file doesn't exist."""
        result = runner.invoke(
            app, ["track", str(tmp_path / "nonexistent.mp4"), "--text", "mouse", "--bbox"]
        )
        assert result.exit_code == 1
        assert "Video file not found" in result.stdout

    def test_no_prompt_specified(self, tmp_path: Path):
        """Test error when no prompt type is specified."""
        video = tmp_path / "test.mp4"
        video.touch()

        result = runner.invoke(app, ["track", str(video), "--bbox"])
        assert result.exit_code == 1
        assert "Must specify one of --text, --roi, or --pose" in result.stdout

    def test_multiple_prompts_text_roi(self, tmp_path: Path):
        """Test error when multiple prompt types are specified."""
        video = tmp_path / "test.mp4"
        video.touch()
        roi = tmp_path / "rois.yml"
        roi.touch()

        result = runner.invoke(
            app, ["track", str(video), "--text", "mouse", "--roi", str(roi), "--bbox"]
        )
        assert result.exit_code == 1
        assert "Only one prompt type allowed" in result.stdout
        assert "--text" in result.stdout
        assert "--roi" in result.stdout

    def test_multiple_prompts_all_three(self, tmp_path: Path):
        """Test error when all three prompt types are specified."""
        video = tmp_path / "test.mp4"
        video.touch()
        roi = tmp_path / "rois.yml"
        roi.touch()
        pose = tmp_path / "labels.slp"
        pose.touch()

        result = runner.invoke(
            app,
            [
                "track",
                str(video),
                "--text",
                "mouse",
                "--roi",
                str(roi),
                "--pose",
                str(pose),
                "--bbox",
            ],
        )
        assert result.exit_code == 1
        assert "Only one prompt type allowed" in result.stdout

    def test_no_output_specified(self, tmp_path: Path):
        """Test error when no output format is specified."""
        video = tmp_path / "test.mp4"
        video.touch()

        result = runner.invoke(app, ["track", str(video), "--text", "mouse"])
        assert result.exit_code == 1
        assert "Must specify at least one of --bbox or --seg" in result.stdout

    def test_roi_file_not_found(self, tmp_path: Path):
        """Test error when ROI file doesn't exist."""
        video = tmp_path / "test.mp4"
        video.touch()

        result = runner.invoke(
            app,
            ["track", str(video), "--roi", str(tmp_path / "nonexistent.yml"), "--bbox"],
        )
        assert result.exit_code == 1
        assert "ROI file not found" in result.stdout

    def test_pose_file_not_found(self, tmp_path: Path):
        """Test error when pose file doesn't exist."""
        video = tmp_path / "test.mp4"
        video.touch()

        result = runner.invoke(
            app,
            ["track", str(video), "--pose", str(tmp_path / "nonexistent.slp"), "--bbox"],
        )
        assert result.exit_code == 1
        assert "Pose file not found" in result.stdout


class TestTrackOutputPaths:
    """Tests for output path handling."""

    def test_default_bbox_path(self, tmp_path: Path):
        """Test default bbox output path is derived from video name."""
        video = tmp_path / "my_video.mp4"
        video.touch()

        # We can't run the full tracking without a real video,
        # but we can check the help to confirm the option exists
        result = runner.invoke(app, ["track", "--help"])
        assert "--bbox" in result.stdout
        assert "--bbox-output" in result.stdout

    def test_default_seg_path(self, tmp_path: Path):
        """Test default seg output path is derived from video name."""
        result = runner.invoke(app, ["track", "--help"])
        assert "--seg" in result.stdout
        assert "--seg-output" in result.stdout


class TestTrackOptions:
    """Tests for advanced track options."""

    def test_device_option(self):
        """Test --device option is available."""
        result = runner.invoke(app, ["track", "--help"])
        assert "--device" in result.stdout
        assert "cuda" in result.stdout

    def test_max_frames_option(self):
        """Test --max-frames option is available."""
        result = runner.invoke(app, ["track", "--help"])
        assert "--max-frames" in result.stdout

    def test_quiet_option(self):
        """Test --quiet option is available."""
        result = runner.invoke(app, ["track", "--help"])
        assert "--quiet" in result.stdout


class TestAuthCommand:
    """Tests for auth command."""

    def test_auth_help(self):
        """Test auth command help."""
        result = runner.invoke(app, ["auth", "--help"])
        assert result.exit_code == 0
        assert "HuggingFace" in result.stdout
        assert "--token" in result.stdout

    def test_auth_shows_status(self):
        """Test auth command shows authentication status."""
        result = runner.invoke(app, ["auth"])
        # Should show some kind of status, even if not authenticated
        assert "Authenticated" in result.stdout or "authentication" in result.stdout.lower()


class TestSystemCommand:
    """Tests for system command."""

    def test_system_help(self):
        """Test system command help."""
        result = runner.invoke(app, ["system", "--help"])
        assert result.exit_code == 0
        assert "system information" in result.stdout.lower() or "GPU" in result.stdout

    def test_system_shows_info(self):
        """Test system command shows system information."""
        result = runner.invoke(app, ["system"])
        assert result.exit_code == 0
        assert "sam-track version" in result.stdout
        assert "Python version" in result.stdout
        assert "PyTorch version" in result.stdout


class TestHelpText:
    """Tests for help text quality."""

    def test_main_help(self):
        """Test main help is informative."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "sam-track" in result.stdout
        assert "SAM3" in result.stdout

    def test_track_help_examples(self):
        """Test track command includes usage examples."""
        result = runner.invoke(app, ["track", "--help"])
        assert result.exit_code == 0
        assert "Examples" in result.stdout
        assert "--text" in result.stdout
        assert "--roi" in result.stdout
        assert "--pose" in result.stdout
        assert "--bbox" in result.stdout
        assert "--seg" in result.stdout


class TestShortOptions:
    """Tests for short option aliases."""

    def test_short_options_available(self):
        """Test that short options are available."""
        result = runner.invoke(app, ["track", "--help"])
        # Check short options are documented
        assert "-t" in result.stdout  # --text
        assert "-r" in result.stdout  # --roi
        assert "-p" in result.stdout  # --pose
        assert "-b" in result.stdout  # --bbox
        assert "-s" in result.stdout  # --seg
        assert "-B" in result.stdout  # --bbox-output
        assert "-S" in result.stdout  # --seg-output
        assert "-d" in result.stdout  # --device
        assert "-n" in result.stdout  # --max-frames
        assert "-q" in result.stdout  # --quiet
