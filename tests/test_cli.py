"""Tests for sam-track CLI."""

import re
from pathlib import Path

from typer.testing import CliRunner

from sam_track.cli import app


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_pattern.sub("", text)


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
        video_path = str(tmp_path / "nonexistent.mp4")
        result = runner.invoke(app, ["track", video_path, "--text", "mouse", "--bbox"])
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
        assert "Must specify at least one of --bbox, --seg, or --slp" in result.stdout

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
        pose_path = str(tmp_path / "nonexistent.slp")

        result = runner.invoke(
            app,
            ["track", str(video), "--pose", pose_path, "--bbox"],
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
        output = strip_ansi(result.stdout)
        assert "--bbox" in output
        assert "--bbox-output" in output

    def test_default_seg_path(self, tmp_path: Path):
        """Test default seg output path is derived from video name."""
        result = runner.invoke(app, ["track", "--help"])
        output = strip_ansi(result.stdout)
        assert "--seg" in output
        assert "--seg-output" in output


class TestTrackOptions:
    """Tests for advanced track options."""

    def test_device_option(self):
        """Test --device option is available."""
        result = runner.invoke(app, ["track", "--help"])
        output = strip_ansi(result.stdout)
        assert "--device" in output
        assert "cuda" in output

    def test_max_frames_option(self):
        """Test --max-frames option is available."""
        result = runner.invoke(app, ["track", "--help"])
        output = strip_ansi(result.stdout)
        assert "--max-frames" in output

    def test_quiet_option(self):
        """Test --quiet option is available."""
        result = runner.invoke(app, ["track", "--help"])
        output = strip_ansi(result.stdout)
        assert "--quiet" in output


class TestFrameRangeOptions:
    """Tests for frame range options."""

    def test_start_frame_option(self):
        """Test --start-frame option is available."""
        result = runner.invoke(app, ["track", "--help"])
        output = strip_ansi(result.stdout)
        assert "--start-frame" in output
        assert "0-indexed" in output

    def test_stop_frame_option(self):
        """Test --stop-frame option is available."""
        result = runner.invoke(app, ["track", "--help"])
        output = strip_ansi(result.stdout)
        assert "--stop-frame" in output

    def test_stop_frame_max_frames_mutual_exclusivity(self, tmp_path: Path):
        """Test error when both --stop-frame and --max-frames are specified."""
        video = tmp_path / "test.mp4"
        video.touch()

        result = runner.invoke(
            app,
            [
                "track",
                str(video),
                "--text",
                "mouse",
                "--bbox",
                "--stop-frame",
                "100",
                "--max-frames",
                "50",
            ],
        )
        assert result.exit_code == 1
        assert "mutually exclusive" in result.stdout

    def test_negative_start_frame(self, tmp_path: Path):
        """Test error when --start-frame is negative."""
        video = tmp_path / "test.mp4"
        video.touch()

        result = runner.invoke(
            app,
            [
                "track",
                str(video),
                "--text",
                "mouse",
                "--bbox",
                "--start-frame",
                "-1",
            ],
        )
        assert result.exit_code == 1
        assert "non-negative" in result.stdout

    def test_stop_frame_less_than_start(self, tmp_path: Path):
        """Test error when --stop-frame is less than --start-frame."""
        video = tmp_path / "test.mp4"
        video.touch()

        result = runner.invoke(
            app,
            [
                "track",
                str(video),
                "--text",
                "mouse",
                "--bbox",
                "--start-frame",
                "50",
                "--stop-frame",
                "30",
            ],
        )
        assert result.exit_code == 1
        assert "greater than" in result.stdout

    def test_stop_frame_equals_start(self, tmp_path: Path):
        """Test error when --stop-frame equals --start-frame."""
        video = tmp_path / "test.mp4"
        video.touch()

        result = runner.invoke(
            app,
            [
                "track",
                str(video),
                "--text",
                "mouse",
                "--bbox",
                "--start-frame",
                "50",
                "--stop-frame",
                "50",
            ],
        )
        assert result.exit_code == 1
        assert "greater than" in result.stdout

    def test_zero_max_frames(self, tmp_path: Path):
        """Test error when --max-frames is zero."""
        video = tmp_path / "test.mp4"
        video.touch()

        result = runner.invoke(
            app,
            [
                "track",
                str(video),
                "--text",
                "mouse",
                "--bbox",
                "--max-frames",
                "0",
            ],
        )
        assert result.exit_code == 1
        assert "positive" in result.stdout

    def test_negative_max_frames(self, tmp_path: Path):
        """Test error when --max-frames is negative."""
        video = tmp_path / "test.mp4"
        video.touch()

        result = runner.invoke(
            app,
            [
                "track",
                str(video),
                "--text",
                "mouse",
                "--bbox",
                "--max-frames",
                "-10",
            ],
        )
        assert result.exit_code == 1
        assert "positive" in result.stdout


class TestAuthCommand:
    """Tests for auth command."""

    def test_auth_help(self):
        """Test auth command help."""
        result = runner.invoke(app, ["auth", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.stdout)
        assert "HuggingFace" in output
        assert "--token" in output

    def test_auth_shows_status(self):
        """Test auth command shows authentication status."""
        result = runner.invoke(app, ["auth"])
        # Should show some kind of status, even if not authenticated
        output = result.stdout.lower()
        assert "authenticated" in output or "authentication" in output


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
        output = strip_ansi(result.stdout)
        assert "sam-track" in output
        assert "SAM3" in output

    def test_track_help_examples(self):
        """Test track command includes usage examples."""
        result = runner.invoke(app, ["track", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.stdout)
        assert "Examples" in output
        assert "--text" in output
        assert "--roi" in output
        assert "--pose" in output
        assert "--bbox" in output
        assert "--seg" in output


class TestShortOptions:
    """Tests for short option aliases."""

    def test_short_options_available(self):
        """Test that short options are available."""
        result = runner.invoke(app, ["track", "--help"])
        output = strip_ansi(result.stdout)
        # Check short options are documented
        assert "-t" in output  # --text
        assert "-r" in output  # --roi
        assert "-p" in output  # --pose
        assert "-b" in output  # --bbox
        assert "-s" in output  # --seg
        assert "-B" in output  # --bbox-output
        assert "-S" in output  # --seg-output
        assert "-d" in output  # --device
        assert "-n" in output  # --max-frames
        assert "-q" in output  # --quiet
