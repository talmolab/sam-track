"""sam-track CLI using Typer."""

from __future__ import annotations

import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from . import __version__

# Minimum NVIDIA driver versions required for CUDA 13.0
# See: https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/
MIN_DRIVER_VERSION_LINUX = "580.65.06"
MIN_DRIVER_VERSION_WINDOWS = "580.65"

app = typer.Typer(
    name="sam-track",
    help="Track objects in videos using SAM3.",
    add_completion=False,
)
console = Console()

# Global interrupt flag for graceful shutdown
_interrupted = False


def _signal_handler(sig, frame):
    """Handle interrupt signal for graceful shutdown."""
    global _interrupted
    _interrupted = True
    console.print("\n[yellow]Interrupt received, finishing current frame...[/yellow]")


class FPSColumn(ProgressColumn):
    """Display processing speed in frames per second."""

    max_refresh = 0.5

    def render(self, task: Task) -> Text:
        if task.speed is not None:
            return Text(f"{task.speed:.1f} fps", style="green")
        return Text("-- fps", style="dim")


class ObjectsColumn(ProgressColumn):
    """Display number of objects being tracked."""

    def render(self, task: Task) -> Text:
        count = task.fields.get("objects", 0)
        return Text(f"{count} obj", style="cyan")


def parse_driver_version(version: str) -> tuple[int, ...]:
    """Parse driver version string into comparable tuple."""
    try:
        return tuple(int(x) for x in version.split("."))
    except ValueError:
        return (0,)


def check_driver_version(driver_version: str) -> tuple[bool, str]:
    """Check if driver version meets minimum requirements for CUDA 13.0.

    Returns:
        Tuple of (is_compatible, minimum_version).
    """
    if sys.platform == "win32":
        min_version = MIN_DRIVER_VERSION_WINDOWS
    else:
        min_version = MIN_DRIVER_VERSION_LINUX

    current = parse_driver_version(driver_version)
    required = parse_driver_version(min_version)

    # Pad tuples to same length for comparison
    max_len = max(len(current), len(required))
    current = current + (0,) * (max_len - len(current))
    required = required + (0,) * (max_len - len(required))

    return current >= required, min_version


def get_nvidia_driver_version() -> str | None:
    """Get NVIDIA driver version from nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return None


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"sam-track version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """sam-track: Track objects in videos using SAM3."""
    pass


@app.command()
def track(
    video: Annotated[Path, typer.Argument(help="Path to input video file.")],
    # Prompt options (mutually exclusive)
    text: Annotated[
        str | None,
        typer.Option(
            "--text",
            "-t",
            help='Text prompt for object detection (e.g., "mouse", "person").',
        ),
    ] = None,
    roi: Annotated[
        Path | None,
        typer.Option(
            "--roi",
            "-r",
            help="ROI YAML file from labelroi with polygon annotations.",
        ),
    ] = None,
    pose: Annotated[
        Path | None,
        typer.Option(
            "--pose",
            "-p",
            help="SLEAP SLP file with pose annotations for point prompts.",
        ),
    ] = None,
    # Output options
    bbox: Annotated[
        bool,
        typer.Option(
            "--bbox",
            "-b",
            help="Enable bounding box output (default: <video>.bbox.json).",
        ),
    ] = False,
    bbox_output: Annotated[
        Path | None,
        typer.Option(
            "--bbox-output",
            "-B",
            help="Custom path for bounding box output (implies --bbox).",
        ),
    ] = None,
    seg: Annotated[
        bool,
        typer.Option(
            "--seg",
            "-s",
            help="Enable segmentation mask output (default: <video>.seg.h5).",
        ),
    ] = False,
    seg_output: Annotated[
        Path | None,
        typer.Option(
            "--seg-output",
            "-S",
            help="Custom path for segmentation output (implies --seg).",
        ),
    ] = None,
    slp_output: Annotated[
        Path | None,
        typer.Option(
            "--slp",
            help="Output path for tracked SLP (default: <pose>.sam-tracked.slp). "
            "Only valid with --pose.",
        ),
    ] = None,
    # Pose integration options
    remove_unmatched: Annotated[
        bool,
        typer.Option(
            "--remove-unmatched",
            help="Remove poses that couldn't be matched to a SAM3 mask from output.",
        ),
    ] = False,
    exclude_nodes: Annotated[
        str | None,
        typer.Option(
            "--exclude-nodes",
            help="Comma-separated list of nodes to exclude from matching.",
        ),
    ] = None,
    filter_by_pose: Annotated[
        bool,
        typer.Option(
            "--filter-by-pose",
            help="Only include masks/boxes that matched a pose in output.",
        ),
    ] = False,
    # Advanced options
    device: Annotated[
        str | None,
        typer.Option(
            "--device",
            "-d",
            help="Device for inference (e.g., cuda, cuda:0, mps, cpu).",
        ),
    ] = None,
    start_frame: Annotated[
        int,
        typer.Option(
            "--start-frame",
            help="Frame index to start processing from (0-indexed).",
        ),
    ] = 0,
    stop_frame: Annotated[
        int | None,
        typer.Option(
            "--stop-frame",
            help="Frame index to stop processing at (exclusive). "
            "Mutually exclusive with --max-frames.",
        ),
    ] = None,
    max_frames: Annotated[
        int | None,
        typer.Option(
            "--max-frames",
            "-n",
            help="Maximum number of frames to process from start. "
            "Mutually exclusive with --stop-frame.",
        ),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress progress output.",
        ),
    ] = False,
    preload: Annotated[
        bool,
        typer.Option(
            "--preload",
            help=(
                "Preload all video frames before tracking. "
                "Uses more memory but may be slightly faster for text prompts."
            ),
        ),
    ] = False,
) -> None:
    """Track objects in a video using SAM3.

    Requires exactly one prompt type (--text, --roi, or --pose) and at least
    one output format (--bbox, --seg, or --slp for pose mode).

    By default, uses streaming mode which processes frames one at a time for
    memory efficiency. Use --preload to load all frames upfront (slightly
    faster for text prompts, but uses more memory).

    Examples:

        # Track with text prompt, output bounding boxes (streaming mode)
        uv run sam-track track video.mp4 --text "mouse" --bbox

        # Track with ROI file, output both formats
        uv run sam-track track video.mp4 --roi rois.yml --bbox --seg

        # Track with pose file, output tracked SLP
        uv run sam-track track video.mp4 --pose labels.slp --slp

        # Track with pose file, all outputs
        uv run sam-track track video.mp4 --pose labels.slp --bbox --seg --slp

        # Use preload mode for faster text prompt processing
        uv run sam-track track video.mp4 --text "mouse" --bbox --preload
    """
    global _interrupted
    _interrupted = False

    # === Validation ===

    # Check video exists
    if not video.exists():
        console.print(f"[red]Error: Video file not found: {video}[/red]")
        raise typer.Exit(1)

    # Validate mutual exclusivity of prompt options
    prompts = [("--text", text), ("--roi", roi), ("--pose", pose)]
    provided_prompts = [(name, val) for name, val in prompts if val is not None]

    if len(provided_prompts) == 0:
        console.print("[red]Error: Must specify one of --text, --roi, or --pose[/red]")
        raise typer.Exit(1)

    if len(provided_prompts) > 1:
        names = ", ".join(name for name, _ in provided_prompts)
        console.print(
            f"[red]Error: Only one prompt type allowed, but got: {names}[/red]"
        )
        raise typer.Exit(1)

    # Validate at least one output format
    has_bbox = bbox or bbox_output is not None
    has_seg = seg or seg_output is not None
    has_slp = slp_output is not None

    if not has_bbox and not has_seg and not has_slp:
        console.print(
            "[red]Error: Must specify at least one of --bbox, --seg, or --slp[/red]"
        )
        raise typer.Exit(1)

    # Validate --slp only with --pose
    if has_slp and pose is None:
        console.print("[red]Error: --slp is only valid with --pose[/red]")
        raise typer.Exit(1)

    # Validate pose-specific options
    if pose is None:
        if remove_unmatched:
            console.print(
                "[red]Error: --remove-unmatched is only valid with --pose[/red]"
            )
            raise typer.Exit(1)
        if exclude_nodes:
            console.print("[red]Error: --exclude-nodes is only valid with --pose[/red]")
            raise typer.Exit(1)
        if filter_by_pose:
            console.print(
                "[red]Error: --filter-by-pose is only valid with --pose[/red]"
            )
            raise typer.Exit(1)

    # Validate ROI file exists
    if roi is not None and not roi.exists():
        console.print(f"[red]Error: ROI file not found: {roi}[/red]")
        raise typer.Exit(1)

    # Validate pose file exists
    if pose is not None and not pose.exists():
        console.print(f"[red]Error: Pose file not found: {pose}[/red]")
        raise typer.Exit(1)

    # Validate frame range options
    if stop_frame is not None and max_frames is not None:
        console.print(
            "[red]Error: --stop-frame and --max-frames are mutually exclusive[/red]"
        )
        raise typer.Exit(1)

    if start_frame < 0:
        console.print("[red]Error: --start-frame must be non-negative[/red]")
        raise typer.Exit(1)

    if stop_frame is not None and stop_frame <= start_frame:
        console.print(
            "[red]Error: --stop-frame must be greater than --start-frame[/red]"
        )
        raise typer.Exit(1)

    if max_frames is not None and max_frames <= 0:
        console.print("[red]Error: --max-frames must be positive[/red]")
        raise typer.Exit(1)

    # Resolve output paths
    bbox_path = bbox_output or (video.with_suffix(".bbox.json") if has_bbox else None)
    seg_path = seg_output or (video.with_suffix(".seg.h5") if has_seg else None)
    slp_path = slp_output or (
        pose.with_suffix(".sam-tracked.slp") if has_slp and pose else None
    )

    # === Setup ===
    if not quiet:
        _print_config(
            video,
            text,
            roi,
            pose,
            bbox_path,
            seg_path,
            slp_path,
            device,
            start_frame,
            stop_frame,
            max_frames,
            preload,
        )

    # Register signal handler for graceful shutdown
    original_handler = signal.signal(signal.SIGINT, _signal_handler)

    try:
        _run_tracking(
            video=video,
            text=text,
            roi=roi,
            pose=pose,
            bbox_path=bbox_path,
            seg_path=seg_path,
            slp_path=slp_path,
            device=device,
            start_frame=start_frame,
            stop_frame=stop_frame,
            max_frames=max_frames,
            quiet=quiet,
            preload=preload,
            remove_unmatched=remove_unmatched,
            exclude_nodes=exclude_nodes,
            filter_by_pose=filter_by_pose,
        )
    except KeyboardInterrupt:
        # Already handled by signal handler
        pass
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)

    if _interrupted:
        console.print("[yellow]Processing interrupted.[/yellow]")
        raise typer.Exit(130)  # Standard exit code for SIGINT


def _print_config(
    video: Path,
    text: str | None,
    roi: Path | None,
    pose: Path | None,
    bbox_path: Path | None,
    seg_path: Path | None,
    slp_path: Path | None,
    device: str | None,
    start_frame: int,
    stop_frame: int | None,
    max_frames: int | None,
    preload: bool,
) -> None:
    """Print configuration summary."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", justify="right")
    table.add_column(style="white")

    table.add_row("Video:", str(video))

    if text:
        table.add_row("Prompt:", f'text: "{text}"')
    elif roi:
        table.add_row("Prompt:", f"roi: {roi}")
    elif pose:
        table.add_row("Prompt:", f"pose: {pose}")

    if bbox_path:
        table.add_row("BBox output:", str(bbox_path))
    if seg_path:
        table.add_row("Seg output:", str(seg_path))
    if slp_path:
        table.add_row("SLP output:", str(slp_path))

    table.add_row("Mode:", "preload" if preload else "streaming")
    if device:
        table.add_row("Device:", device)

    # Show frame range info
    if start_frame > 0 or stop_frame is not None or max_frames is not None:
        if stop_frame is not None:
            table.add_row("Frame range:", f"{start_frame}:{stop_frame}")
        elif max_frames is not None:
            table.add_row("Frame range:", f"{start_frame}:+{max_frames}")
        else:
            table.add_row("Start frame:", str(start_frame))

    console.print()
    console.print(Panel(table, title="[bold]sam-track[/bold]", border_style="blue"))
    console.print()


def _run_tracking(
    video: Path,
    text: str | None,
    roi: Path | None,
    pose: Path | None,
    bbox_path: Path | None,
    seg_path: Path | None,
    slp_path: Path | None,
    device: str | None,
    start_frame: int,
    stop_frame: int | None,
    max_frames: int | None,
    quiet: bool,
    preload: bool,
    remove_unmatched: bool = False,
    exclude_nodes: str | None = None,
    filter_by_pose: bool = False,
) -> None:
    """Run the tracking pipeline.

    Supports two modes:
    - Streaming (default): Memory-efficient frame-by-frame processing.
    - Preload: Load all frames upfront, slightly faster for text prompts.
    """
    global _interrupted

    from sleap_io import Video

    from .outputs import BBoxWriter, SegmentationWriter, SLPWriter
    from .prompts import (
        PosePromptHandler,
        PromptType,
        ROIPromptHandler,
        TextPromptHandler,
    )
    from .reconciliation import IDReconciler
    from .tracker import SAM3Tracker

    # Load video
    if not quiet:
        console.print("Loading video...")
    vid = Video.from_filename(str(video))
    num_frames = len(vid)
    video_height, video_width = vid.shape[1], vid.shape[2]

    if not quiet:
        console.print(f"  {video_width}x{video_height}, {num_frames} frames")

    # Determine prompt type and load prompt
    if not quiet:
        console.print("Loading prompt...")

    if text:
        prompt_handler = TextPromptHandler(text)
        prompt = prompt_handler.load()
        use_text = True
        prompt_type_str = "text"
        prompt_value_str = text
    elif roi:
        prompt_handler = ROIPromptHandler(roi)
        prompt = prompt_handler.load()
        use_text = False
        prompt_type_str = "roi"
        prompt_value_str = str(roi)
    else:  # pose
        prompt_handler = PosePromptHandler(pose)
        prompt = prompt_handler.load()
        use_text = False
        prompt_type_str = "pose"
        prompt_value_str = str(pose)

    if not quiet:
        if prompt.prompt_type == PromptType.TEXT:
            console.print(f'  Text: "{prompt.text}"')
        else:
            console.print(f"  Objects: {prompt.num_objects}")
            for obj_id in prompt.obj_ids:
                console.print(f"    - {prompt.get_name(obj_id)} (id={obj_id})")

    # Validate start_frame against video length
    if start_frame >= num_frames:
        console.print(
            f"[red]Error: --start-frame ({start_frame}) exceeds video length "
            f"({num_frames} frames)[/red]"
        )
        raise typer.Exit(1)

    # Calculate frame range
    # stop_frame takes precedence if specified (already validated as mutually exclusive)
    if stop_frame is not None:
        # Clamp stop_frame to video length
        actual_stop = min(stop_frame, num_frames)
        frames_to_process = actual_stop - start_frame
    elif max_frames is not None:
        # start_frame + max_frames, clamped to video length
        frames_to_process = min(max_frames, num_frames - start_frame)
    else:
        # Process from start_frame to end
        frames_to_process = num_frames - start_frame

    # Initialize tracker
    if not quiet:
        console.print("Loading SAM3 model...")
    tracker = SAM3Tracker(device=device)

    # Check for multi-frame pose mode
    is_multi_frame_pose = pose is not None and prompt_handler.num_labeled_frames > 1

    if is_multi_frame_pose and not quiet:
        console.print(
            f"  Multi-frame mode: {prompt_handler.num_labeled_frames} labeled frames"
        )

    # Parse exclude_nodes
    excluded_nodes_set = None
    if exclude_nodes:
        excluded_nodes_set = set(n.strip() for n in exclude_nodes.split(","))

    # Initialize output writers
    bbox_writer = None
    seg_writer = None
    slp_writer = None
    reconciler = None

    if bbox_path:
        bbox_writer = BBoxWriter(
            output_path=bbox_path,
            video_path=video,
            video_width=video_width,
            video_height=video_height,
            fps=getattr(vid, "fps", None),
            total_frames=frames_to_process,
            prompt_type=prompt_type_str,
            prompt_value=prompt_value_str,
            obj_names=prompt.obj_names,
        )

    if seg_path:
        seg_writer = SegmentationWriter(
            output_path=seg_path,
            video_path=video,
            video_width=video_width,
            video_height=video_height,
            max_objects=max(10, prompt.num_objects) if not use_text else 10,
            fps=getattr(vid, "fps", None),
            total_frames=frames_to_process,
            obj_names=prompt.obj_names,
        )

    if slp_path and pose is not None:
        import sleap_io as sio

        source_labels = sio.load_slp(str(pose), open_videos=False)
        slp_writer = SLPWriter(
            output_path=slp_path,
            source_labels=source_labels,
            remove_unmatched=remove_unmatched,
        )

        # Initialize reconciler for pose matching
        reconciler = IDReconciler(
            skeleton=prompt_handler.skeleton,
            exclude_nodes=excluded_nodes_set or set(),
        )

    try:
        if preload:
            _run_tracking_preload(
                vid=vid,
                tracker=tracker,
                prompt=prompt,
                prompt_handler=prompt_handler if pose else None,
                use_text=use_text,
                video_height=video_height,
                video_width=video_width,
                start_frame=start_frame,
                frames_to_process=frames_to_process,
                bbox_writer=bbox_writer,
                seg_writer=seg_writer,
                slp_writer=slp_writer,
                reconciler=reconciler,
                filter_by_pose=filter_by_pose,
                quiet=quiet,
            )
        else:
            # Streaming mode (no pose reconciliation - requires preload)
            _run_tracking_streaming(
                vid=vid,
                tracker=tracker,
                prompt=prompt,
                prompt_handler=prompt_handler if pose else None,
                use_text=use_text,
                video_height=video_height,
                video_width=video_width,
                start_frame=start_frame,
                frames_to_process=frames_to_process,
                bbox_writer=bbox_writer,
                seg_writer=seg_writer,
                slp_writer=slp_writer,
                reconciler=reconciler,
                filter_by_pose=filter_by_pose,
                quiet=quiet,
            )

    except Exception as e:
        # Check for CUDA OOM specifically
        import torch

        if isinstance(e, torch.cuda.OutOfMemoryError):
            console.print()
            console.print("[red]Error: GPU out of memory[/red]")
            console.print()
            console.print("Try one of these solutions:")
            console.print("  1. Use --max-frames to process fewer frames")
            console.print("  2. Use a GPU with more memory")
            console.print("  3. Close other GPU applications")
            if preload:
                console.print(
                    "  4. Remove --preload flag (streaming mode uses less memory)"
                )
            raise typer.Exit(1)
        console.print()
        console.print(f"[red]Error during tracking: {e}[/red]")
        raise typer.Exit(1)

    finally:
        tracker.close()
        if seg_writer:
            seg_writer.close()
        # Note: slp_writer doesn't need explicit close - finalize() handles it


def _run_tracking_streaming(
    vid,
    tracker,
    prompt,
    prompt_handler,
    use_text: bool,
    video_height: int,
    video_width: int,
    start_frame: int,
    frames_to_process: int,
    bbox_writer,
    seg_writer,
    slp_writer,
    reconciler,
    filter_by_pose: bool,
    quiet: bool,
) -> None:
    """Run tracking in streaming mode (frame-by-frame, memory-efficient)."""
    global _interrupted

    import numpy as np

    # Initialize streaming session
    if not quiet:
        console.print("Initializing streaming session...")

    tracker.init_streaming_session(
        use_text=use_text,
        num_frames=frames_to_process,
    )

    # Add prompt
    # For visual prompts in streaming mode, we need to pass original_size
    if use_text:
        tracker.add_text_prompt(prompt.text)
    else:
        tracker.add_prompt(prompt, original_size=(video_height, video_width))

    # Run tracking with progress
    if not quiet:
        console.print("Tracking...")
        console.print()

    frames_processed = 0
    total_objects = 0

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        FPSColumn(),
        ObjectsColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=quiet,
    )

    # Get GT frame indices for reconciliation
    gt_frame_indices = set()
    if reconciler and prompt_handler:
        gt_frame_indices = set(prompt_handler.labeled_frame_indices)

    # For filter_by_pose, track which object IDs matched poses
    matched_obj_ids: set[int] = set()

    with progress:
        task = progress.add_task("Processing", total=frames_to_process)

        for i in range(frames_to_process):
            if _interrupted:
                break

            # Actual frame index in the video
            frame_idx = start_frame + i

            # Load single frame
            frame = vid[frame_idx]

            # Ensure RGB format
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.shape[-1] == 4:
                frame = frame[..., :3]

            # Process frame
            result = tracker.process_frame(frame)

            # Update result's frame_idx to actual video frame index
            # (tracker uses internal 0-based counter)
            from .tracker import TrackingResult

            result = TrackingResult(
                frame_idx=frame_idx,
                object_ids=result.object_ids,
                masks=result.masks,
                boxes=result.boxes,
                scores=result.scores,
            )

            # Run reconciliation at GT frames
            if reconciler and frame_idx in gt_frame_indices:
                lf = prompt_handler._build_frame_map().get(frame_idx)
                if lf is not None:
                    import sleap_io as sio

                    gt_instances = [i for i in lf.instances if type(i) is sio.Instance]
                    pred_instances = [
                        i for i in lf.instances if type(i) is sio.PredictedInstance
                    ]
                    instances = gt_instances if gt_instances else pred_instances

                    # Match poses to masks
                    assignments = reconciler.match_frame(
                        frame_idx=frame_idx,
                        poses=instances,
                        masks=result.masks,
                        object_ids=result.object_ids,
                        scores=result.scores,
                    )

                    # Track matched object IDs for filtering
                    for a in assignments:
                        matched_obj_ids.add(a.sam3_obj_id)

                    # Add to SLP writer
                    if slp_writer:
                        slp_writer.add_frame_assignments(
                            frame_idx=frame_idx,
                            assignments=assignments,
                            original_instances=instances,
                        )

            # Update outputs (with optional pose filtering)
            if filter_by_pose and matched_obj_ids:
                mask = np.isin(result.object_ids, list(matched_obj_ids))
                if mask.any():
                    filtered_result = type(result)(
                        frame_idx=result.frame_idx,
                        object_ids=result.object_ids[mask],
                        masks=result.masks[mask],
                        boxes=result.boxes[mask],
                        scores=result.scores[mask],
                    )
                    if bbox_writer:
                        bbox_writer.add_result(filtered_result)
                    if seg_writer:
                        seg_writer.add_result(filtered_result)
            else:
                if bbox_writer:
                    bbox_writer.add_result(result)
                if seg_writer:
                    seg_writer.add_result(result)

            frames_processed += 1
            total_objects = max(total_objects, result.num_objects)

            progress.update(task, advance=1, objects=result.num_objects)

    # Save outputs
    _save_outputs(
        bbox_writer=bbox_writer,
        seg_writer=seg_writer,
        slp_writer=slp_writer,
        reconciler=reconciler,
        frames_processed=frames_processed,
        frames_to_process=frames_to_process,
        total_objects=total_objects,
        quiet=quiet,
    )


def _run_tracking_preload(
    vid,
    tracker,
    prompt,
    prompt_handler,
    use_text: bool,
    video_height: int,
    video_width: int,
    start_frame: int,
    frames_to_process: int,
    bbox_writer,
    seg_writer,
    slp_writer,
    reconciler,
    filter_by_pose: bool,
    quiet: bool,
) -> None:
    """Run tracking in preload mode (all frames loaded upfront)."""
    global _interrupted

    import numpy as np

    # Load video frames from the specified range
    if not quiet:
        console.print("Loading video frames...")

    video_frames = []
    for i in range(frames_to_process):
        if _interrupted:
            break
        # Load from actual video frame index
        frame = vid[start_frame + i]
        # Ensure RGB format
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[-1] == 4:
            frame = frame[..., :3]
        video_frames.append(frame)

    if _interrupted:
        console.print("[yellow]Interrupted during frame loading[/yellow]")
        return

    # Initialize tracking session
    if not quiet:
        console.print("Initializing tracking session...")

    tracker.init_session(
        video_frames=video_frames,
        use_text=use_text,
        video_storage_device="cpu",
        max_vision_cache_size=4,
    )

    # Add prompt
    if use_text:
        tracker.add_text_prompt(prompt.text)
    else:
        tracker.add_prompt(prompt)

    # Run tracking with progress
    if not quiet:
        console.print("Tracking...")
        console.print()

    frames_processed = 0
    total_objects = 0

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        FPSColumn(),
        ObjectsColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=quiet,
    )

    # Get GT frame indices for reconciliation
    gt_frame_indices = set()
    if reconciler and prompt_handler:
        gt_frame_indices = set(prompt_handler.labeled_frame_indices)

    # For filter_by_pose, track which object IDs matched poses
    matched_obj_ids: set[int] = set()

    with progress:
        task = progress.add_task("Processing", total=frames_to_process)

        for result in tracker.propagate():
            if _interrupted:
                break

            # Translate internal frame index to actual video frame index
            frame_idx = start_frame + result.frame_idx

            # Update result with actual video frame index
            from .tracker import TrackingResult

            result = TrackingResult(
                frame_idx=frame_idx,
                object_ids=result.object_ids,
                masks=result.masks,
                boxes=result.boxes,
                scores=result.scores,
            )

            # Run reconciliation at GT frames
            if reconciler and frame_idx in gt_frame_indices:
                lf = prompt_handler._build_frame_map().get(frame_idx)
                if lf is not None:
                    import sleap_io as sio

                    gt_instances = [i for i in lf.instances if type(i) is sio.Instance]
                    pred_instances = [
                        i for i in lf.instances if type(i) is sio.PredictedInstance
                    ]
                    instances = gt_instances if gt_instances else pred_instances

                    # Match poses to masks
                    assignments = reconciler.match_frame(
                        frame_idx=frame_idx,
                        poses=instances,
                        masks=result.masks,
                        object_ids=result.object_ids,
                        scores=result.scores,
                    )

                    # Track matched object IDs for filtering
                    for a in assignments:
                        matched_obj_ids.add(a.sam3_obj_id)

                    # Add to SLP writer
                    if slp_writer:
                        slp_writer.add_frame_assignments(
                            frame_idx=frame_idx,
                            assignments=assignments,
                            original_instances=instances,
                        )

            # Update outputs (with optional pose filtering)
            if filter_by_pose and matched_obj_ids:
                mask = np.isin(result.object_ids, list(matched_obj_ids))
                if mask.any():
                    filtered_result = type(result)(
                        frame_idx=result.frame_idx,
                        object_ids=result.object_ids[mask],
                        masks=result.masks[mask],
                        boxes=result.boxes[mask],
                        scores=result.scores[mask],
                    )
                    if bbox_writer:
                        bbox_writer.add_result(filtered_result)
                    if seg_writer:
                        seg_writer.add_result(filtered_result)
            else:
                if bbox_writer:
                    bbox_writer.add_result(result)
                if seg_writer:
                    seg_writer.add_result(result)

            frames_processed += 1
            total_objects = max(total_objects, result.num_objects)

            progress.update(task, advance=1, objects=result.num_objects)

    # Save outputs
    _save_outputs(
        bbox_writer=bbox_writer,
        seg_writer=seg_writer,
        slp_writer=slp_writer,
        reconciler=reconciler,
        frames_processed=frames_processed,
        frames_to_process=frames_to_process,
        total_objects=total_objects,
        quiet=quiet,
    )


def _save_outputs(
    bbox_writer,
    seg_writer,
    slp_writer,
    reconciler,
    frames_processed: int,
    frames_to_process: int,
    total_objects: int,
    quiet: bool,
) -> None:
    """Save tracking outputs and print summary."""
    global _interrupted

    if not quiet:
        console.print()

    # Apply track name resolution from reconciler to writers
    if reconciler:
        from .reconciliation import TrackNameResolver

        resolver = TrackNameResolver.from_reconciler(reconciler)
        canonical_mapping = resolver.get_canonical_mapping()

        if canonical_mapping:
            if bbox_writer:
                bbox_writer.apply_track_name_mapping(canonical_mapping)
            if seg_writer:
                seg_writer.apply_track_name_mapping(canonical_mapping)
            if slp_writer:
                slp_writer.apply_track_name_mapping(canonical_mapping)

    if bbox_writer:
        bbox_writer.save()
        if not quiet:
            console.print(
                f"[green]Saved bounding boxes:[/green] {bbox_writer.output_path}"
            )

    if seg_writer:
        seg_writer.finalize()
        if not quiet:
            console.print(
                f"[green]Saved segmentation masks:[/green] {seg_writer.output_path}"
            )

    if slp_writer:
        slp_writer.finalize()
        if not quiet:
            console.print(
                f"[green]Saved tracked poses:[/green] {slp_writer.output_path}"
            )
            if reconciler:
                swaps = reconciler.detect_swaps()
                if swaps:
                    console.print(
                        f"  [yellow]Warning: {len(swaps)} identity swap(s) "
                        f"detected[/yellow]"
                    )

    # Summary
    if not quiet:
        console.print()
        if _interrupted:
            console.print(
                f"[yellow]Processed {frames_processed}/{frames_to_process} frames "
                f"(interrupted)[/yellow]"
            )
        else:
            console.print(
                f"[green]Processed {frames_processed} frames, "
                f"{total_objects} objects tracked[/green]"
            )


@app.command()
def auth(
    token: str = typer.Option(
        None,
        "--token",
        "-t",
        help="Login with a HuggingFace token.",
    ),
) -> None:
    """Check HuggingFace authentication status for SAM3 model access.

    SAM3 is a gated model that requires:
    1. A HuggingFace account
    2. Accepting the model license at https://huggingface.co/facebook/sam3

    Use --token to login, or run 'uvx hf auth login' for interactive authentication.
    """
    from huggingface_hub import login as hf_login

    from .auth import (
        SAM3_REPO_ID,
        check_authentication,
        check_model_access,
        get_username,
    )

    # Handle login request first
    if token:
        console.print()
        console.print("[bold]HuggingFace Login[/bold]")
        try:
            hf_login(token=token)
            console.print("[green]Successfully logged in with provided token.[/green]")
            console.print()
        except Exception as e:
            console.print(f"[red]Login failed: {e}[/red]")
            raise typer.Exit(1)

    # Check authentication status
    is_authenticated = check_authentication()
    username = get_username() if is_authenticated else None
    has_model_access = check_model_access()

    # Build status table
    table = Table(title="HuggingFace Authentication Status", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    # Authentication status
    if is_authenticated:
        table.add_row("Authenticated", "[green]Yes[/green]")
        table.add_row("Username", username or "Unknown")
    else:
        table.add_row("Authenticated", "[red]No[/red]")

    # Model access status
    table.add_row("SAM3 model", SAM3_REPO_ID)
    if has_model_access:
        table.add_row("Model access", "[green]Granted[/green]")
    else:
        table.add_row("Model access", "[red]Not granted[/red]")

    console.print()
    console.print(table)

    # Provide actionable guidance
    issues_found = False

    if not is_authenticated:
        issues_found = True
        console.print()
        console.print("[red]⚠ Not authenticated with HuggingFace Hub[/red]")
        console.print()
        console.print("  [bold]Step 1:[/bold] Create a token at:")
        console.print(
            "    [link=https://huggingface.co/settings/tokens]"
            "https://huggingface.co/settings/tokens[/link]"
        )
        console.print()
        console.print("    - Click [bold]Create new token[/bold]")
        console.print("    - Name it: [cyan]sam-track[/cyan]")
        console.print(
            "    - Select [bold]Read[/bold] permission (top tab, not fine-grained)"
        )
        console.print()
        console.print("  [bold]Step 2:[/bold] Login with your token:")
        console.print("    [cyan]uv run sam-track auth --token hf_...[/cyan]")
        console.print()
        console.print("  Alternatives:")
        console.print("    [cyan]uvx hf auth login[/cyan]  (interactive)")
        console.print("    [cyan]export HF_TOKEN=hf_...[/cyan]  (env var)")

    if is_authenticated and not has_model_access:
        issues_found = True
        console.print()
        console.print("[red]⚠ No access to SAM3 model[/red]")
        console.print()
        console.print("  SAM3 is a gated model that requires accepting Meta's license.")
        console.print()
        console.print(
            f"  Request access at: [link=https://huggingface.co/{SAM3_REPO_ID}]"
            f"https://huggingface.co/{SAM3_REPO_ID}[/link]"
        )
        console.print()
        console.print(
            "  After approval, run [cyan]uv run sam-track auth[/cyan] again to verify."
        )

    if not issues_found:
        console.print()
        console.print("[green]✓[/green] Ready to use SAM3!")
        console.print()
        console.print(
            '  Run [cyan]uv run sam-track track <video> --text "object"[/cyan] '
            "to get started."
        )
    else:
        raise typer.Exit(1)


@app.command()
def system() -> None:
    """Display system information and GPU status."""
    import torch

    # System info table
    table = Table(title="System Information", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("sam-track version", __version__)
    table.add_row("Python version", sys.version.split()[0])
    table.add_row("Platform", platform.platform())
    table.add_row("PyTorch version", torch.__version__)
    table.add_row("CUDA available", str(torch.cuda.is_available()))

    # Check driver version - do this even if CUDA is not available
    # because an old driver can cause CUDA to appear unavailable
    driver_version = get_nvidia_driver_version()
    driver_warning = None

    if driver_version:
        is_compatible, min_version = check_driver_version(driver_version)
        if is_compatible:
            table.add_row("Driver version", f"{driver_version}")
        else:
            table.add_row("Driver version", f"[red]{driver_version}[/red]")
            driver_warning = (driver_version, min_version)

    if torch.cuda.is_available():
        table.add_row("CUDA version", torch.version.cuda or "N/A")
        table.add_row("cuDNN version", str(torch.backends.cudnn.version()))
        table.add_row("GPU count", str(torch.cuda.device_count()))

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        table.add_row("MPS available", "True")

    console.print(table)

    # Driver version warning
    if driver_warning:
        current, minimum = driver_warning
        console.print()
        console.print(
            f"[red]⚠ Driver version {current} is below the minimum required "
            f"({minimum}) for CUDA 13.0.[/red]"
        )
        console.print(
            "[yellow]  Please update your NVIDIA driver: "
            "https://www.nvidia.com/drivers[/yellow]"
        )
        if not torch.cuda.is_available():
            console.print(
                "[yellow]  This is likely why CUDA is not available.[/yellow]"
            )

    # GPU details
    if torch.cuda.is_available():
        console.print()
        gpu_table = Table(title="GPU Details")
        gpu_table.add_column("ID", style="cyan")
        gpu_table.add_column("Name", style="white")
        gpu_table.add_column("Compute Cap.", style="green")
        gpu_table.add_column("Memory", style="yellow")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            gpu_table.add_row(
                str(i),
                props.name,
                f"{props.major}.{props.minor}",
                f"{memory_gb:.1f} GB",
            )

        console.print(gpu_table)

        # Quick functionality test
        console.print()
        try:
            x = torch.randn(100, 100, device="cuda")
            y = torch.randn(100, 100, device="cuda")
            _ = torch.mm(x, y)
            console.print("[green]✓[/green] CUDA tensor operations working")
        except Exception as e:
            console.print(f"[red]✗[/red] CUDA tensor operations failed: {e}")

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        console.print()
        try:
            x = torch.randn(100, 100, device="mps")
            y = torch.randn(100, 100, device="mps")
            _ = torch.mm(x, y)
            console.print("[green]✓[/green] MPS tensor operations working")
        except Exception as e:
            console.print(f"[red]✗[/red] MPS tensor operations failed: {e}")

    else:
        console.print()
        console.print("[yellow]![/yellow] No GPU acceleration available, using CPU")


if __name__ == "__main__":
    app()
