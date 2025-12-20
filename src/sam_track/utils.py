"""Common utility functions for sam-track."""

from pathlib import Path

import torch


def get_device(device: str | None = None) -> torch.device:
    """Get the best available device for inference.

    Args:
        device: Explicit device string (e.g., "cuda:0", "cpu", "mps").
            If None, automatically selects the best available device.

    Returns:
        torch.device object for the selected device.

    Example:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device("cuda:1")  # Specific GPU
    """
    if device is not None:
        return torch.device(device)

    # Check for CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # Fallback to CPU
    return torch.device("cpu")


def get_device_info(device: torch.device | None = None) -> dict:
    """Get information about the specified device.

    Args:
        device: Device to query. If None, uses the default device.

    Returns:
        Dictionary with device information including:
            - type: Device type string
            - name: Device name (for GPUs)
            - memory_total: Total memory in GB (for GPUs)
            - memory_free: Free memory in GB (for GPUs)
    """
    if device is None:
        device = get_device()

    info = {"type": device.type, "device": str(device)}

    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        info["name"] = props.name
        info["memory_total"] = props.total_memory / (1024**3)
        info["memory_free"] = (
            props.total_memory - torch.cuda.memory_allocated(idx)
        ) / (1024**3)
        info["compute_capability"] = f"{props.major}.{props.minor}"
    elif device.type == "mps":
        info["name"] = "Apple Silicon GPU"
    else:
        info["name"] = "CPU"

    return info


def resolve_output_path(
    video_path: Path,
    output_path: str | Path | None,
    suffix: str,
    extension: str,
) -> Path:
    """Resolve output path based on video path and user input.

    If output_path is provided, uses that. Otherwise, generates a path
    based on the video path with the given suffix and extension.

    Args:
        video_path: Path to the source video file.
        output_path: User-specified output path, or None for auto.
        suffix: Suffix to add before extension (e.g., "bbox", "seg").
        extension: File extension including dot (e.g., ".json", ".h5").

    Returns:
        Resolved output Path.

    Example:
        >>> resolve_output_path(Path("video.mp4"), None, "bbox", ".json")
        Path("video.bbox.json")
        >>> resolve_output_path(Path("video.mp4"), "out.json", "bbox", ".json")
        Path("out.json")
    """
    if output_path is not None:
        return Path(output_path)

    video_path = Path(video_path)
    return video_path.parent / f"{video_path.stem}.{suffix}{extension}"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like "1h 23m 45s" or "5m 30s" or "45s".
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)


def format_size(size_bytes: int) -> str:
    """Format file size in bytes to human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string like "1.5 GB" or "256 MB".
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"
