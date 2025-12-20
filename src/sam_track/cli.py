"""sam-track CLI using Typer."""

import platform
import shutil
import subprocess
import sys

import typer
from rich.console import Console
from rich.table import Table

from . import __version__

app = typer.Typer(
    name="sam-track",
    help="Track objects in videos using SAM3.",
    add_completion=False,
)
console = Console()


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
    video: str = typer.Argument(..., help="Path to input video file."),
) -> None:
    """Track objects in a video using SAM3."""
    console.print("[yellow]sam-track is not yet implemented.[/yellow]")
    console.print(f"Would track: {video}")


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

    if torch.cuda.is_available():
        driver_version = get_nvidia_driver_version()
        if driver_version:
            table.add_row("Driver version", driver_version)
        table.add_row("CUDA version", torch.version.cuda or "N/A")
        table.add_row("cuDNN version", str(torch.backends.cudnn.version()))
        table.add_row("GPU count", str(torch.cuda.device_count()))

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        table.add_row("MPS available", "True")

    console.print(table)

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
