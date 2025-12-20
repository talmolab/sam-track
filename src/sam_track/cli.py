"""sam-track CLI using Typer."""

import platform
import shutil
import subprocess
import sys

import typer
from rich.console import Console
from rich.table import Table

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
    video: str = typer.Argument(..., help="Path to input video file."),
) -> None:
    """Track objects in a video using SAM3."""
    console.print("[yellow]sam-track is not yet implemented.[/yellow]")
    console.print(f"Would track: {video}")


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
    from .auth import (
        SAM3_REPO_ID,
        check_authentication,
        check_model_access,
        get_username,
    )
    from huggingface_hub import login as hf_login

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
        console.print("    - Select [bold]Read[/bold] permission (top tab, not fine-grained)")
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
        console.print("  After approval, run [cyan]uv run sam-track auth[/cyan] again to verify.")

    if not issues_found:
        console.print()
        console.print("[green]✓[/green] Ready to use SAM3!")
        console.print()
        console.print("  Run [cyan]uv run sam-track track <video> --text \"object\"[/cyan] to get started.")
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
