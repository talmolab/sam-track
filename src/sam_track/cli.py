"""sam-track CLI using Typer."""

import typer
from rich.console import Console

from . import __version__

app = typer.Typer(
    name="sam-track",
    help="Track objects in videos using SAM3.",
    add_completion=False,
)
console = Console()


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
    console.print(f"[yellow]sam-track is not yet implemented.[/yellow]")
    console.print(f"Would track: {video}")


if __name__ == "__main__":
    app()
