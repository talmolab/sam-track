"""HuggingFace Hub authentication utilities."""

import os
import sys

from huggingface_hub import auth_check, login, whoami
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from rich.console import Console

console = Console()

# SAM3 model repository on HuggingFace Hub
SAM3_REPO_ID = "facebook/sam3"


def check_authentication() -> bool:
    """Check if user is authenticated with HuggingFace Hub.

    Returns:
        True if authenticated, False otherwise.
    """
    try:
        whoami()
        return True
    except Exception:
        return False


def check_model_access(repo_id: str = SAM3_REPO_ID) -> bool:
    """Check if user has access to the specified model.

    Args:
        repo_id: HuggingFace model repository ID.

    Returns:
        True if user has access, False otherwise.
    """
    try:
        auth_check(repo_id)
        return True
    except GatedRepoError:
        return False
    except RepositoryNotFoundError:
        return False
    except Exception:
        return False


def get_username() -> str | None:
    """Get the authenticated user's name.

    Returns:
        Username if authenticated, None otherwise.
    """
    try:
        user = whoami()
        return user.get("name") or user.get("fullname")
    except Exception:
        return None


def ensure_authenticated(
    repo_id: str = SAM3_REPO_ID,
    interactive: bool = True,
) -> None:
    """Ensure user is authenticated and has access to the model.

    This function checks for authentication in the following order:
    1. HF_TOKEN environment variable
    2. Saved token in ~/.cache/huggingface/token
    3. Interactive login prompt (if interactive=True)

    Args:
        repo_id: HuggingFace model repository ID to check access for.
        interactive: If True, prompt for login if not authenticated.

    Raises:
        SystemExit: If authentication fails or user lacks model access.
    """
    # Check for environment variable first
    if os.environ.get("HF_TOKEN"):
        console.print("[green]Using HF_TOKEN from environment[/green]")
        if check_model_access(repo_id):
            return
        console.print(
            f"\n[red]Your token does not have access to {repo_id}.[/red]\n"
            f"Request access at: https://huggingface.co/{repo_id}"
        )
        sys.exit(1)

    # Check saved token
    if check_authentication():
        username = get_username()
        if check_model_access(repo_id):
            console.print(f"[green]Authenticated as {username}[/green]")
            return
        console.print(
            f"\n[yellow]Authenticated as {username}, "
            f"but no access to {repo_id}.[/yellow]\n"
            f"Request access at: https://huggingface.co/{repo_id}"
        )
        if not interactive:
            sys.exit(1)

    if not interactive:
        console.print(
            "[red]Not authenticated with HuggingFace Hub.[/red]\n"
            "Set HF_TOKEN environment variable or run:\n"
            "  uv run huggingface-cli login"
        )
        sys.exit(1)

    # Prompt for login
    console.print(
        "\n[yellow]SAM3 requires authentication with HuggingFace Hub.[/yellow]\n"
        f"Get your token from: https://huggingface.co/settings/tokens\n"
        f"Then request access at: https://huggingface.co/{repo_id}\n"
    )

    try:
        login()
    except Exception as e:
        console.print(f"[red]Login failed: {e}[/red]")
        sys.exit(1)

    # Verify access after login
    if not check_model_access(repo_id):
        console.print(
            f"\n[red]You don't have access to {repo_id}.[/red]\n"
            f"Request access at: https://huggingface.co/{repo_id}"
        )
        sys.exit(1)

    username = get_username()
    console.print(f"[green]Successfully authenticated as {username}[/green]")
