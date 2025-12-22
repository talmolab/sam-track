# Contributing to sam-track

Thank you for your interest in contributing to sam-track!

## Development Setup

### Prerequisites

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- NVIDIA GPU with driver 580.65+ (Linux) or 580.65 (Windows) for CUDA 13.0
- Or Apple Silicon Mac for MPS support

### Clone and Install

```bash
git clone https://github.com/talmolab/sam-track
cd sam-track
uv sync
```

This installs all dependencies including dev tools.

### HuggingFace Authentication

SAM3 is a gated model. Set up authentication once:

```bash
# Interactive login
uv run huggingface-cli login

# Or set environment variable
export HF_TOKEN="hf_xxxxx"
```

Then request access at https://huggingface.co/facebook/sam3

## Development Workflow

### Running the CLI

```bash
# Run directly with uv
uv run sam-track --help
uv run sam-track track video.mp4 --text "mouse" --bbox

# Or as a module
uv run python -m sam_track
```

### Code Style

We use:
- **black** for formatting (88 char line length)
- **ruff** for linting
- **mypy** for type checking
- **Google-style docstrings**

Format and lint before committing:

```bash
# Format code
uv run black src/ tests/

# Lint and auto-fix
uv run ruff check src/ tests/ --fix

# Type check
uv run mypy src/
```

### Testing

We use pytest for all tests:

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_cli.py

# Run with coverage
uv run pytest --cov=sam_track --cov-report=term-missing
```

Tests are in `tests/` mirroring the `src/sam_track/` structure. Use the `tmp_path` fixture for any file I/O tests.

### Adding Dependencies

```bash
# Add a runtime dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Sync after changes
uv sync
```

## Git Practices

### Branching

- Create feature branches from `main`
- Use descriptive branch names: `feat/feature-name`, `fix/bug-description`

### Commits

Use conventional commit messages:

```
feat: Add new feature
fix: Fix bug in X
docs: Update README
test: Add tests for Y
refactor: Restructure Z
```

### Pull Requests

- Squash merge PRs to main
- Include a clear description of changes
- Reference any related issues

## Project Structure

```
sam-track/
├── src/sam_track/          # Main package
│   ├── __init__.py         # Version and exports
│   ├── cli.py              # CLI with typer
│   ├── auth.py             # HuggingFace auth
│   ├── tracker.py          # SAM3Tracker wrapper
│   ├── reconciliation.py   # Pose-mask matching
│   ├── prompts/            # Prompt handlers
│   │   ├── base.py         # Prompt, PromptHandler ABC
│   │   ├── text.py         # TextPromptHandler
│   │   ├── roi.py          # ROIPromptHandler
│   │   └── pose.py         # PosePromptHandler
│   └── outputs/            # Output writers
│       ├── bbox.py         # BBoxWriter (JSON)
│       ├── seg.py          # SegmentationWriter (HDF5)
│       └── slp.py          # SLPWriter (SLEAP)
├── tests/                  # Test suite
│   ├── data/               # Test fixtures
│   ├── prompts/            # Prompt handler tests
│   └── outputs/            # Output writer tests
├── pyproject.toml          # Project config
├── ROADMAP.md              # Implementation status
└── CLAUDE.md               # AI assistant guidance
```

## Key Documents

- **ROADMAP.md** - Implementation phases and task checklist
- **DESIGN.md** - Project requirements and CLI interface spec

## Common Commands

```bash
# Install and sync
uv sync

# Run CLI
uv run sam-track --help

# Run tests
uv run pytest

# Format code
uv run black src/ tests/
uv run ruff check src/ tests/ --fix

# Type check
uv run mypy src/

# Check system/GPU
uv run sam-track system

# Check HuggingFace auth
uv run sam-track auth
```

## Questions?

Open an issue on GitHub if you have questions or run into problems.
