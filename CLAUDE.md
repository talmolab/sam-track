# CLAUDE.md

Development guidance for `sam-track`.

## Key Documents

- `DESIGN.md` - Project requirements and CLI interface spec
- `ROADMAP.md` - Implementation phases and task checklist

## Development Workflow

1. Use `uv` for all Python operations
2. Run with `uv run sam-track` or `uv run python -m sam_track`
3. Sync dependencies with `uv sync`
4. Add dependencies with `uv add <package>`

## Code Style

- Format with `black` (88 char line length)
- Lint with `ruff`
- Google-style docstrings
- Type hints on all public functions

## Testing

- Use `pytest` for all tests
- Run tests with `uv run pytest`
- Place tests in `tests/` mirroring `src/` structure
- Use `tmp_path` fixture for file I/O tests

## Git Practices

- Create feature branches from `main`
- Squash merge PRs
- Use conventional commit messages

## HuggingFace Authentication

SAM3 is a gated model. For development:
```bash
# One-time setup
uv run huggingface-cli login
# Or set environment variable
export HF_TOKEN="hf_xxxxx"
```

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
```
