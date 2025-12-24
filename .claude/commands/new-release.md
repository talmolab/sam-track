# Create a New Release

Scaffold a new version release: document changes, bump version, create PR, merge, and publish GitHub release.

## Arguments

`$ARGUMENTS` should be the new version number (e.g., `0.1.2`). If not provided, suggest the next patch version based on the current version.

## Instructions

### Phase 1: Discover Changes

1. **Find last release**: Run `gh release list --limit 1` and `git tag -l 'v*' --sort=-version:refname | head -1` to find the most recent release tag.

2. **Get current version**: Read `pyproject.toml` to find the current version.

3. **Determine new version**: If `$ARGUMENTS` is provided, use that. Otherwise, suggest incrementing the patch version (e.g., `0.1.0` → `0.1.1`).

4. **List changes**: Run `git log <last-tag>..HEAD --oneline` to see all commits since the last release.

5. **Get PR details**: For each merged PR (identified by `(#N)` in commit messages), use `gh pr view N --json title,body,files` to get full details.

### Phase 2: Document Changes

6. **Create investigation folder**: Create `scratch/YYYY-MM-DD-vX.Y.Z-release/README.md` documenting:
   - All PRs and their summaries
   - Categorized changes (New Features, Bug Fixes, Documentation, etc.)
   - Draft release notes

### Phase 3: Version Bump

7. **Ensure on main**: Run `git checkout main && git pull` to ensure we're on the latest main.

8. **Create release branch**: Create branch `release/vX.Y.Z`.

9. **Update version files**: Update version in:
   - `pyproject.toml` (line with `version = "..."`)
   - `src/sam_track/__init__.py` (line with `__version__ = "..."`)

10. **Commit**: Create commit with message `chore: Bump version to X.Y.Z`.

### Phase 4: Create and Merge PR

11. **Push branch**: Run `git push -u origin release/vX.Y.Z`.

12. **Create PR**: Use `gh pr create` with:
    - Title: `chore: Release vX.Y.Z`
    - Body: Summary of all changes since last release (from Phase 2)

13. **Merge PR**: Run `gh pr merge --squash --delete-branch`.

14. **Update local**: Run `git checkout main && git pull`.

### Phase 5: Publish Release

15. **Create GitHub release**: Use `gh release create vX.Y.Z` with:
    - Title: `sam-track vX.Y.Z`
    - Release notes containing:
      - `## What's Changed` section with categorized bullet points
      - Links to PRs (e.g., `(#14)`)
      - Full changelog link: `**Full Changelog**: https://github.com/talmolab/sam-track/compare/vOLD...vNEW`

16. **Verify release**: Run `gh release view vX.Y.Z` to confirm the release was created.

17. **Report**: Share the release URL with the user.

## Release Notes Format

```markdown
## What's Changed

### New Features
- Feature description (#PR)

### Bug Fixes
- Fix description (#PR)

### Documentation
- Doc update description (#PR)

**Full Changelog**: https://github.com/talmolab/sam-track/compare/vOLD...vNEW
```

## Important Safety Rules

- NEVER skip version bump in both `pyproject.toml` AND `__init__.py`
- NEVER create a release tag that already exists
- NEVER force push or use destructive git commands
- If there are no changes since the last release, inform the user and stop
- If CI fails, fix issues before proceeding with the release
- Always verify the release was created successfully before finishing

## Useful Commands Reference

```bash
# Find releases and tags
gh release list --limit 5
git tag -l 'v*' --sort=-version:refname

# Get commits since last release
git log v0.1.0..HEAD --oneline
git log v0.1.0..HEAD --format='%H %s'

# Get PR details
gh pr view 14 --json title,body,files

# Create release
gh release create v0.1.1 --title "sam-track v0.1.1" --notes "Release notes here"

# View release
gh release view v0.1.1
```
