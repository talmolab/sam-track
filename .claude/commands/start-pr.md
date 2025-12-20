# Start a Pull Request

Create a new branch, commit current changes, push, and open a pull request.

## Instructions

1. **Determine branch name**: Based on the current session context (what we've been working on), suggest an appropriate branch name following the pattern `<type>/<short-description>` where type is one of: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`. If the context is unclear, ask the user what the branch should be named.

2. **Check git status**: Run `git status` to see what changes exist. If there are no changes to commit, inform the user and stop.

3. **Create and switch to branch**: Create a new branch with the determined name and switch to it.

4. **Stage changes**: Stage all relevant changes. Be thoughtful about what to include - skip any files that look like they shouldn't be committed (secrets, large binaries, etc.).

5. **Commit**: Create a commit with a clear, descriptive message summarizing all the changes. Follow conventional commit format. End the commit message with the standard Claude Code attribution.

6. **Push**: Push the branch to origin with `-u` to set upstream tracking.

7. **Create PR**: Use `gh pr create` to open a pull request. The PR should have:
   - A clear title summarizing the changes
   - A body with:
     - `## Summary` section with bullet points describing what changed
     - `## Test plan` section (if applicable) with how to verify the changes
     - The Claude Code attribution footer

8. **Report**: Share the PR URL with the user when complete.

## Important

- Do NOT force push or use destructive git commands
- Do NOT commit files that look like secrets or credentials
- If the repo has uncommitted changes on main, ask before proceeding
- If a branch with the same name already exists, ask the user what to do
