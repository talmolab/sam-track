# Finish and Merge a Pull Request

Finalize the current PR: commit remaining changes, ensure CI passes, update PR description, and squash merge.

## Instructions

### Phase 1: Commit and Push

1. **Check current branch**: Ensure we're on a feature branch, not `main`. If on `main`, stop and ask the user which PR to finish.

2. **Commit pending changes**: Run `git status`. If there are uncommitted changes:
   - Stage relevant changes
   - Create a commit with a clear message describing what was added/fixed
   - Use standard Claude Code attribution

3. **Push**: Push changes to the remote branch.

### Phase 2: Wait for CI

4. **Check CI status**: Use `gh pr checks` or `gh run list` to check the status of CI workflows.

5. **Poll for completion**: If CI is still running, launch a background agent to poll every 30 seconds until CI completes. Use the Task tool with `run_in_background: true` to monitor, then retrieve results with TaskOutput.

6. **Handle CI failure**: If CI fails:
   - Fetch the logs using `gh run view <run-id> --log-failed`
   - Analyze the failure and fix the issues
   - Commit and push the fix
   - Return to step 4 (poll for CI again)
   - Repeat until CI passes or ask user for help after 3 attempts

### Phase 3: Update PR

7. **Review PR description**: Once CI is green, use `gh pr view` to read the current PR title and body.

8. **Update if needed**: Compare the PR description against actual changes in the branch (`git log main..HEAD`, `git diff main...HEAD --stat`). If the description is outdated or inaccurate:
   - Use `gh pr edit` to update the title and/or body
   - Ensure the summary accurately reflects all commits

### Phase 4: Merge Main and Resolve Conflicts

9. **Check if up to date**: Run `git fetch origin main` and check if the branch is behind main.

10. **Merge main**: If behind, merge main into the feature branch:
    ```bash
    git merge origin/main
    ```

11. **Resolve conflicts**: If there are merge conflicts:
    - List conflicted files
    - Resolve each conflict carefully, preserving both the feature changes and main updates
    - Stage resolved files and complete the merge commit
    - Push the merge commit

12. **Wait for CI again**: If changes were made (merge commit), return to Phase 2 to wait for CI to pass again.

### Phase 5: Squash Merge

13. **Final merge**: Once CI is green and branch is up to date, squash merge the PR:
    ```bash
    gh pr merge --squash --delete-branch
    ```

14. **Report**: Confirm the PR was merged and provide the merge commit info.

## Important Safety Rules

- NEVER force push or use `--force` flags
- NEVER merge to main without CI passing
- NEVER delete branches before merge is complete
- If conflicts are complex or unclear, ASK the user before resolving
- If CI fails more than 3 times, stop and ask the user for guidance
- Always preserve the intent of both the feature branch and main when resolving conflicts

## Useful Commands Reference

```bash
# Check PR status
gh pr status
gh pr view

# Check CI status
gh pr checks
gh run list --branch <branch>
gh run view <run-id>
gh run view <run-id> --log-failed

# Update PR
gh pr edit --title "New Title" --body "New body"

# Merge
gh pr merge --squash --delete-branch
```
