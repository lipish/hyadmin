# Branch Deletion Instructions

## Overview

This PR addresses the issue of removing unused branches from the repository. Due to GitHub authentication requirements, the automated agent cannot directly delete remote branches. However, this PR provides all necessary tools and documentation for a repository administrator to complete the task.

## What This PR Provides

1. **Automated Deletion Script** (`delete-unused-branches.sh`)
   - Interactive script that safely deletes all unused branches
   - Includes confirmation prompt before deletion
   - Provides clear feedback on success/failure for each branch
   - Verifies remaining branches after deletion

2. **Updated Documentation** (`UNUSED_BRANCHES.md`)
   - Complete list of 5 branches to be deleted with rationale for each
   - Three different deletion methods (script, CLI, or GitHub UI)
   - Verification instructions

3. **This README** - Step-by-step guide for completing the task

## Branches to be Deleted

The following 5 remote branches will be deleted:

1. **copilot/explain-git-clone-files** - Merged in PR #7
2. **copilot/git-history-cleanup-docs** - Same commit as #1, merged
3. **copilot/simplify-repo-file-structure** - Merged in PR #6
4. **copilot/move-api-web-management** - Abandoned initial plan only
5. **copilot/remove-unused-branches** - Previous attempt at this task

## How to Complete This Task

### Recommended: Use the Provided Script

```bash
# 1. Clone the repository (if not already cloned)
git clone https://github.com/lipish/hyadmin.git
cd hyadmin

# 2. Checkout this PR branch
git checkout copilot/remove-unused-branches-again

# 3. Run the deletion script
./delete-unused-branches.sh

# 4. Type 'yes' when prompted to confirm deletion

# 5. Verify the branches are deleted
git ls-remote --heads origin
```

### Alternative: Manual Deletion

If you prefer to delete branches manually:

```bash
git push origin --delete \
  copilot/explain-git-clone-files \
  copilot/git-history-cleanup-docs \
  copilot/simplify-repo-file-structure \
  copilot/move-api-web-management \
  copilot/remove-unused-branches
```

### Alternative: Use GitHub Web Interface

1. Visit https://github.com/lipish/hyadmin/branches
2. Find each branch listed above
3. Click the trash/delete icon next to each branch

## After Deletion

1. Verify only expected branches remain:
   ```bash
   git ls-remote --heads origin
   ```
   
   Should show:
   - `refs/heads/main`
   - `refs/heads/copilot/remove-unused-branches-again`

2. Merge this PR to main

3. Delete the `copilot/remove-unused-branches-again` branch

## Why Automated Deletion Wasn't Possible

The automated agent operates in a sandboxed environment with limited GitHub credentials that allow:
- ✅ Reading repository data
- ✅ Creating commits and pushing to PR branches (via report_progress tool)
- ❌ Direct branch deletion via git push --delete
- ❌ Administrative operations via GitHub API

This is a security measure to prevent unauthorized changes to repository structure.

## Questions?

If you encounter any issues or have questions about this process, please comment on this PR.
