# Unused Branches Cleanup

## Status: Ready for Deletion ⏳

This PR provides tools and documentation for removing unused branches from the repository. Due to authentication requirements, the actual branch deletion must be performed by a repository administrator with push permissions.

A helper script (`delete-unused-branches.sh`) has been provided to simplify this process.

---

This document lists the branches that were identified as no longer needed and provides instructions for removing them.

## ⚠️ Action Required: Delete Remote Branches

The following remote branches still exist on GitHub and need to be deleted by a repository administrator with appropriate permissions:

### 1. copilot/explain-git-clone-files
- **Status**: Merged into main via PR #7
- **Commit**: 8e10e030
- **Reason**: This branch was successfully merged and is no longer needed
- **Command**: `git push origin --delete copilot/explain-git-clone-files`

### 2. copilot/git-history-cleanup-docs
- **Status**: Points to same commit as copilot/explain-git-clone-files (merged)
- **Commit**: 8e10e030  
- **Reason**: This branch points to the same commit that was merged in PR #7
- **Command**: `git push origin --delete copilot/git-history-cleanup-docs`

### 3. copilot/simplify-repo-file-structure
- **Status**: Merged into main via PR #6
- **Commit**: 29f16fab
- **Reason**: This branch was successfully merged and is no longer needed
- **Command**: `git push origin --delete copilot/simplify-repo-file-structure`

### 4. copilot/move-api-web-management
- **Status**: Abandoned work (only contains initial plan commit)
- **Commit**: 2b0bac3c
- **Reason**: This branch only has an "Initial plan" commit and was never completed or merged
- **Command**: `git push origin --delete copilot/move-api-web-management`

### 5. copilot/remove-unused-branches
- **Status**: Previous attempt at this same cleanup task
- **Commit**: 1e34dd7a
- **Reason**: This branch was for the same task as copilot/remove-unused-branches-again and is no longer needed
- **Command**: `git push origin --delete copilot/remove-unused-branches`

## Local Cleanup Completed

The following local tracking branches have been deleted from this repository:
- ✅ copilot/explain-git-clone-files
- ✅ copilot/git-history-cleanup-docs
- ✅ copilot/simplify-repo-file-structure
- ✅ copilot/move-api-web-management
- ✅ copilot/remove-unused-branches

## Branches Kept

- **main**: The primary branch
- **copilot/remove-unused-branches-again**: Current working branch for this cleanup task (can be deleted after this PR is merged)

## How to Delete Remote Branches

### Option 1: Using the Provided Script (Recommended)

A helper script `delete-unused-branches.sh` has been provided in the repository root. To use it:

```bash
cd /path/to/hyadmin
./delete-unused-branches.sh
```

The script will:
1. List all branches to be deleted
2. Ask for confirmation
3. Delete each branch from the remote
4. Verify the remaining branches

### Option 2: Manual Deletion via Git Command Line

To delete the remote branches manually, a repository administrator with appropriate permissions should run:

```bash
git push origin --delete \
  copilot/explain-git-clone-files \
  copilot/git-history-cleanup-docs \
  copilot/simplify-repo-file-structure \
  copilot/move-api-web-management \
  copilot/remove-unused-branches
```

Or delete them individually:

```bash
git push origin --delete copilot/explain-git-clone-files
git push origin --delete copilot/git-history-cleanup-docs
git push origin --delete copilot/simplify-repo-file-structure
git push origin --delete copilot/move-api-web-management
git push origin --delete copilot/remove-unused-branches
```

### Option 3: Delete via GitHub Web Interface

1. Go to https://github.com/lipish/hyadmin/branches
2. Find each branch listed above
3. Click the trash icon next to each branch to delete it

## Verification

After deletion, verify that only the expected branches remain:

```bash
git ls-remote --heads origin
```

Expected output should show:
- `refs/heads/main`
- `refs/heads/copilot/remove-unused-branches-again` (can be deleted after this PR is merged)
