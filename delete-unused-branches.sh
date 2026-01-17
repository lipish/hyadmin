#!/bin/bash
#
# Script to delete unused remote branches from GitHub
# This script should be run by a repository administrator with appropriate permissions
#

set -e

echo "======================================================================"
echo "Deleting Unused Branches from GitHub"
echo "======================================================================"
echo ""

# List of branches to delete
BRANCHES=(
    "copilot/explain-git-clone-files"
    "copilot/git-history-cleanup-docs"
    "copilot/simplify-repo-file-structure"
    "copilot/move-api-web-management"
    "copilot/remove-unused-branches"
)

echo "The following branches will be deleted from origin:"
for branch in "${BRANCHES[@]}"; do
    echo "  - $branch"
done
echo ""

# Confirm before proceeding
read -p "Do you want to proceed with deletion? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Deleting branches..."
echo ""

# Delete each branch
for branch in "${BRANCHES[@]}"; do
    echo "Deleting: $branch"
    if git push origin --delete "$branch" 2>&1; then
        echo "✓ Successfully deleted: $branch"
    else
        echo "✗ Failed to delete: $branch (may already be deleted)"
    fi
    echo ""
done

echo "======================================================================"
echo "Branch deletion complete!"
echo "======================================================================"
echo ""
echo "Verifying remaining branches..."
git ls-remote --heads origin

echo ""
echo "Expected branches:"
echo "  - main"
echo "  - copilot/remove-unused-branches-again (can be deleted after PR is merged)"
