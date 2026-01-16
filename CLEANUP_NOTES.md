# Repository Cleanup Notes

## Current Status

We've removed large files from git tracking (~10MB of vendored dependencies and auto-generated files), but they still exist in the git history. This means:

- **New commits**: Won't include these files (repository stays clean going forward)
- **Existing history**: Still contains the large files (historical commits)
- **Clone size**: Will be reduced for shallow clones (`git clone --depth 1`)

## Files Removed from Tracking

1. **Perfetto amalgamated files** (~9.5MB)
   - `engine/csrc/perfetto/perfetto.h` (7.0MB)
   - `engine/csrc/perfetto/perfetto.cc` (2.5MB)

2. **Static web assets** (~400KB)
   - Bootstrap CSS and JS
   - jQuery

3. **Lock files** (~163KB)
   - `web/package-lock.json`

## Impact

- Files removed from current tree: 6 files
- Lines removed: 246,945 lines
- Size reduction for new clones: ~10MB
- Tracked files reduced: 155 → 150 files

## Future Cleanup (Optional)

To completely remove these files from git history and reduce clone size for full clones, you would need to rewrite git history. This is a **destructive operation** that requires coordination with all contributors:

### Option 1: BFG Repo Cleaner (Recommended)

```bash
# Install BFG Repo Cleaner
brew install bfg  # or download from https://rtyley.github.io/bfg-repo-cleaner/

# Clone a fresh copy
git clone --mirror https://github.com/lipish/hyadmin.git

# Remove large files from history
bfg --delete-files perfetto.h hyadmin.git
bfg --delete-files perfetto.cc hyadmin.git
bfg --delete-files bootstrap.min.css hyadmin.git
bfg --delete-files jquery.min.js hyadmin.git
bfg --delete-files bootstrap.bundle.min.js hyadmin.git
bfg --delete-files package-lock.json hyadmin.git

# Clean up
cd hyadmin.git
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push (DESTRUCTIVE - coordinate with team!)
git push --force
```

### Option 2: git filter-repo

```bash
# Install git-filter-repo
pip install git-filter-repo

# Create a fresh clone
git clone https://github.com/lipish/hyadmin.git
cd hyadmin

# Remove files from history
git filter-repo --path engine/csrc/perfetto/perfetto.h --invert-paths
git filter-repo --path engine/csrc/perfetto/perfetto.cc --invert-paths
git filter-repo --path engine/heyi/server/static/css/bootstrap.min.css --invert-paths
git filter-repo --path engine/heyi/server/static/js/jquery.min.js --invert-paths
git filter-repo --path engine/heyi/server/static/js/bootstrap.bundle.min.js --invert-paths
git filter-repo --path web/package-lock.json --invert-paths

# Force push (DESTRUCTIVE - coordinate with team!)
git push --force --all
```

### Important Notes

⚠️ **Warning**: History rewriting is destructive and requires:
- All team members to re-clone the repository
- Updating all PRs and forks
- Potential issues with CI/CD systems
- Loss of git history integrity

**Recommendation**: Only do this if the repository size is causing significant problems. For most use cases, the current cleanup (removing from tracking) is sufficient.

## Measuring Impact

To see the current repository size:

```bash
# Count git objects
git count-objects -vH

# Check repository size
du -sh .git

# Check largest objects in history
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  grep "^blob" | sort -k3 -n -r | head -20
```

## Shallow Clones (Recommended for Users)

Users who don't need the full history can use shallow clones:

```bash
# Clone only the latest commit (fastest)
git clone --depth 1 https://github.com/lipish/hyadmin.git

# Clone with limited history (e.g., last 10 commits)
git clone --depth 10 https://github.com/lipish/hyadmin.git
```

This significantly reduces clone time and disk usage since the large files only exist in older commits.
