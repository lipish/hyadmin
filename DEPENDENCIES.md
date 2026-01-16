# Dependencies Guide

This document explains how to obtain external dependencies that are excluded from git to keep the repository size small.

## Perfetto (Performance Tracing)

The Perfetto tracing library uses amalgamated header files that are auto-generated and very large (~9.5MB).

### Download Perfetto Headers

To build the project, you need to download the Perfetto amalgamated headers:

```bash
# Download perfetto.h and perfetto.cc
cd engine/csrc/perfetto/
wget https://raw.githubusercontent.com/google/perfetto/main/sdk/perfetto.h
wget https://raw.githubusercontent.com/google/perfetto/main/sdk/perfetto.cc
```

Alternatively, you can download from the official Perfetto releases:
- Visit: https://perfetto.dev/docs/instrumentation/tracing-sdk
- Download the amalgamated SDK files

## Static Web Assets (Bootstrap & jQuery)

The static web assets for the server UI are excluded from git. You have two options:

### Option 1: Use CDN (Recommended)

Modify `engine/heyi/server/static/index.html` and `status.html` to use CDN links:

```html
<!-- Bootstrap CSS -->
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">

<!-- jQuery -->
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>

<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
```

### Option 2: Download Locally

```bash
cd engine/heyi/server/static/

# Create directories
mkdir -p css js

# Download Bootstrap CSS
wget -O css/bootstrap.min.css https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css

# Download jQuery
wget -O js/jquery.min.js https://code.jquery.com/jquery-3.6.0.min.js

# Download Bootstrap JS
wget -O js/bootstrap.bundle.min.js https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js
```

## Web Frontend (package-lock.json)

The `package-lock.json` file is excluded to avoid merge conflicts. To install dependencies:

```bash
cd web/
npm install
```

This will regenerate the `package-lock.json` file automatically.

## Why These Files Are Excluded

- **Perfetto files**: 9.5MB of auto-generated code that can be easily downloaded
- **Static assets**: 400KB of minified libraries available via CDN or package managers
- **package-lock.json**: Can cause frequent merge conflicts and is automatically regenerated

By excluding these ~10MB of files, we reduce the repository size significantly, making clones faster and reducing git object count.

## Vendored Dependencies (Kept in Repository)

The following large vendored dependencies are kept in the repository because they are actively used in the build:

### llama.cpp (~1.5MB)
- `engine/csrc/operators/moe/llama.cpp/` - GGML library for efficient neural network inference
- Source: https://github.com/ggerganov/llama.cpp

### llamafile (~200KB)
- `engine/csrc/operators/moe/llamafile/` - Optimized matrix multiplication routines
- Source: https://github.com/Mozilla-Ocho/llamafile

These files are kept in the repository for now but could potentially be managed as git submodules in the future for easier updates.
