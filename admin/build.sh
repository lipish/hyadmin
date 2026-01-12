#!/bin/bash

# Build Heyi Admin (Rust + React)

set -e

echo "Building Heyi Admin..."

# Build React frontend
echo "Building React frontend..."
cd frontend
npm install
npm run build
cd ..

# Build Rust backend
echo "Building Rust backend..."
cargo build --release

echo "Build completed successfully!"
echo "Run with: cargo run --release"