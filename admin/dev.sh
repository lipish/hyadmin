#!/bin/bash

# Development script for Heyi Admin

set -e

echo "Starting Heyi Admin in development mode..."

# Install frontend dependencies if needed
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Start frontend in development mode (in background)
echo "Starting React development server..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

# Wait a moment for frontend to start
sleep 2

# Start Rust backend
echo "Starting Rust backend..."
export RUST_LOG=heyi_admin=debug
cargo run

# Cleanup: kill frontend when backend exits
kill $FRONTEND_PID 2>/dev/null || true