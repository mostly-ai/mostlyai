#!/bin/bash
set -e

# Check if uv is already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    pip install uv
    echo "uv installed successfully."
else
    echo "uv is already installed."
fi

# Use uv to install packages
echo "Installing packages with uv..."
uv sync --frozen

echo "Build completed successfully."
