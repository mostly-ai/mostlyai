#!/bin/bash
set -e  # Exit on first error
set -x  # Print commands as they execute (debugging)

# Install uv
pip install uv

# Set up virtual environment
uv venv

# Install dependencies
uv sync --extra dev --frozen

# Ensure ipykernel is available
uv run pip install ipykernel