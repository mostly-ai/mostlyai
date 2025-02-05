#!/bin/bash
set -e  # Exit immediately if a command fails
set -x  # Print commands as they execute (for debugging)

# Install uv package manager
pip install uv

# Set up virtual environment using uv
uv venv
uv sync --extra dev --frozen

# Activate the virtual environment
source .venv/bin/activate

# Ensure pip and ipykernel are installed and up-to-date
uv pip install --upgrade --force-reinstall pip ipykernel

# Register the Jupyter kernel explicitly
python -m ipykernel install --user --name=python3 --display-name "Python 3 (Dev Container)" --prefix=/home/vscode/.local