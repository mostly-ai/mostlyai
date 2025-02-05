#!/bin/bash
set -e
set -x

pip install uv
uv venv
uv sync --extra dev --frozen

# Activate venv explicitly before installing ipykernel
source .venv/bin/activate
pip install --upgrade --force-reinstall ipykernel

# Register the kernel explicitly (optional, but ensures VS Code detects it)
python -m ipykernel install --user