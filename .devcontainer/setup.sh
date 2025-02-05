#!/bin/bash
set -e
set -x

pip install uv
uv venv
uv sync --extra dev --frozen

source .venv/bin/activate
uv pip install --upgrade --force-reinstall pip ipykernel

# Register the kernel explicitly (optional, but ensures VS Code detects it)
python -m ipykernel install --user