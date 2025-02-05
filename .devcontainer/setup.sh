#!/bin/bash
set -e  # Exit on error
set -x  # Debug mode

pip install uv
uv venv
uv sync --extra dev --frozen