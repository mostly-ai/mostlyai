#!/bin/bash
# Install uv package manager
pip install uv

# Set up virtual environment using uv
uv venv

# Conditionally add `--extra local` if SDK_MODE is "local"
if [ "$SDK_MODE" == "local" ]; then
    echo "Running in LOCAL SDK mode"
    uv sync --extra local-cpu --extra dev --frozen
else
    echo "Running in CLIENT SDK mode"
    uv sync --extra dev --frozen
fi

# Activate the virtual environment
source .venv/bin/activate

# Ensure pip and ipykernel are installed and up-to-date
uv pip install --upgrade --force-reinstall pip jupyter ipywidgets ipykernel jupyter_contrib_nbextensions

# Register the Jupyter kernel explicitly
python -m ipykernel install --user --name=python3 --display-name "Python 3 (Dev Container)"