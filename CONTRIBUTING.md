# Contributing to Synthetic Data SDK

Thanks for your interest in contributing to Synthetic Data SDK! Follow these guidelines to set up your environment and streamline your contributions.

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mostly-ai/mostlyai.git
   cd mostlyai
   ```
   If you don’t have direct write access to `mostlyai`, fork the repository first and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/mostlyai.git
   cd mostlyai
   ```

2. **Install `uv` (if not installed already)**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   For alternative installation methods, visit the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

3. **Create a virtual environment and install dependencies**:
   ```bash
   uv sync --frozen --extra local --python=3.10  # For CPU
   source .venv/bin/activate
   ```
   If using GPU, run:
   ```bash
   uv sync --frozen --extra local-gpu --python=3.10  # For GPU support
   source .venv/bin/activate
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. **Ensure your local `main` branch is up to date**:
   ```bash
   git checkout main
   git reset --hard origin/main
   git pull origin main
   ```

2. **Create a new feature or bugfix branch**:
   ```bash
   git checkout -b my-feature-branch
   ```

3. **Implement your changes.**

4. **Run tests and pre-commit hooks**:
   ```bash
   pytest
   pre-commit run
   ```

5. **Commit your changes with a descriptive message**:
   ```bash
   git add .
   git commit -m "feat: add a clear description of your feature"
   ```
   Follow the [Conventional Commits](https://gist.github.com/qoomon/5dfcdf8eec66a051ecd85625518cfd13) format.

6. **Push your changes**:
   ```bash
   git push origin my-feature-branch
   ```

7. **Open a pull request on GitHub.**
