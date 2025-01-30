# Contributing

## Getting started with development

### Setup

1. Clone the `mostlyai` repository
   ```bash
   git clone https://github.com/mostly-ai/mostlyai.git
   cd mostlyai
   ```
   - if you intend to create PRs, and don't have permissions to write directly to `mostlyai` repo,
   create a fork of `mostlyai` first and clone the fork instead

2. Install `uv`, if you don't have it already
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   - for more installation options, check [uv installation page](https://docs.astral.sh/uv/getting-started/installation/)

3. Initialize new virtual environment and install dependencies
   ```bash
   uv sync --frozen --extra local-cpu --python=3.10  # install dependencies into a new virtual environment
   source .venv/bin/activate                         # activate the virtual environment
   ```
   - use `uv sync --frozen --extra local-gpu --python=3.10`, if you intend to use GPU

4. Install pre-commit hooks
   ```bash
    pre-commit install
    ```

### Development workflow

1. Make sure you are on the clean, up-to-date `main` branch
   ```bash
   git checkout main
   git pull
   git reset --hard origin/main
   ```

2. Create a new branch for your feature or bug fix
   ```bash
    git checkout -b feature-branch
    ```

3. Make your changes
4. Run tests
   ```bash
   uv run -- pytest
   ```

5. Run pre-commit hooks
   ```bash
   uv run -- pre-commit
   ```

6. Commit your changes
    ```bash
    git add .
    git commit -m "feat: your feature description"
    ```
   - please follow [conventional commits](https://gist.github.com/qoomon/5dfcdf8eec66a051ecd85625518cfd13), when writing commit messages

7. Push your changes
    ```bash
    git push origin feature-branch
    ```

8. Go to GitHub and create a PR.
