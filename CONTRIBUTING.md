# Contributing to Synthetic Data SDK

Thanks for your interest in contributing to Synthetic Data SDK! Here are a few guidelines to help you get started:


## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/mostly-ai/mostlyai.git
    cd mostlyai
    ```
   If you intend to create pull requests and don't have permissions to write directly to the `mostlyai` repo,
   create a fork of `mostlyai` first and clone the fork instead:
    ```bash
    git clone https://github.com/<your-username>/mostlyai.git
    cd mostlyai
    ```

2. Install `uv`, if you don't have it already:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
   For more installation options, visit [uv installation page](https://docs.astral.sh/uv/getting-started/installation/).

3. Initialize new virtual environment and install dependencies:
    ```bash
    uv sync --frozen --extra local-cpu --python=3.10  # install dependencies into a new virtual environment
    source .venv/bin/activate                         # activate the virtual environment
    ```
   Execute `uv sync --frozen --extra local-gpu --python=3.10`, if you intend to use GPU.

4. Install pre-commit hooks:
    ```bash
    pre-commit install
    ```


## Development Workflow

1. Make sure you are on a clean, up-to-date `main` branch:
    ```bash
    git checkout main
    git reset --hard origin/main
    git pull origin main
    ```

2. Create a new branch for your feature or bugfix:
    ```bash
    git checkout -b my-feature-branch
    ```

3. Make your changes.

4. Run the tests & pre-commit hooks:
    ```bash
    pytest
    pre-commit run
    ```

5. Commit your changes with a clear and descriptive commit message:
    ```bash
    git add .
    git commit -m "feat: add description of your feature here"
    ```
   Follow [conventional commits](https://gist.github.com/qoomon/5dfcdf8eec66a051ecd85625518cfd13), when writing commit messages.


6. Push your changes:
    ```bash
    git push origin my-feature-branch
    ```

5. Go to GitHub and open a pull request.
