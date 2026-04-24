# CLAUDE.md

Notes for AI coding agents working on this repo. Keep it short; add things here only when they would have saved real debugging time.

- **Local-mode training/generation runs in a subprocess; crashes are silent.** If a test hangs or a generator stays `in_progress`, the child process (`mostlyai/sdk/_local/cli.py`, spawned from `mostlyai/sdk/_local/routes.py`) has almost certainly died. Read `<home_dir>/generators/<id>/job.log` (or `<home_dir>/synthetic-datasets/<id>/job.log`) for the real traceback — it will not surface in pytest output.

- **CI does NOT install `[local]` / `[local-gpu]` from `uv.lock`.** `run-tests-cpu.yaml` / `run-tests-gpu.yaml` do `uv sync --frozen --only-group dev` then `uv pip install ".[local]" ...`, which re-resolves engine/qa/pandas/numpy/pyarrow fresh. `CONTRIBUTING.md` (`uv sync --frozen --extra local`) gives a different, locked set. To reproduce a CI-only failure locally, run CI's exact install sequence.

- **Watch for pandas 3.0 breakage in `mostlyai/sdk/_data/`.** `DataFrame.groupby(col).apply(func)` no longer includes the grouping column in `func`'s input or the result, and `include_groups` is gone. Grep `\.groupby\(.*\)\.apply\(` before bumping deps; in the pull pipeline any silent column drop there surfaces much later as a `KeyError` in `pull_utils.py::_hash_column` inside the training subprocess.
