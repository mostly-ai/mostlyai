# GitHub Workflows

This directory uses a simple split:

- **Entrypoint workflows** (`push`, `pull_request`, or `workflow_dispatch`)
- **Reusable workflows** (`workflow_call`) that are referenced by entrypoints

## Entrypoints

- `ci.yaml` — main CI on `push`/`pull_request`
- `ci-gpu-manual.yaml` — manual GPU test trigger (`workflow_dispatch`)
- `release-step-1-bump.yaml` — manual release bump and GitHub release
- `release-step-2-publish.yaml` — PyPI publish, triggered by release completion or manual dispatch

## Reusable Workflows

- `ci-reusable-pre-commit.yaml`
- `ci-reusable-tests-cpu.yaml`
- `ci-reusable-tests-gpu.yaml`
- `ci-reusable-build-docker-image.yaml`
