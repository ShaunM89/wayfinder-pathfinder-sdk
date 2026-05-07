# Contributing to Pathfinder SDK

## Development Setup

```bash
# Clone the repo
git clone https://github.com/ShaunM89/wayfinder-pathfinder-sdk.git
cd wayfinder-pathfinder-sdk

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Running Tests

```bash
# All tests
pytest

# Unit tests only (fast)
pytest -q --ignore=tests/test_async.py --ignore=tests/test_batch.py

# With coverage
pytest --cov=pathfinder_sdk --cov-report=term-missing
```

## Code Quality

We use `ruff` for linting and formatting, and `mypy` for type checking.
These run automatically on every commit via pre-commit hooks, and in CI on every push.

```bash
# Manual lint check
ruff check src tests

# Manual format check
ruff format --check src tests

# Manual type check
mypy src/pathfinder_sdk
```

## Pre-commit Hooks

The pre-commit configuration runs:
- `ruff check --fix` (lint + auto-fix)
- `ruff format` (format)
- `ruff format --check` (format consistency)
- Fast unit tests (excludes slower integration-style tests)

To skip pre-commit temporarily (not recommended):
```bash
git commit --no-verify -m "your message"
```

## Submitting Changes

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes with tests
3. Ensure all tests pass: `pytest`
4. Ensure linting passes: `ruff check src tests && ruff format --check src tests`
5. Push and open a pull request

## Release Process

Releases are tagged with semantic versioning (`v0.1.0`, `v0.2.0`, etc.).
CI automatically builds and validates the wheel on every push to `main`.
