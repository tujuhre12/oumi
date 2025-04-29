# Oumi Development Guide

## Build/Lint/Test Commands
- Setup: `make setup` (creates conda env, installs dependencies)
- Run all tests: `make test` 
- Run a single test: `pytest tests/path/to/test.py::test_function_name`
- Lint and check: `make check` (runs pre-commit hooks)
- Format code: `make format` (ruff formatter)
- Type check: `pre-commit run pyright` or `pre-commit run pyright --all-files`


## Code Style
- Follow Google Python Style Guide
- Use absolute imports only (no relative imports)
- Type annotations required for all functions
- Docstrings: Use Google-style with descriptive verbs ("Builds" not "Build")
- Use list, dict for type hints, instead of typing.List, typing.Dict
- Formatting: 88-character line limit
- Name code entities according to PEP8 conventions
- Error handling: Use specific exceptions with informative messages
- No wildcard imports (from x import *)

## CLI Argument Style
- CLI commands support both shorthand and longform arguments
- Shorthand arguments are expanded to their fully qualified equivalents
- Both forms can be used interchangeably, but are mutually exclusive for the same parameter
- Examples:
  - Shorthand: `--model llama3` expands to `--model.model_name llama3`
  - Shorthand: `--dataset alpaca` expands to `--data.train.datasets[0].dataset_name alpaca`
- To add new shorthand arguments, update the `SHORTHAND_MAPPINGS` dictionary in `cli_utils.py`

## Imports
Sort imports with isort (handled by pre-commit):
1. Standard library
2. Third-party packages
3. Oumi packages (oumi.*)