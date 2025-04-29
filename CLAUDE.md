# Oumi Development Guide

## Commands
- **Build/Install**: `pip install -e ".[dev]"` or `make setup`
- **Lint**: `pre-commit run --all-files` or `make check`
- **Format**: `ruff format .` or `make format`
- **Typecheck**: `pyright` or `pre-commit run pyright --all-files`
- **Run tests**: `pytest tests/` or specific test: `pytest tests/path/to/test.py::test_function`
- **Test with coverage**: `make coverage`

## Code Style
- Follow Google's Python Style Guide
- Use absolute imports only (no relative imports)
- Use descriptive-style verbs for docstrings (e.g., "Builds" not "Build")
- Add type annotations to all functions
- Use Google-style docstrings
- For type hints, do not use typing.Dict, List, etc. Use the python types dict, list

## Naming & Organization
- Class names: PascalCase
- Function/method names: snake_case
- Constants: UPPER_SNAKE_CASE
- Organize imports: standard lib → third-party → oumi modules
  
## Error Handling
- Use specific exceptions when possible
- Add informative error messages
- Include Apache 2.0 license header in all new files

## Development Workflow
- Run pre-commit hooks before committing
- Add tests for new features and bug fixes
- Follow PR guidelines in CONTRIBUTING.md